# This file is part of the "Smartville" project.
# Copyright (c) 2024 University of Insubria
# Licensed under the Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# For the full text of the license, visit:
# https://www.apache.org/licenses/LICENSE-2.0

# Smartville is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License 2.0 for more details.

# You should have received a copy of the Apache License 2.0
# along with Smartville. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

# Additional licensing information for third-party dependencies
# used in this file can be found in the accompanying `NOTICE` file.

import os
import json
import time
import threading
import traceback
from collections import Counter

import torch


class FlowDataRecorder:
    """
    Buffers fully-labelled flow samples observed by TigerBrain.process_input
    and periodically flushes them to disk as torch shards, so an offline
    script can later replay the exact same tensors the controller saw online
    (see offline replay design notes).

    Deliberately records ONLY the ground-truth natural-language label
    (`element_class`) and never the derived `zda`/`test_zda` booleans:
    those depend on the curriculum state (current_knowledge['G1s'/'G2s'])
    at the moment of capture, which itself mutates whenever *that specific
    run's* agent buys CTI (see NewTigerEnvironment.perform_epistemic_action).
    Baking them into the dataset would permanently couple any future replay
    to one agent's purchase history. zda/test_zda must be (re)computed at
    replay time from whatever curriculum the replaying run is configured
    with, exactly as TigerBrain.get_zda_labels does online.

    Storage layout (one directory per collection run):
        <out_dir>/run_<timestamp>/
            manifest.json        -- written once, full config snapshot needed
                                     to reconstruct TigerBrain/NewTigerEnvironment
            shards_index.jsonl   -- one line appended per shard flushed,
                                     for fast inspection without loading shards
            shard_000000.pt      -- torch.save'd dict, see _flush_locked
            shard_000001.pt
            ...

    Space/performance choices (see also report discussion):
        - Samples are buffered in memory and written in one torch.save call
          per shard (amortized I/O, no per-sample disk writes).
        - packet_features are integral byte values in [0, 255]: stored as
          uint8 instead of float32 (4x smaller, lossless). flow_features
          are genuinely continuous (byte counts, durations) and stay float32.
        - No compression is applied: the controller is a real-time process
          and compression CPU cost would compete with flow-stats polling.
          Shards can be compressed externally (e.g. zstd) if disk space is
          the binding constraint.
    """

    MANIFEST_NAME = "manifest.json"
    INDEX_NAME = "shards_index.jsonl"

    def __init__(
        self,
        out_dir,
        logger,
        shard_size=2000,
        flush_interval_secs=30,
        use_packet_feats=False,
        use_node_feats=False,
        run_metadata=None,
    ):
        self.logger = logger
        self.shard_size = int(shard_size)
        self.flush_interval_secs = float(flush_interval_secs)
        self.use_packet_feats = bool(use_packet_feats)
        self.use_node_feats = bool(use_node_feats)

        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(out_dir, f"run_{self.run_id}")

        self._lock = threading.Lock()
        self._buffer_flow = []
        self._buffer_packet = []
        self._buffer_node = []
        self._buffer_labels = []
        self._buffer_flow_ids = []
        self._buffer_ticks = []

        self._shard_idx = 0
        self._total_samples_recorded = 0
        self._total_samples_buffered = 0
        self._last_flush_time = time.monotonic()
        self._closed = False

        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except Exception:
            self.logger.error(
                f"[DataRecorder] Could not create output dir {self.out_dir}: "
                f"{traceback.format_exc()}"
            )
            raise

        self._write_manifest(run_metadata or {})
        self.logger.info(
            f"[DataRecorder] Initialized. out_dir={self.out_dir} "
            f"shard_size={self.shard_size} flush_interval_secs={self.flush_interval_secs} "
            f"use_packet_feats={self.use_packet_feats} use_node_feats={self.use_node_feats}"
        )

    def _write_manifest(self, run_metadata):
        manifest_path = os.path.join(self.out_dir, self.MANIFEST_NAME)
        try:
            with open(manifest_path, "w") as f:
                json.dump(run_metadata, f, indent=2, default=str)
            self.logger.info(f"[DataRecorder] Wrote manifest to {manifest_path}")
        except Exception:
            self.logger.error(
                f"[DataRecorder] Failed to write manifest to {manifest_path}: "
                f"{traceback.format_exc()}"
            )

    def record(self, batch, element_classes, tick, flow_ids=None):
        """
        batch: object exposing .flow_features / .packet_features / .node_features
               tensors already extracted by TigerBrain (e.g. via
               stack_flow_tensors), with shape [N, ...] each.
        element_classes: list[str] of length N, ground-truth NL label per sample.
        tick: int, a monotonically increasing counter identifying when this
              batch was observed (preserves arrival order for replay).
        flow_ids: optional list[str] of length N, for traceability/debugging.
        """
        if self._closed:
            self.logger.warning("[DataRecorder] record() called after close(); ignoring batch.")
            return

        try:
            n = batch.flow_features.shape[0]
        except Exception:
            self.logger.error(
                f"[DataRecorder] record() received a malformed batch (no usable "
                f"flow_features tensor): {traceback.format_exc()}"
            )
            return

        if n != len(element_classes):
            self.logger.error(
                f"[DataRecorder] Sample count mismatch at tick={tick}: flow_features "
                f"has {n} rows but {len(element_classes)} labels were given. Dropping "
                f"this batch entirely to avoid misaligned records."
            )
            return

        with self._lock:
            try:
                self._buffer_flow.append(batch.flow_features.detach().to("cpu").to(torch.float32))

                if self.use_packet_feats:
                    if batch.packet_features is None:
                        self.logger.warning(
                            f"[DataRecorder] tick={tick}: use_packet_feats=True but "
                            f"batch.packet_features is None; this batch's packet "
                            f"features will be MISSING from the shard (shapes across "
                            f"samples in a shard must match -- skipping packet storage "
                            f"for the whole run is safer than partial/ragged data)."
                        )
                        self.use_packet_feats = False
                    else:
                        # packet bytes are integral in [0, 255] -> uint8 is lossless and 4x smaller than float32.
                        self._buffer_packet.append(batch.packet_features.detach().to("cpu").to(torch.uint8))

                if self.use_node_feats:
                    if batch.node_features is None:
                        self.logger.warning(
                            f"[DataRecorder] tick={tick}: use_node_feats=True but "
                            f"batch.node_features is None; node feature storage will "
                            f"be disabled for the remainder of this run."
                        )
                        self.use_node_feats = False
                    else:
                        self._buffer_node.append(batch.node_features.detach().to("cpu").to(torch.float32))

                self._buffer_labels.extend(element_classes)
                self._buffer_ticks.extend([int(tick)] * n)
                self._buffer_flow_ids.extend(flow_ids if flow_ids is not None else ["unknown"] * n)
                self._total_samples_buffered += n

                self.logger.debug(
                    f"[DataRecorder] tick={tick}: buffered {n} samples "
                    f"({self._total_samples_buffered}/{self.shard_size} in current shard). "
                    f"Labels seen so far in this shard: {dict(Counter(self._buffer_labels))}"
                )
            except Exception:
                self.logger.error(
                    f"[DataRecorder] Error while buffering batch at tick={tick}: "
                    f"{traceback.format_exc()}"
                )
                return

            time_elapsed = time.monotonic() - self._last_flush_time
            if self._total_samples_buffered >= self.shard_size:
                self._flush_locked(reason=f"shard_size ({self.shard_size}) reached")
            elif self._total_samples_buffered > 0 and time_elapsed >= self.flush_interval_secs:
                self._flush_locked(reason=f"flush_interval_secs ({self.flush_interval_secs}s) elapsed")

    def flush(self):
        """Force a flush of whatever is currently buffered (e.g. on shutdown)."""
        with self._lock:
            if self._total_samples_buffered > 0:
                self._flush_locked(reason="manual flush")
            else:
                self.logger.info("[DataRecorder] flush() called but buffer is empty; nothing to write.")

    def _flush_locked(self, reason):
        """Must be called while holding self._lock."""
        n = self._total_samples_buffered
        if n == 0:
            return

        shard_name = f"shard_{self._shard_idx:06d}.pt"
        shard_path = os.path.join(self.out_dir, shard_name)
        start_time = time.monotonic()

        try:
            flow_features = torch.cat(self._buffer_flow, dim=0)
            packet_features = torch.cat(self._buffer_packet, dim=0) if self._buffer_packet else None
            node_features = torch.cat(self._buffer_node, dim=0) if self._buffer_node else None

            if flow_features.shape[0] != n:
                self.logger.error(
                    f"[DataRecorder] Internal inconsistency before writing {shard_name}: "
                    f"expected {n} buffered samples but concatenated flow_features has "
                    f"{flow_features.shape[0]} rows. Writing anyway, but downstream "
                    f"tensors/labels/ticks may be misaligned -- investigate immediately."
                )

            shard = {
                "flow_features": flow_features,
                "packet_features": packet_features,
                "node_features": node_features,
                "element_classes": list(self._buffer_labels),
                "flow_ids": list(self._buffer_flow_ids),
                "tick": list(self._buffer_ticks),
            }

            torch.save(shard, shard_path)
            elapsed_ms = (time.monotonic() - start_time) * 1000
            file_size_mb = os.path.getsize(shard_path) / (1024 * 1024)

            label_hist = dict(Counter(self._buffer_labels))
            tick_min, tick_max = min(self._buffer_ticks), max(self._buffer_ticks)

            self.logger.info(
                f"[DataRecorder] Flushed {shard_name} ({reason}): {n} samples, "
                f"{file_size_mb:.2f} MB, write took {elapsed_ms:.1f} ms, "
                f"ticks=[{tick_min},{tick_max}], labels={label_hist} -> {shard_path}"
            )

            self._append_index_entry({
                "shard": shard_name,
                "num_samples": n,
                "tick_min": tick_min,
                "tick_max": tick_max,
                "label_histogram": label_hist,
                "file_size_bytes": os.path.getsize(shard_path),
                "written_at": time.time(),
                "flow_feat_shape": list(flow_features.shape),
                "packet_feat_shape": list(packet_features.shape) if packet_features is not None else None,
                "node_feat_shape": list(node_features.shape) if node_features is not None else None,
            })

            self._total_samples_recorded += n
            self._shard_idx += 1

        except Exception:
            self.logger.error(
                f"[DataRecorder] FAILED to flush {shard_name} to {shard_path}: "
                f"{traceback.format_exc()}. These {n} buffered samples will be LOST "
                f"(buffer is cleared regardless of write outcome to avoid unbounded "
                f"memory growth while new samples keep arriving)."
            )
        finally:
            self._buffer_flow.clear()
            self._buffer_packet.clear()
            self._buffer_node.clear()
            self._buffer_labels.clear()
            self._buffer_flow_ids.clear()
            self._buffer_ticks.clear()
            self._total_samples_buffered = 0
            self._last_flush_time = time.monotonic()

    def _append_index_entry(self, entry):
        index_path = os.path.join(self.out_dir, self.INDEX_NAME)
        try:
            with open(index_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            self.logger.error(
                f"[DataRecorder] Failed to append shard index entry to {index_path}: "
                f"{traceback.format_exc()}"
            )

    def get_status_dict(self):
        """Lightweight stats for periodic logging/W&B reporting from the caller."""
        with self._lock:
            return {
                "data_collection/total_samples_recorded": self._total_samples_recorded,
                "data_collection/samples_buffered": self._total_samples_buffered,
                "data_collection/shards_written": self._shard_idx,
            }

    def close(self):
        self.flush()
        self._closed = True
        self.logger.info(
            f"[DataRecorder] Closed. Total samples recorded this run: "
            f"{self._total_samples_recorded} across {self._shard_idx} shard(s) in {self.out_dir}"
        )
