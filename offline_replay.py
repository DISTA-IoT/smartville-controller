#!/usr/bin/env python3
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

"""
Offline replay/training entry point for FlowDataRecorder output
(see data_recorder.py / tiger_brain_new.py's data_collection_mode).

Loads a collection run's manifest.json, reconstructs a TigerBrain instance
directly (no FastAPI/POX/uvicorn involved), then replays the recorded
shards in arrival order through the *exact same* downstream code path
(TigerBrain._process_batch) that the live online run used -- so this is
not a re-derivation of training logic, it is the original logic fed
recorded data instead of live flows.

Run this directly inside the controller container (or anywhere with read
access to the data_collection_dir mount and the smartController package on
PYTHONPATH), e.g.:

    python3 offline_replay.py /pox/pox/smartController/tiger_data_collection/run_20250101_120000

Logging is intentionally verbose ("obsessive") at every stage: manifest
reconstruction, kwarg overrides applied, per-shard load, per-tick batch
replay, label histograms, knowledge-base growth, and any malformed
shard/sample that gets skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import types
from collections import Counter

import torch
import yaml

# Under POX, this repo's checkout directory is itself named/mounted as
# "smartController", which is why every internal module here imports its
# siblings as `smartController.X` (e.g. tiger_brain_new.py's own imports).
# Running this script directly (no POX) means no such package exists on
# sys.path, so register a fake one whose __path__ points at this directory.
# That lets tiger_brain_new.py's unmodified `smartController.X` imports
# resolve exactly as they do under POX.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if "smartController" not in sys.modules:
    _smart_controller_pkg = types.ModuleType("smartController")
    _smart_controller_pkg.__path__ = [_THIS_DIR]
    sys.modules["smartController"] = _smart_controller_pkg

from smartController.tiger_brain_new import TigerBrain
from smartController.wandb_tracker import WandBTracker

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "shards_index.jsonl"


def build_logger(level):
    logger = logging.getLogger("offline_replay")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def load_manifest(run_dir, logger):
    manifest_path = os.path.join(run_dir, MANIFEST_NAME)
    logger.info(f"[offline_replay] Loading manifest from {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    logger.info(
        f"[offline_replay] Manifest loaded: keys={sorted(manifest.keys())} "
        f"device={manifest.get('device')} use_packet_feats={manifest.get('use_packet_feats')} "
        f"use_node_feats={manifest.get('use_node_feats')} hidden_size={manifest.get('hidden_size')} "
        f"inference_model_variant={manifest.get('inference_model_variant')}"
    )
    return manifest


def load_index(run_dir, logger):
    index_path = os.path.join(run_dir, INDEX_NAME)
    entries = []
    with open(index_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"[offline_replay] Malformed JSON on {INDEX_NAME}:{line_num}: {e} -- skipping entry.")
    logger.info(f"[offline_replay] Loaded {len(entries)} shard entries from {index_path}")
    return entries


def apply_set_overrides(kwargs, overrides, logger):
    """
    Applies generic --set dotted.path=value overrides on top of the manifest-derived
    kwargs, last (highest precedence -- can override anything build_kwargs already
    set, including the dedicated --agency/--load-pretrained/etc. flags). `value` is
    parsed with yaml.safe_load so plain CLI strings get sensible Python types
    (DuelingDDQN -> str, 0.01 -> float, true -> bool, null -> None, [1,2] -> list)
    without forcing the user to quote/escape anything beyond what their shell needs.
    """
    for raw in overrides or []:
        if "=" not in raw:
            raise ValueError(f"--set override {raw!r} is not of the form path.to.key=value")
        path, raw_value = raw.split("=", 1)
        path = path.strip()
        keys = path.split(".")
        if not path or any(not k for k in keys):
            raise ValueError(f"--set override {raw!r} has an empty/malformed key path {path!r}")
        value = yaml.safe_load(raw_value)

        node = kwargs
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        old = node.get(keys[-1])
        node[keys[-1]] = value
        logger.info(f"[offline_replay] Override (--set): {path} {old!r} -> {value!r}")


def build_kwargs(manifest, logger, args):
    """
    Reconstructs the kwargs dict TigerBrain.__init__ expects, from the
    manifest snapshot plus CLI overrides. Mirrors how dash.py's
    init_controller_args() assembles the live controller's kwargs (Hydra
    cfg as a dict, plus container_ips/ips_containers/traffic_dict/
    inference_model_variant), minus everything offline replay legitimately
    doesn't need (monitor_ip, POX/network wiring, etc).
    """
    kwargs = {
        "intrusion_detection": dict(manifest.get("intrusion_detection", {})),
        "neural_modules": dict(manifest.get("neural_modules", {})),
        "knowledge": dict(manifest.get("knowledge", {})),
        "rewards": dict(manifest.get("rewards", {})),
        "health": dict(manifest.get("health", {})),
        "wandb": dict(manifest.get("wandb", {})),
        "traffic_dict": manifest.get("traffic_dict", {}),
        "container_ips": manifest.get("container_ips", {}),
        "ips_containers": manifest.get("ips_containers", {}),
        "use_packet_feats": bool(manifest.get("use_packet_feats", False)),
        "node_features": bool(manifest.get("use_node_feats", False)),
        "device": args.device or manifest.get("device", "cpu"),
        "inference_model_variant": manifest.get("inference_model_variant", "default"),
        "logger": logger,
    }

    # --- Critical override: never let the replayed TigerBrain spin up a SECOND
    # FlowDataRecorder. The whole point of this script is to consume already-recorded
    # data, not write more of it.
    was_collecting = kwargs["intrusion_detection"].get("data_collection_mode", False)
    kwargs["intrusion_detection"]["data_collection_mode"] = False
    logger.info(
        f"[offline_replay] Override: intrusion_detection.data_collection_mode "
        f"{was_collecting} -> False (this run replays recorded data; it must not "
        f"record a second copy of itself)."
    )

    if args.pretrained_models_dir is not None:
        old = kwargs["intrusion_detection"].get("pretrained_models_dir")
        kwargs["intrusion_detection"]["pretrained_models_dir"] = args.pretrained_models_dir
        logger.info(f"[offline_replay] Override: intrusion_detection.pretrained_models_dir {old!r} -> {args.pretrained_models_dir!r}")

    if args.load_pretrained:
        old = kwargs["intrusion_detection"].get("pretrained_inference")
        kwargs["intrusion_detection"]["pretrained_inference"] = True
        logger.info(f"[offline_replay] Override: intrusion_detection.pretrained_inference {old} -> True")

    if args.agency:
        old = kwargs["intrusion_detection"].get("agency")
        kwargs["intrusion_detection"]["agency"] = True
        logger.info(f"[offline_replay] Override: intrusion_detection.agency {old} -> True")

    kwargs["intrusion_detection"].setdefault("save_models", True)
    if args.no_save:
        kwargs["intrusion_detection"]["save_models"] = False
        logger.info("[offline_replay] Override: intrusion_detection.save_models -> False (--no-save)")

    # wandb: forced disabled by default, since this is an offline replay over
    # already-collected data, not a fresh wandb-tracked experiment. step_counter
    # and the resource monitor are still exercised (WandBTracker is instantiated
    # for real), just with mode="disabled" so nothing hits the network.
    kwargs["wandb"]["wb_tracking"] = bool(args.wandb)
    if args.wandb_run_name:
        kwargs["wandb"]["wb_run_name"] = args.wandb_run_name
    else:
        kwargs["wandb"]["wb_run_name"] = f"{kwargs['wandb'].get('wb_run_name', 'run')}_offline_replay"
    kwargs["wandb"].setdefault("wb_project_name", "smartville_offline_replay")
    kwargs["wandb"].setdefault("wb_group_name", "offline_replay")
    kwargs["wandb"].setdefault("resource_monitor_interval_secs", 30)
    logger.info(
        f"[offline_replay] wandb config: wb_tracking={kwargs['wandb']['wb_tracking']} "
        f"wb_run_name={kwargs['wandb']['wb_run_name']} wb_project_name={kwargs['wandb']['wb_project_name']}"
    )

    apply_set_overrides(kwargs, args.set, logger)

    return kwargs


def check_feature_coverage(manifest, logger):
    """
    The shards only contain packet_features/node_features if BOTH the model
    config wanted them AND data_collection_use_packet_feats/_node_feats were
    true at capture time (see FlowDataRecorder construction in TigerBrain).
    If the model expects a feature stream that wasn't captured, replay would
    silently train on None tensors where the model expects real ones --
    that's a correctness bug, not something to paper over, so we abort loudly.
    """
    idd = manifest.get("intrusion_detection", {})
    model_wants_packet = bool(manifest.get("use_packet_feats", False))
    model_wants_node = bool(manifest.get("use_node_feats", False))
    captured_packet = bool(idd.get("data_collection_use_packet_feats", False)) and model_wants_packet
    captured_node = bool(idd.get("data_collection_use_node_feats", False)) and model_wants_node

    logger.info(
        f"[offline_replay] Feature coverage check: model_wants_packet_feats={model_wants_packet} "
        f"captured_packet_feats={captured_packet}; model_wants_node_feats={model_wants_node} "
        f"captured_node_feats={captured_node}"
    )

    problems = []
    if model_wants_packet and not captured_packet:
        problems.append(
            "model config has use_packet_feats=True but this run was captured with "
            "data_collection_use_packet_feats=False (or it was off in the model config "
            "at capture time) -- packet_features tensors are NOT in the shards. "
            "Replaying would feed the model None where it expects real packet tensors."
        )
    if model_wants_node and not captured_node:
        problems.append(
            "model config has use_node_feats=True but this run was captured with "
            "data_collection_use_node_feats=False -- node_features tensors are NOT in "
            "the shards. Replaying would feed the model None where it expects real node tensors."
        )
    if problems:
        for p in problems:
            logger.error(f"[offline_replay] FATAL feature coverage mismatch: {p}")
        raise RuntimeError("Recorded shards do not cover all feature streams the model requires; see logged errors above.")


def iter_tick_groups(shard, shard_name, logger):
    """
    Yields (tick, flow_features_slice, packet_features_slice, node_features_slice,
    element_classes_slice) for each contiguous run of identical tick values in
    this shard. A single FlowDataRecorder.record() call (one online tick) is
    always buffered atomically and flushed as a whole, so a tick's samples
    are never split across two shards -- grouping within each shard alone is
    sufficient to reconstruct the original per-tick batch boundaries.
    """
    ticks = shard.get("tick")
    flow_features = shard.get("flow_features")
    packet_features = shard.get("packet_features")
    node_features = shard.get("node_features")
    element_classes = shard.get("element_classes")

    if ticks is None or flow_features is None or element_classes is None:
        logger.error(f"[offline_replay] {shard_name}: missing tick/flow_features/element_classes -- skipping entire shard.")
        return

    n = len(ticks)
    if flow_features.shape[0] != n or len(element_classes) != n:
        logger.error(
            f"[offline_replay] {shard_name}: length mismatch (tick={n}, "
            f"flow_features={flow_features.shape[0]}, element_classes={len(element_classes)}) "
            f"-- skipping entire shard to avoid misaligned tensors/labels."
        )
        return

    start = 0
    while start < n:
        end = start + 1
        while end < n and ticks[end] == ticks[start]:
            end += 1
        yield (
            ticks[start],
            flow_features[start:end],
            packet_features[start:end] if packet_features is not None else None,
            node_features[start:end] if node_features is not None else None,
            element_classes[start:end],
        )
        start = end


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Path to a specific run_<timestamp>/ collection directory.")
    parser.add_argument("--device", default=None, help="Override manifest's device (e.g. 'cuda:0'). Default: use manifest's recorded device.")
    parser.add_argument("--pretrained-models-dir", default=None, help="Override intrusion_detection.pretrained_models_dir (where models load from / save to).")
    parser.add_argument("--load-pretrained", action="store_true", help="Force intrusion_detection.pretrained_inference=True (load pretrained weights before replaying).")
    parser.add_argument("--agency", action="store_true", help="Force intrusion_detection.agency=True, so the Decision Module acts/learns during replay even if the collection run recorded agency=False (e.g. the data_collection.yaml profile sets agency: false).")
    parser.add_argument(
        "--set", action="append", metavar="PATH.TO.KEY=VALUE", default=None,
        help="Override any manifest config value by dotted path before reconstructing TigerBrain, e.g. "
             "--set intrusion_detection.agent=DuelingDDQN --set neural_modules.hidden_size=128. "
             "Repeatable. Applied last, after --agency/--load-pretrained/etc., so it can override those too. "
             "Top-level path segment must be one of: intrusion_detection, neural_modules, knowledge, rewards, "
             "health, wandb. Value is parsed with yaml.safe_load (so true/1.0/null/[a,b] get real types; plain "
             "words like DuelingDDQN become strings).")
    parser.add_argument("--no-save", action="store_true", help="Disable model checkpoint saving for this replay run.")
    parser.add_argument(
        "--repetitions", type=int, default=8,
        help="Number of full passes over the (possibly --max-shards-truncated) set of recorded shards "
             "(default: 20). Each pass replays the shards in the same order, feeding the IM/DM "
             "--repetitions times the gradient steps over this fixed recorded dataset -- the offline "
             "equivalent of training for multiple epochs. --max-samples, if set, is a hard cap across "
             "the whole run (all repetitions combined), not per-repetition.")
    parser.add_argument("--max-shards", type=int, default=None, help="Stop after replaying this many shards.")
    parser.add_argument("--max-samples", type=int, default=None, help="Stop after replaying this many total samples (across all ticks/shards).")
    parser.add_argument("--wandb", action="store_true", help="Opt in to real (online) wandb tracking for this replay run. Default: disabled (offline, no network calls).")
    parser.add_argument("--wandb-run-name", default=None, help="Explicit wandb run name for this replay (default: '<original_run_name>_offline_replay').")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger = build_logger(getattr(logging, args.log_level))
    run_dir = os.path.abspath(args.run_dir)

    logger.info(f"[offline_replay] ===== Starting offline replay of {run_dir} =====")
    if not os.path.isdir(run_dir):
        logger.error(f"[offline_replay] run_dir does not exist or is not a directory: {run_dir}")
        return 1

    manifest = load_manifest(run_dir, logger)
    index_entries = load_index(run_dir, logger)
    if not index_entries:
        logger.error("[offline_replay] No shard entries found in shards_index.jsonl -- nothing to replay.")
        return 1

    try:
        check_feature_coverage(manifest, logger)
    except RuntimeError as e:
        logger.error(f"[offline_replay] Aborting before TigerBrain construction: {e}")
        return 1

    try:
        kwargs = build_kwargs(manifest, logger, args)
    except RuntimeError as e:
        logger.error(f"[offline_replay] Aborting: {e}")
        return 1

    logger.info("[offline_replay] Constructing WandBTracker...")
    wb_tracker = WandBTracker(kwargs)
    logger.info("[offline_replay] WandBTracker constructed. Constructing TigerBrain...")

    try:
        brain = TigerBrain(kwargs, wb_tracker=wb_tracker)
    except Exception:
        logger.error(f"[offline_replay] Failed to construct TigerBrain: {traceback.format_exc()}")
        wb_tracker.shutdown()
        return 1

    logger.info(
        f"[offline_replay] TigerBrain constructed. device={brain.device} "
        f"data_collection_mode={brain.data_collection_mode} agency={brain.agency} "
        f"online_evaluation={brain.online_evaluation} save_models_flag={brain.save_models_flag}"
    )

    shard_entries = index_entries if args.max_shards is None else index_entries[:args.max_shards]
    shard_entries = shard_entries[1:] # skip first entry (ugly packet repetition warm up during collection)
    logger.info(f"[offline_replay] Will replay {len(shard_entries)} of {len(index_entries)} shard(s) (max_shards={args.max_shards}).")
    logger.info(
        f"[offline_replay] --repetitions={args.repetitions}: the same {len(shard_entries)} shard(s) will be "
        f"replayed {args.repetitions} time(s) in full-pass order (no reshuffling between passes), giving "
        f"the IM/DM {args.repetitions}x the gradient steps over this recorded data."
    )

    overall_start = time.monotonic()
    total_samples_replayed = 0
    total_ticks_replayed = 0
    total_shards_replayed = 0
    total_shard_errors = 0
    aggregate_label_histogram = Counter()
    stop = False

    for epoch in range(args.repetitions):
        if stop:
            break
        logger.info(f"[offline_replay] ===== Repetition {epoch + 1}/{args.repetitions} over {len(shard_entries)} shard(s) =====")

        for shard_idx, entry in enumerate(shard_entries):
            if stop:
                break
            shard_name = entry.get("shard")
            shard_path = os.path.join(run_dir, shard_name)
            logger.info(
                f"[offline_replay] --- Repetition {epoch + 1}/{args.repetitions}, Shard {shard_idx + 1}/{len(shard_entries)}: "
                f"{shard_name} (expected {entry.get('num_samples')} samples, "
                f"ticks=[{entry.get('tick_min')},{entry.get('tick_max')}]) ---"
            )

            if not os.path.isfile(shard_path):
                logger.error(f"[offline_replay] {shard_name}: file missing on disk at {shard_path} -- skipping shard.")
                total_shard_errors += 1
                continue

            try:
                shard = torch.load(shard_path, map_location=brain.device, weights_only=False)
            except Exception:
                logger.error(f"[offline_replay] {shard_name}: torch.load failed: {traceback.format_exc()} -- skipping shard.")
                total_shard_errors += 1
                continue

            shard_start = time.monotonic()
            shard_samples = 0
            shard_ticks = 0

            for tick, flow_feats, packet_feats, node_feats, classes in iter_tick_groups(shard, shard_name, logger):
                if args.max_samples is not None and total_samples_replayed >= args.max_samples:
                    logger.info(f"[offline_replay] Reached --max-samples={args.max_samples}; stopping replay.")
                    stop = True
                    break

                n = flow_feats.shape[0]
                if args.max_samples is not None and total_samples_replayed + n > args.max_samples:
                    remaining = args.max_samples - total_samples_replayed
                    logger.info(
                        f"[offline_replay] Truncating tick={tick}'s batch from {n} to {remaining} "
                        f"samples to respect --max-samples={args.max_samples}."
                    )
                    flow_feats = flow_feats[:remaining]
                    packet_feats = packet_feats[:remaining] if packet_feats is not None else None
                    node_feats = node_feats[:remaining] if node_feats is not None else None
                    classes = classes[:remaining]
                    n = remaining

                try:
                    brain.process_input_from_record(flow_feats, packet_feats, node_feats, classes, tick=tick)
                except Exception:
                    logger.error(
                        f"[offline_replay] {shard_name} tick={tick}: process_input_from_record raised: "
                        f"{traceback.format_exc()} -- skipping this tick's batch, continuing replay."
                    )
                    continue

                shard_samples += n
                shard_ticks += 1
                total_samples_replayed += n
                total_ticks_replayed += 1
                aggregate_label_histogram.update(Counter(classes))

                if args.max_samples is not None and total_samples_replayed >= args.max_samples:
                    stop = True
                    break

            shard_elapsed = time.monotonic() - shard_start
            total_shards_replayed += 1
            logger.info(
                f"[offline_replay] {shard_name} done: {shard_ticks} tick-batches, {shard_samples} samples "
                f"replayed in {shard_elapsed:.2f}s. Running totals: samples={total_samples_replayed} "
                f"ticks={total_ticks_replayed} shards={total_shards_replayed} "
                f"known_classes={brain.current_known_classes_count} step_counter={wb_tracker.step_counter} "
                f"batch_processing_allowed={brain.batch_processing_allowed}"
            )

    overall_elapsed = time.monotonic() - overall_start
    logger.info(
        f"[offline_replay] ===== Replay complete in {overall_elapsed:.2f}s: "
        f"repetitions={args.repetitions} shards_replayed={total_shards_replayed} shard_errors={total_shard_errors} "
        f"ticks_replayed={total_ticks_replayed} samples_replayed={total_samples_replayed} "
        f"final_known_classes={brain.current_known_classes_count} "
        f"final_step_counter={wb_tracker.step_counter} "
        f"aggregate_label_histogram={dict(aggregate_label_histogram)} ====="
    )

    logger.info("[offline_replay] Shutting down TigerBrain (flushes wandb, stops monitor threads)...")
    brain.shutdown()
    logger.info("[offline_replay] Done.")

    return 0 if total_shard_errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
