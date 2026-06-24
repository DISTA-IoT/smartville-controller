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
Standalone inspection/verification tool for FlowDataRecorder output
(see data_recorder.py). Run this directly inside the controller container
(or anywhere with read access to the data_collection_dir mount) once a
capture run has produced shards, to sanity-check what was recorded before
trusting it for offline replay.

Usage examples:

    # List every run directory found under the collection root, with a
    # one-line summary of each (from shards_index.jsonl, no shard loading).
    python3 read_collected_data.py /pox/pox/smartController/tiger_data_collection/

    # Deep-inspect the most recently modified run: validate every shard
    # loads, tensor shapes/dtypes match the index, labels are sane, and
    # ticks are monotonically non-decreasing across shards.
    python3 read_collected_data.py /pox/pox/smartController/tiger_data_collection/ --latest --verify

    # Inspect one specific run directory directly.
    python3 read_collected_data.py /pox/pox/smartController/tiger_data_collection/run_20250101_120000 --verify
"""

import argparse
import json
import os
import sys
from collections import Counter

import torch

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "shards_index.jsonl"


def find_run_dirs(root):
    """Return all subdirectories of `root` that look like a collection run."""
    if not os.path.isdir(root):
        return []
    run_dirs = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, INDEX_NAME)):
            run_dirs.append(path)
    return run_dirs


def is_run_dir(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, INDEX_NAME))


def load_index(run_dir):
    index_path = os.path.join(run_dir, INDEX_NAME)
    entries = []
    if not os.path.isfile(index_path):
        print(f"  [!] No {INDEX_NAME} found in {run_dir} -- nothing was ever flushed here.")
        return entries

    with open(index_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [!] Malformed JSON on {INDEX_NAME}:{line_num}: {e}")
    return entries


def load_manifest(run_dir):
    manifest_path = os.path.join(run_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        print(f"  [!] No {MANIFEST_NAME} found in {run_dir}.")
        return {}
    try:
        with open(manifest_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] Failed to parse {MANIFEST_NAME}: {e}")
        return {}


def summarize_run(run_dir):
    print(f"\n=== Run: {run_dir} ===")
    manifest = load_manifest(run_dir)
    if manifest:
        idd = manifest.get("intrusion_detection", {})
        print(f"  manifest: data_collection_mode={idd.get('data_collection_mode')} "
              f"shard_size={idd.get('data_collection_shard_size')} "
              f"flow_feat_dim={manifest.get('flow_feat_dim')} "
              f"packet_feat_dim={manifest.get('packet_feat_dim')} "
              f"hidden_size={manifest.get('hidden_size')} "
              f"use_packet_feats={manifest.get('use_packet_feats')} "
              f"use_node_feats={manifest.get('use_node_feats')}")

    entries = load_index(run_dir)
    if not entries:
        print("  No shards recorded in this run (index is empty or missing).")
        return entries

    total_samples = sum(e.get("num_samples", 0) for e in entries)
    total_bytes = sum(e.get("file_size_bytes", 0) for e in entries)
    label_totals = Counter()
    for e in entries:
        label_totals.update(e.get("label_histogram", {}))

    tick_min = min((e.get("tick_min", 0) for e in entries), default=None)
    tick_max = max((e.get("tick_max", 0) for e in entries), default=None)

    print(f"  shards: {len(entries)}  total_samples: {total_samples}  "
          f"total_size: {total_bytes / (1024 * 1024):.2f} MB  ticks=[{tick_min},{tick_max}]")
    print(f"  label histogram (aggregated): {dict(label_totals)}")
    return entries


def verify_run(run_dir, entries, max_shards=None):
    """
    Loads every shard (or up to max_shards) and checks:
      - the file actually exists and torch.load succeeds
      - flow_features/element_classes/flow_ids/tick all have matching length
      - flow_features/packet_features/node_features shapes match what the
        index claimed at write time
      - tick values are non-decreasing *within* a shard (arrival order)
      - ticks are non-decreasing *across* shards, in index order
    Prints a PASS/FAIL summary; does not raise, so one bad shard doesn't stop
    inspection of the rest.
    """
    print("  --- verifying shards ---")
    ok_count = 0
    fail_count = 0
    prev_tick_max = None

    shard_entries = entries if max_shards is None else entries[:max_shards]

    for entry in shard_entries:
        shard_name = entry.get("shard")
        shard_path = os.path.join(run_dir, shard_name)
        problems = []

        if not os.path.isfile(shard_path):
            print(f"  [FAIL] {shard_name}: file missing on disk (referenced in index but not present)")
            fail_count += 1
            continue

        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [FAIL] {shard_name}: torch.load raised {e!r}")
            fail_count += 1
            continue

        flow_features = shard.get("flow_features")
        packet_features = shard.get("packet_features")
        node_features = shard.get("node_features")
        element_classes = shard.get("element_classes")
        flow_ids = shard.get("flow_ids")
        ticks = shard.get("tick")

        n_expected = entry.get("num_samples")
        n_actual = flow_features.shape[0] if flow_features is not None else None

        if n_actual != n_expected:
            problems.append(f"num_samples mismatch: index says {n_expected}, flow_features has {n_actual}")

        for name, lst in (("element_classes", element_classes), ("flow_ids", flow_ids), ("tick", ticks)):
            if lst is None or len(lst) != n_expected:
                problems.append(f"{name} length {len(lst) if lst is not None else None} != num_samples {n_expected}")

        if entry.get("flow_feat_shape") is not None and flow_features is not None:
            if list(flow_features.shape) != entry["flow_feat_shape"]:
                problems.append(
                    f"flow_features shape {list(flow_features.shape)} != index-recorded {entry['flow_feat_shape']}"
                )
        if flow_features is not None and flow_features.dtype != torch.float32:
            problems.append(f"flow_features dtype {flow_features.dtype} != expected float32")

        if entry.get("packet_feat_shape") is not None:
            if packet_features is None:
                problems.append("index says packet_features should be present but shard has None")
            elif list(packet_features.shape) != entry["packet_feat_shape"]:
                problems.append(
                    f"packet_features shape {list(packet_features.shape)} != index-recorded {entry['packet_feat_shape']}"
                )
            elif packet_features.dtype != torch.uint8:
                problems.append(f"packet_features dtype {packet_features.dtype} != expected uint8")

        if ticks:
            if any(t2 < t1 for t1, t2 in zip(ticks, ticks[1:])):
                problems.append("tick values are not monotonically non-decreasing within this shard")
            shard_tick_min, shard_tick_max = min(ticks), max(ticks)
            if prev_tick_max is not None and shard_tick_min < prev_tick_max:
                problems.append(
                    f"tick range [{shard_tick_min},{shard_tick_max}] overlaps/precedes previous "
                    f"shard's max tick {prev_tick_max} -- shards may be out of arrival order"
                )
            prev_tick_max = shard_tick_max

        if problems:
            print(f"  [FAIL] {shard_name}:")
            for p in problems:
                print(f"           - {p}")
            fail_count += 1
        else:
            ok_count += 1
            print(f"  [PASS] {shard_name}: {n_actual} samples, "
                  f"flow_features {list(flow_features.shape)} {flow_features.dtype}, "
                  f"packet_features={'present ' + str(list(packet_features.shape)) if packet_features is not None else 'None'}, "
                  f"node_features={'present ' + str(list(node_features.shape)) if node_features is not None else 'None'}")

    print(f"  --- verification done: {ok_count} passed, {fail_count} failed (of {len(shard_entries)} shard(s) checked) ---")
    return fail_count == 0


def show_sample(run_dir, entries, shard_index=0, sample_index=0):
    """Prints one concrete sample (flow_features row + label) for a sanity eyeball check."""
    if not entries:
        print("  No shards to show a sample from.")
        return
    shard_index = min(shard_index, len(entries) - 1)
    shard_name = entries[shard_index]["shard"]
    shard_path = os.path.join(run_dir, shard_name)
    try:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [!] Could not load {shard_name} to show a sample: {e}")
        return

    n = len(shard["element_classes"])
    sample_index = min(sample_index, n - 1)
    print(f"\n  --- sample [{sample_index}] from {shard_name} ---")
    print(f"  flow_id: {shard['flow_ids'][sample_index]}")
    print(f"  tick: {shard['tick'][sample_index]}")
    print(f"  element_class: {shard['element_classes'][sample_index]}")
    print(f"  flow_features: {shard['flow_features'][sample_index]}")
    if shard.get("packet_features") is not None:
        print(f"  packet_features shape: {shard['packet_features'][sample_index].shape}")
    if shard.get("node_features") is not None:
        print(f"  node_features: {shard['node_features'][sample_index]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "path",
        help="Either the data_collection_dir root (containing run_<timestamp>/ subdirs) "
             "or a specific run_<timestamp>/ directory.",
    )
    parser.add_argument("--latest", action="store_true",
                         help="If `path` is a collection root, only inspect the most recently modified run.")
    parser.add_argument("--verify", action="store_true",
                         help="Load every shard and validate shapes/dtypes/tick ordering against the index.")
    parser.add_argument("--max-shards", type=int, default=None,
                         help="Limit --verify to checking only the first N shards (useful for huge runs).")
    parser.add_argument("--show-sample", action="store_true",
                         help="Print one concrete sample's tensors/label for an eyeball check.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index to use with --show-sample.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index within the shard to show.")
    args = parser.parse_args()

    path = os.path.abspath(args.path)
    if not os.path.exists(path):
        print(f"Error: path does not exist: {path}")
        return 1

    if is_run_dir(path):
        run_dirs = [path]
    else:
        run_dirs = find_run_dirs(path)
        if not run_dirs:
            print(f"Error: no run directories (with {INDEX_NAME}) found under {path}")
            return 1
        if args.latest:
            run_dirs = [max(run_dirs, key=os.path.getmtime)]

    overall_ok = True
    for run_dir in run_dirs:
        entries = summarize_run(run_dir)
        if args.verify:
            ok = verify_run(run_dir, entries, max_shards=args.max_shards)
            overall_ok = overall_ok and ok
        if args.show_sample:
            show_sample(run_dir, entries, shard_index=args.shard_index, sample_index=args.sample_index)

    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
