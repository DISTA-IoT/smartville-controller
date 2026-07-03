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
CTI-acquisition probe: a thin, opinionated wrapper around ``offline_replay.py``
for the specific experiment of watching the anomaly-detection <-> CTI
transition per zero-day (G2) class.

It runs the *agency* replay (DM + IM game) with a periodic-CTI policy on a
deliberately LONG period, so the epistemic (buy-CTI) actions are spread far
apart and each G2's before/after story is easy to read on wandb. It also
neutralizes episode termination so the whole replay is ONE continuous episode
-- no bankruptcy reset, no budget-win reset, and a huge step horizon -- because
an episode reset would forget every bought class and reload the pretrained IM
weights, wiping out exactly the transition we want to observe.

Pair it with the per-G2 scalars added to tiger_brain_new.py:

  * unsupervised_scores/<g2>  -- prototypical anomaly-detection confidence of a
                                 G2's traffic BEFORE it is bought (per cluster
                                 decision, no aggregation).
  * supervised_scores/<g2>    -- closed-set classification confidence of a G2's
                                 traffic AFTER it is bought (per predicted-class
                                 group decision, no aggregation).
  * epistemic_delay/<g2>      -- DM steps between buying a G2 and the first later
                                 online tick whose batch actually carries that
                                 class again (CTI-acquisition latency).

Everything else (agent type, rewards, curriculum G2 list, model variant, ...)
comes from the collection run's manifest.json exactly as a normal
offline_replay does; this wrapper only forces the handful of knobs that make
the probe readable. Any of them can still be overridden with --set, which is
applied last.

Typical use (do the 7 epistemic actions and read the plots):

    python3 offline_cti_probe.py /pox/pox/smartController/tiger_data_collection/run_XXXX \
        --cti-period 1000 --wandb-run-name cti_probe_ddqn

By default it loads the pretrained IM (the 85%-AD checkpoint you want to start
from), tracks to wandb, and does NOT save model checkpoints (so the probe never
clobbers your good pretrained weights). Flip those with --no-load-pretrained,
--no-wandb, and --save respectively.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OFFLINE_REPLAY = os.path.join(_THIS_DIR, "offline_replay.py")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Path to a run_<timestamp>/ collection directory (same arg offline_replay.py takes).")

    parser.add_argument(
        "--cti-period", type=int, default=1000,
        help="Periodic-CTI period in DM steps: action 2 (buy CTI) is force-issued "
             "every this-many steps whenever an unbought G2 still exists. Larger = "
             "buys spread further apart = easier-to-read before/after plots. "
             "Default: 1000.")
    parser.add_argument(
        "--min-budget", type=float, default=-999999.0,
        help="Budget floor below which an episode would normally end (bankruptcy). "
             "Set very negative so the probe never resets on bankruptcy. Default: -999999.")
    parser.add_argument(
        "--max-episode-steps", type=int, default=100_000_000,
        help="Step horizon after which an episode ends. Set huge so the whole "
             "replay stays a single continuous episode (bought CTI knowledge and "
             "IM finetuning persist throughout). Default: 100000000.")
    parser.add_argument(
        "--repetitions", type=int, default=None,
        help="Full passes over the recorded shards (offline_replay's --repetitions). "
             "More passes = more DM steps = more room to fit all 7 periodic buys and "
             "their aftermath. Omit to use offline_replay's own default (20).")

    parser.add_argument("--device", default=None, help="Override the manifest's device (e.g. cuda:0).")
    parser.add_argument("--pretrained-models-dir", default=None, help="Override intrusion_detection.pretrained_models_dir.")

    # Sensible probe defaults, each individually flippable.
    parser.add_argument("--no-load-pretrained", dest="load_pretrained", action="store_false",
                        help="Do NOT force-load pretrained IM weights before replaying (default: load them).")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false",
                        help="Disable wandb tracking (default: enabled -- the probe's whole point is the new scalars).")
    parser.add_argument("--save", dest="no_save", action="store_false",
                        help="Allow model checkpoint saving (default: OFF, so the probe never overwrites your pretrained models).")
    parser.set_defaults(load_pretrained=True, wandb=True, no_save=True)

    parser.add_argument("--wandb-run-name", default="cti_probe", help="wandb run name for this probe. Default: cti_probe.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--set", action="append", metavar="PATH.TO.KEY=VALUE", default=None,
        help="Extra manifest override(s), passed straight through to offline_replay.py and "
             "applied LAST (so they win over this wrapper's forced knobs). Repeatable.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the offline_replay.py command that would run, then exit without running it.")

    args = parser.parse_args()

    if not os.path.isfile(_OFFLINE_REPLAY):
        parser.error(f"could not find offline_replay.py next to this script at {_OFFLINE_REPLAY}")

    cmd = [sys.executable, _OFFLINE_REPLAY, args.run_dir, "--agency", "--log-level", args.log_level]

    if args.load_pretrained:
        cmd.append("--load-pretrained")
    if args.no_save:
        cmd.append("--no-save")
    if args.wandb:
        cmd += ["--wandb", "--wandb-run-name", args.wandb_run_name]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.pretrained_models_dir is not None:
        cmd += ["--pretrained-models-dir", args.pretrained_models_dir]
    if args.repetitions is not None:
        cmd += ["--repetitions", str(args.repetitions)]

    # Forced knobs that make the probe a single, readable, non-terminating
    # episode driven by periodic CTI. All go through offline_replay's generic
    # --set (yaml-parsed), so ints/floats/bools land as real types.
    forced = {
        "intrusion_detection.cti_period": args.cti_period,
        "intrusion_detection.min_budget": args.min_budget,
        "intrusion_detection.max_episode_steps": args.max_episode_steps,
        # Never end an episode on the budget ceiling either -- one long episode.
        "intrusion_detection.disable_budget_win_termination": "true",
        # Make sure the other two scripted-CTI ablation modes are off, so
        # cti_period is unambiguously the policy in force.
        "intrusion_detection.greedy_cti": "false",
        "intrusion_detection.no_epistemic_actions": "false",
    }
    for key, value in forced.items():
        cmd += ["--set", f"{key}={value}"]

    # User --set overrides last, so they can override anything above.
    for override in (args.set or []):
        cmd += ["--set", override]

    print("[offline_cti_probe] Running:\n  " + " ".join(cmd) + "\n", flush=True)
    if args.dry_run:
        print("[offline_cti_probe] --dry-run: not executing.")
        return 0

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
