#!/usr/bin/env python3
"""Multi-seed agent-ablation sweep over a single recorded run, via offline_replay.py.

Offline counterpart to tiger/tests/seeded_agents.py. That script drives a
live controller through dash_cli.py (GNS3 topology, POX, traffic replay,
the works) one seeded run at a time. This script needs none of that: it
replays the *same* previously-recorded FlowDataRecorder run
(data_recorder.py / data_collection_and_offline_training_implementation_report.md)
once per (seed, agent, ablation-mode) combination, each time as a plain
`python3 offline_replay.py <run_dir> ...` subprocess in this repo. No
GNS3 topology, no POX, no live controller process, no `tiger` or other
sibling repo, no other container -- a recorded run directory and this
checkout of smartville-controller are sufficient.

Sweeps the value-learning (DQN-family) agents -- DQN, DDQN, DuelingDQN,
DuelingDDQN -- by default, across the same four mutually-exclusive
epistemic-action ablation modes described in drl_description.md /
tiger_brain_new.py's `_select_unknown_cluster_action`:

- "baseline":     learned policy decides action 2 (buy CTI) on its own,
                   i.e. greedy_cti=False, cti_period=-1, no_epistemic_actions=False
                   (the manifest's recorded defaults, left untouched).
- "no_epistemic": intrusion_detection.no_epistemic_actions=true -- any agent-chosen
                   action 2 is remapped to 1 (block); epistemic actions are
                   effectively disabled.
- "periodic_cti": intrusion_detection.cti_period=<N> -- action is hard-forced to 2
                   every N steps, otherwise the agent is queried but its own 2's
                   are remapped to 1.
- "greedy_cti":   intrusion_detection.greedy_cti=true -- action is forced to 2
                   whenever an unbought G2 class is available, otherwise the
                   agent is queried but its own 2's are remapped to 1.

Each (seed, agent, mode) combination is one offline_replay.py invocation:
the manifest's recorded config is taken as the base, and only
intrusion_detection.agent / .seed / the one active ablation knob /
wandb.wb_group_name are overridden via offline_replay.py's generic --set
PATH.TO.KEY=VALUE flag. --agency is always forced on (the whole point of
this sweep is to exercise the Decision Module, even if the recorded run
was captured with intrusion_detection.agency=false, as the
data_collection.yaml profile sets).

Run-name convention (deliberately different from tiger/tests/seeded_agents.py):
wandb's run name is always exactly the agent string (e.g. "DQN",
"DuelingDDQN"), regardless of ablation mode, so results group by agent in
W&B's "group by run name" view. The seed and ablation mode are still fully
recorded -- in the console label (agent-mode-seedN), in wandb.wb_group_name,
and inside each run's own logged config (intrusion_detection.seed/
.no_epistemic_actions/.cti_period/.greedy_cti) -- only the run *name* is
collapsed onto the agent. One consequence: since offline_replay.py / TigerBrain
key checkpoint filenames by wandb.wb_run_name, runs that share an agent but
differ only in ablation mode or seed will write checkpoints to the same path,
each overwriting the previous one in this sweep (the IM itself is still
trained from scratch / reloaded fresh per run per the manifest's
pretrained_inference setting -- there's no cross-run weight leakage, only
last-checkpoint-wins on disk). Pass --no-save if you don't need checkpoints
at all for a sweep like this.

Sweep order matches tiger/tests/seeded_agents.py: seeds are the OUTERMOST
loop, (agent, mode) the innermost -- so an interrupted sweep always leaves
every started seed with a complete set of configs, rather than uneven
coverage across seeds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_AGENTS = ["DQN", "DDQN", "DuelingDQN", "DuelingDDQN"]
DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_CTI_PERIOD = 10
DEFAULT_WANDB_GROUP_NAME = "offline-agents-seeded"

ABLATION_MODES = ["baseline", "no_epistemic", "periodic_cti", "greedy_cti"]


def ablation_set_overrides(mode: str, cti_period: int) -> list[str]:
    """
    Returns the --set PATH=VALUE strings for offline_replay.py that realize
    one ablation mode, mirroring tiger/tests/seeded_agents.py's
    ablation_overrides(). "baseline" returns none, leaving the manifest's
    recorded defaults (greedy_cti=False, cti_period=-1, no_epistemic_actions=false)
    in place.
    """
    if mode == "baseline":
        return []
    if mode == "no_epistemic":
        return ["intrusion_detection.no_epistemic_actions=true"]
    if mode == "periodic_cti":
        return [f"intrusion_detection.cti_period={cti_period}"]
    if mode == "greedy_cti":
        return ["intrusion_detection.greedy_cti=true"]
    raise ValueError(f"Unknown ablation mode: {mode!r}")


def fail_loudly(message: str) -> None:
    """
    Abort the whole sweep immediately and unmissably, instead of silently
    skipping a broken run or burning through the rest of a multi-hour sweep
    on top of a config that's already known to be invalid.
    """
    banner = "!" * 78
    print(f"\n{banner}\n[ABORT] {message}\n{banner}\n", file=sys.stderr, flush=True)
    sys.exit(1)


def run_one(
    offline_replay_path: Path,
    run_dir: Path,
    agent: str,
    mode: str,
    seed: int,
    cti_period: int,
    group_name: str,
    wandb_enabled: bool,
    passthrough_args: list[str],
) -> None:
    # label is just for console/log messages -- fully descriptive (agent,
    # mode, seed). The wandb run name (set below) is intentionally just the
    # agent string, per this script's grouping convention -- see module docstring.
    label = f"{agent}-{mode}-seed{seed}"
    print(f"\n[step] Replaying {label} ...", flush=True)

    overrides = [
        f"intrusion_detection.agent={agent}",
        f"intrusion_detection.seed={seed}",
        *ablation_set_overrides(mode, cti_period),
        f"wandb.wb_group_name={group_name}",
    ]

    cmd = [sys.executable, str(offline_replay_path), str(run_dir), "--agency"]
    if wandb_enabled:
        cmd += ["--wandb", "--wandb-run-name", agent]
    for override in overrides:
        cmd += ["--set", override]
    cmd += passthrough_args

    print(f"[step] Running: {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"[ok] {label} replay completed cleanly.", flush=True)
    elif result.returncode == 2:
        print(
            f"[warn] {label}: offline_replay.py exited 2 (one or more shards failed to "
            "load/parse; the run is still mostly usable but inspect its logs).",
            flush=True,
        )
    else:
        fail_loudly(f"{label}: offline_replay.py exited with code {result.returncode}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one recorded FlowDataRecorder run through multiple seeded "
            "agent x epistemic-action-ablation configs via offline_replay.py, "
            "for statistical significance without a live controller/topology."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir", help="Path to a specific recorded run_<timestamp>/ directory (passed straight through to offline_replay.py).")
    parser.add_argument(
        "--agents", nargs="+", default=DEFAULT_AGENTS,
        help=f"Agent types to sweep (default: {DEFAULT_AGENTS}). Each becomes intrusion_detection.agent "
             "AND the wandb run name for its runs.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help=f"Seeds to run per agent (default: {DEFAULT_SEEDS}). Outermost loop: every "
             "(agent, ablation-mode) config is replayed once per seed before moving to the next seed.",
    )
    parser.add_argument(
        "--ablation-modes", nargs="+", default=ABLATION_MODES, choices=ABLATION_MODES,
        help=f"Epistemic-action ablation modes to sweep (default: {ABLATION_MODES}).",
    )
    parser.add_argument(
        "--cti-period", type=int, default=DEFAULT_CTI_PERIOD,
        help=f"Value of intrusion_detection.cti_period used by the 'periodic_cti' ablation mode "
             f"(default: {DEFAULT_CTI_PERIOD}).",
    )
    parser.add_argument(
        "--wandb-group-name", default=DEFAULT_WANDB_GROUP_NAME,
        help=f"wandb.wb_group_name shared by every run in this sweep, for later aggregation "
             f"(default: {DEFAULT_WANDB_GROUP_NAME!r}).",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable real wandb tracking for this sweep (default: enabled, with --wandb-run-name "
             "set to the agent string, since grouping by run_name is the point of this script).",
    )
    parser.add_argument(
        "--offline-replay-path", type=Path,
        default=Path(__file__).resolve().parent / "offline_replay.py",
        help="Path to offline_replay.py (default: alongside this script).",
    )
    # Passthrough flags, forwarded verbatim to every offline_replay.py invocation.
    parser.add_argument("--device", default=None, help="Forwarded to offline_replay.py --device.")
    parser.add_argument("--pretrained-models-dir", default=None, help="Forwarded to offline_replay.py --pretrained-models-dir.")
    parser.add_argument("--load-pretrained", action="store_true", help="Forwarded to offline_replay.py --load-pretrained.")
    parser.add_argument("--no-save", action="store_true", help="Forwarded to offline_replay.py --no-save.")
    parser.add_argument("--max-shards", type=int, default=None, help="Forwarded to offline_replay.py --max-shards (e.g. for a smoke-test sweep).")
    parser.add_argument("--max-samples", type=int, default=None, help="Forwarded to offline_replay.py --max-samples.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Forwarded to offline_replay.py --log-level.")

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"run_dir not found: {run_dir}")
        return 1

    offline_replay_path = args.offline_replay_path.resolve()
    if not offline_replay_path.exists():
        print(f"offline_replay.py not found: {offline_replay_path}")
        return 1

    passthrough_args: list[str] = ["--log-level", args.log_level]
    if args.device is not None:
        passthrough_args += ["--device", args.device]
    if args.pretrained_models_dir is not None:
        passthrough_args += ["--pretrained-models-dir", args.pretrained_models_dir]
    if args.load_pretrained:
        passthrough_args.append("--load-pretrained")
    if args.no_save:
        passthrough_args.append("--no-save")
    if args.max_shards is not None:
        passthrough_args += ["--max-shards", str(args.max_shards)]
    if args.max_samples is not None:
        passthrough_args += ["--max-samples", str(args.max_samples)]

    total_runs = len(args.seeds) * len(args.agents) * len(args.ablation_modes)
    run_idx = 0
    for seed in args.seeds:
        for agent in args.agents:
            for mode in args.ablation_modes:
                run_idx += 1
                print(
                    f"\n===== Run {run_idx}/{total_runs}: seed={seed} agent={agent} mode={mode} =====",
                    flush=True,
                )
                run_one(
                    offline_replay_path=offline_replay_path,
                    run_dir=run_dir,
                    agent=agent,
                    mode=mode,
                    seed=seed,
                    cti_period=args.cti_period,
                    group_name=args.wandb_group_name,
                    wandb_enabled=not args.no_wandb,
                    passthrough_args=passthrough_args,
                )

    print("\n[done] Offline seeded sweep completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
