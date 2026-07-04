#!/usr/bin/env python3
"""Multi-GPU parallel IM-pretraining sweep, via offline_replay.py.

Sibling of offline_seeded_agents_parallel.py, but for a completely different
experiment: that script exercises the Decision Module (DM) -- the RL agent
that decides accept/block/buy-CTI -- by forcing --agency on and sweeping
(seed, agent, ablation-mode) combinations. This script does the opposite. It
trains the Inference Module (IM) -- the classifier/confidence-decoder/
kernel-regression stack that does closed-set classification, anomaly
detection and clustering -- FROM SCRATCH on a recorded run, with:

- NO agency: intrusion_detection.agency is forced to false (--agency is
  never passed to offline_replay.py, and an explicit
  --set intrusion_detection.agency=false override is added so a run_dir
  recorded WITH agency=true, e.g. one captured for the seeded-agents sweep,
  can still be reused here -- the DM/mitigation_agent still exists and still
  gets exercised by TigerBrain's bookkeeping, it just never decides an
  action; every accept/block/CTI-buy decision it would normally make is
  skipped, so no DM learning signal is generated here at all). This is
  always true regardless of what you sweep below -- there is no DM
  agent/ablation-mode axis in this script, on purpose.
- NO pretrained-inference loading: --load-pretrained is never passed, and an
  explicit --set intrusion_detection.pretrained_inference=false override is
  added (again, so a run_dir recorded with pretrained_inference=true doesn't
  silently load old weights instead of training from a fresh init) -- the
  classifier/confidence_decoder are constructed from scratch and their only
  training signal is this replay.
- online_evaluation forced ON (intrusion_detection.online_evaluation=true):
  TigerBrain only ever calls check_progress_and_save() -- the thing that
  actually writes classifier/confidence_decoder checkpoints to disk -- from
  inside its periodic online-evaluation block (see
  tiger_brain_new.py:_process_batch's online_evaluation branch). Without
  this, --save would be a no-op: the IM would train for the whole sweep and
  never touch disk. --online-eval-step-freq/--online-eval-rounds let you
  tune how often/how thoroughly that evaluation (and thus save-checkpointing)
  runs, without having to hand-edit the recorded manifest.
- Checkpoint saving ON by default (the entire point of this script is
  producing pretrained_models_dir/*.pt files for later --load-pretrained
  runs), unlike offline_seeded_agents(_parallel).py, which defaults saving
  OFF. Pass --no-save only if you want a dry run of the training itself
  without writing checkpoints.

What IS swept here: --variants x --seeds. Each --variants entry names a
module under im_models/ (e.g. "default", "mahalanobis", "optim" --
see smartController.im_models and TigerBrain.load_models_from_variant() in
tiger_brain_new.py) and is passed through as an
--set inference_model_variant=<variant> override, which is how TigerBrain
now picks which ASAP architecture (classifier/confidence-decoder/
kernel-regression-loss classes) to import -- see the "Load IM neural module
variants from an in-repo package" change; the old exec()'d-JSON-source
mechanism is gone, so there is no longer a models-source-file path to point
at. --variants is required and validated against the im_models/ directory
next to this script before anything is launched (a typo'd variant name would
otherwise only surface as an ImportError deep inside every single GPU job).
--seeds repeats each variant with multiple seeds, since TigerBrain seeds
both the classifier's/confidence_decoder's weight init and the
training-time sampling/replay-buffer draw order from
intrusion_detection.seed (see tiger_brain_new.py's
`torch.manual_seed(self.seed)` right before init_inference_neural_modules()
runs) -- multiple seeds give you a real read on how much of a given
variant's learned performance is signal vs. init/ordering luck, the same
statistical-significance rationale offline_seeded_agents.py applies to the
DM's agent/mode sweep.

Checkpoint naming / collisions: unlike the DM ablation sweep (where
checkpoints are keyed only by wandb.wb_run_name and wb_run_name() deliberately
collapses several (agent, mode) combos onto the same run name -- see
offline_seeded_agents.py's module docstring), the IM's saved filenames are
ALSO suffixed with the live wandb_run_name (tiger_brain_new.py's
check_progress_and_save: f"{classifier_path[:-3]}single_{wandb_run_name}.pt"
etc.) and this script gives every (variant, seed) pair its own run name
(`im_pretrain-<variant>-seed<seed>`, see wb_run_name() below), so different
variants and different seeds in the same sweep do NOT overwrite each other's
checkpoints -- each pair gets its own
`..._single_im_pretrain-<variant>-seed<seed>.pt` / `..._coupled_...pt` files
under --pretrained-models-dir. Re-running this script twice with the same
(variant, seed) pair does still overwrite that pair's files, same as any
other checkpointed training script.

Run this directly inside a checkout of this repo, pointed at a recorded run
directory:

    python3 offline_pretrain_im_parallel.py /data/tiger_data_collection/run_20250101_120000 \\
        --variants default mahalanobis optim --seeds 1 2 3

GPU autodetection, per-job log files, Ctrl-C behaviour, and the summary table
are identical to offline_seeded_agents_parallel.py -- see that script's module
docstring for the full rationale; this script reuses its detect_gpu_ids()
directly (`import offline_seeded_agents_parallel as par`) rather than
re-deriving it. A job that fails (offline_replay.py exit code not in {0, 2})
is logged and recorded as failed, but the sweep keeps going; the script's own
exit code is 1 if any job failed (or had shard errors), 0 if every job
replayed cleanly.
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import offline_seeded_agents_parallel as par

DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_WANDB_GROUP_NAME = "offline-im-pretrain"
DEFAULT_PRETRAINED_MODELS_DIR = "./im_pretrained_models/"
IM_MODELS_DIR = Path(__file__).resolve().parent / "im_models"


def available_variants() -> list[str]:
    """
    Lists the inference_model_variant names TigerBrain can actually load --
    one per im_models/<name>.py module (see im_models/__init__.py and
    TigerBrain.load_models_from_variant()). Used to fail fast on a typo'd
    --variants entry instead of discovering it via an ImportError inside a
    GPU job's log file.
    """
    if not IM_MODELS_DIR.is_dir():
        return []
    return sorted(
        p.stem for p in IM_MODELS_DIR.glob("*.py")
        if p.stem != "__init__"
    )


def wb_run_name(variant: str, seed: int) -> str:
    return f"im_pretrain-{variant}-seed{seed}"


@dataclass(frozen=True)
class Job:
    variant: str
    seed: int

    @property
    def label(self) -> str:
        return wb_run_name(self.variant, self.seed)


@dataclass
class JobResult:
    job: Job
    gpu_id: int
    returncode: int
    elapsed_seconds: float
    log_path: Path


def build_jobs(variants: list[str], seeds: list[int]) -> list[Job]:
    # Variants outermost, seeds innermost -- so an interrupted sweep's
    # "first N jobs done" reads as full seed coverage of the earliest
    # variants before later ones, mirroring offline_seeded_agents_parallel's
    # seed-outermost convention for its own two swept axes.
    return [Job(variant=variant, seed=seed) for variant in variants for seed in seeds]


def job_overrides(
    job: Job,
    group_name: str,
    online_eval_step_freq: int | None,
    online_eval_rounds: int | None,
) -> list[str]:
    overrides = [
        f"intrusion_detection.seed={job.seed}",
        f"inference_model_variant={job.variant}",
        # Belt-and-suspenders: --agency/--load-pretrained are never passed to
        # offline_replay.py either, but a run_dir recorded with agency=true or
        # pretrained_inference=true (e.g. one captured for the seeded-agents
        # sweep) would otherwise silently keep those manifest values -- this
        # is an IM-from-scratch, no-agency pretraining run regardless of what
        # the recording happened to have set.
        "intrusion_detection.agency=false",
        "intrusion_detection.pretrained_inference=false",
        "intrusion_detection.ad_loss_backprop_to_encoder=true",
        # Required for check_progress_and_save() to ever run -- see module
        # docstring. Harmless if the manifest already had it on.
        "intrusion_detection.online_evaluation=true",
        f"wandb.wb_group_name={group_name}",
    ]
    if online_eval_step_freq is not None:
        overrides.append(f"intrusion_detection.online_eval_step_freq={online_eval_step_freq}")
    if online_eval_rounds is not None:
        overrides.append(f"intrusion_detection.online_evaluation_rounds={online_eval_rounds}")
    return overrides


def run_job(
    job: Job,
    gpu_queue: "queue.Queue[int]",
    offline_replay_path: Path,
    run_dir: Path,
    pretrained_models_dir: Path,
    group_name: str,
    online_eval_step_freq: int | None,
    online_eval_rounds: int | None,
    wandb_enabled: bool,
    passthrough_args: list[str],
    logs_dir: Path,
    print_lock: threading.Lock,
) -> JobResult:
    gpu_id = gpu_queue.get()
    try:
        overrides = job_overrides(job, group_name, online_eval_step_freq, online_eval_rounds)
        cmd = [
            sys.executable, str(offline_replay_path), str(run_dir),
            "--device", f"cuda:{gpu_id}",
            "--pretrained-models-dir", str(pretrained_models_dir),
        ]
        if wandb_enabled:
            cmd += ["--wandb", "--wandb-run-name", wb_run_name(job.variant, job.seed)]
        for override in overrides:
            cmd += ["--set", override]
        cmd += passthrough_args

        log_path = logs_dir / f"{job.label}.log"
        with print_lock:
            print(f"[start] {job.label} on cuda:{gpu_id} (log: {log_path})", flush=True)

        start = time.monotonic()
        with open(log_path, "w") as log_file:
            log_file.write(f"# command: {' '.join(cmd)}\n\n")
            log_file.flush()
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - start

        with print_lock:
            status = "ok" if result.returncode == 0 else ("warn" if result.returncode == 2 else "FAIL")
            print(
                f"[{status}] {job.label} on cuda:{gpu_id} finished in {elapsed:.1f}s "
                f"(exit={result.returncode})",
                flush=True,
            )
        return JobResult(job=job, gpu_id=gpu_id, returncode=result.returncode, elapsed_seconds=elapsed, log_path=log_path)
    finally:
        gpu_queue.put(gpu_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-GPU IM-pretraining sweep over a single recorded run: no agency, no "
            "pretrained-inference loading, checkpoint saving on by default -- trains the "
            "classifier/confidence_decoder/KR stack from scratch, one job per (--variants x "
            "--seeds) pair, scheduled across local GPUs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir", help="Path to a recorded run_<timestamp>/ directory (passed straight through to offline_replay.py).")
    parser.add_argument(
        "--variants", nargs="+", required=True,
        help="inference_model_variant names to sweep, i.e. im_models/<name>.py modules in this repo "
             "(e.g. --variants default mahalanobis optim). One job per (variant, seed) pair. "
             f"Currently available: {available_variants()}.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help=f"Seeds to pretrain each variant with (default: {DEFAULT_SEEDS}). One offline_replay.py job per (variant, seed) pair.")
    parser.add_argument(
        "--pretrained-models-dir", default=DEFAULT_PRETRAINED_MODELS_DIR,
        help=f"Where classifier/confidence_decoder checkpoints are written (and, if --load-pretrained "
             f"were ever passed elsewhere, later loaded from) -- forwarded to offline_replay.py's "
             f"--pretrained-models-dir (default: {DEFAULT_PRETRAINED_MODELS_DIR!r}). Created if missing.",
    )
    parser.add_argument("--wandb-group-name", default=DEFAULT_WANDB_GROUP_NAME, help=f"wandb.wb_group_name shared by every run (default: {DEFAULT_WANDB_GROUP_NAME!r}).")
    parser.add_argument("--no-wandb", action="store_true", help="Disable real wandb tracking for this sweep (default: enabled, --wandb-run-name set per (variant, seed) pair via wb_run_name()).")
    parser.add_argument("--no-save", action="store_true", help="Disable model checkpoint saving (default: enabled -- see module docstring, this is the whole point of the script).")
    parser.add_argument("--online-eval-step-freq", type=int, default=None, help="Override intrusion_detection.online_eval_step_freq (how often the eval/save check runs). Default: leave the recorded manifest's value.")
    parser.add_argument("--online-eval-rounds", type=int, default=None, help="Override intrusion_detection.online_evaluation_rounds (how many batches each eval/save check averages over). Default: leave the recorded manifest's value.")
    parser.add_argument("--repetitions", type=int, default=None, help="Forwarded to offline_replay.py --repetitions (default: offline_replay.py's own default of 20).")
    parser.add_argument("--max-shards", type=int, default=None, help="Forwarded to offline_replay.py --max-shards (e.g. for a smoke-test sweep).")
    parser.add_argument("--max-samples", type=int, default=None, help="Forwarded to offline_replay.py --max-samples.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Forwarded to offline_replay.py --log-level.")
    parser.add_argument("--offline-replay-path", type=Path, default=Path(__file__).resolve().parent / "offline_replay.py", help="Path to offline_replay.py (default: alongside this script).")

    parser.add_argument(
        "--gpus", nargs="+", type=int, default=None,
        help="GPU indices to use, e.g. --gpus 0 1 2 3 4 5 6 7. Default: autodetect all "
             "visible GPUs via nvidia-smi; aborts if none can be detected and this isn't given.",
    )
    parser.add_argument(
        "--jobs-per-gpu", type=int, default=1,
        help="Concurrent offline_replay.py processes per GPU (default: 1). Raise this only if you've "
             "confirmed a single recorded-run replay's GPU memory footprint leaves headroom for more.",
    )
    parser.add_argument(
        "--logs-dir", type=Path, default=None,
        help="Directory for per-job log files (default: ./offline_sweep_logs/<wandb-group-name>/). "
             "Created if missing. Each job's full offline_replay.py stdout/stderr goes to "
             "<logs-dir>/im_pretrain-<variant>-seed<seed>.log -- check these for the actual training logs; "
             "console output here is just a one-line-per-job progress/status feed.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned GPU slot count and full job list (with the command each job would run) "
             "and exit, without launching anything.",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"run_dir not found: {run_dir}")
        return 1

    offline_replay_path = args.offline_replay_path.resolve()
    if not offline_replay_path.exists():
        print(f"offline_replay.py not found: {offline_replay_path}")
        return 1

    known_variants = available_variants()
    unknown_variants = [v for v in args.variants if v not in known_variants]
    if unknown_variants:
        print(
            f"[offline_pretrain_im_parallel] Unknown --variants {unknown_variants} -- no matching "
            f"im_models/<name>.py module. Available variants: {known_variants}."
        )
        return 1

    pretrained_models_dir = Path(args.pretrained_models_dir).resolve()
    pretrained_models_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = args.gpus if args.gpus is not None else par.detect_gpu_ids()
    if not gpu_ids:
        print(
            "[offline_pretrain_im_parallel] No GPUs given and nvidia-smi-based autodetection "
            "found none. Pass --gpus explicitly (e.g. --gpus 0 1 2 3 4 5 6 7)."
        )
        return 1
    print(f"[offline_pretrain_im_parallel] Using GPUs: {gpu_ids} (jobs_per_gpu={args.jobs_per_gpu})")

    logs_dir = args.logs_dir or (Path("offline_sweep_logs") / args.wandb_group_name)
    logs_dir.mkdir(parents=True, exist_ok=True)

    passthrough_args: list[str] = ["--log-level", args.log_level]
    if args.no_save:
        passthrough_args.append("--no-save")
    if args.repetitions is not None:
        passthrough_args += ["--repetitions", str(args.repetitions)]
    if args.max_shards is not None:
        passthrough_args += ["--max-shards", str(args.max_shards)]
    if args.max_samples is not None:
        passthrough_args += ["--max-samples", str(args.max_samples)]

    jobs = build_jobs(args.variants, args.seeds)
    slot_ids = [gpu_id for gpu_id in gpu_ids for _ in range(args.jobs_per_gpu)]
    print(
        f"[offline_pretrain_im_parallel] {len(jobs)} job(s) total "
        f"(variants={args.variants} x seeds={args.seeds}), "
        f"{len(slot_ids)} concurrent slot(s) ({len(gpu_ids)} GPU(s) x {args.jobs_per_gpu} job(s)/GPU), "
        f"pretrained_models_dir={pretrained_models_dir}"
    )

    if args.dry_run:
        print(f"[dry-run] logs_dir={logs_dir}")
        for job in jobs:
            overrides = job_overrides(job, args.wandb_group_name, args.online_eval_step_freq, args.online_eval_rounds)
            print(f"[dry-run] {job.label}: --set " + " --set ".join(overrides))
        return 0

    gpu_queue: "queue.Queue[int]" = queue.Queue()
    for slot_gpu_id in slot_ids:
        gpu_queue.put(slot_gpu_id)
    print_lock = threading.Lock()

    overall_start = time.monotonic()
    results: list[JobResult] = []
    interrupted = False
    executor = ThreadPoolExecutor(max_workers=len(slot_ids))
    try:
        futures = [
            executor.submit(
                run_job, job, gpu_queue, offline_replay_path, run_dir, pretrained_models_dir,
                args.wandb_group_name, args.online_eval_step_freq, args.online_eval_rounds,
                not args.no_wandb, passthrough_args, logs_dir, print_lock,
            )
            for job in jobs
        ]
        try:
            for future in as_completed(futures):
                results.append(future.result())
        except KeyboardInterrupt:
            interrupted = True
            print(
                "\n[offline_pretrain_im_parallel] Interrupted -- cancelling queued jobs "
                "(jobs already running are being killed by the same Ctrl-C and will still "
                "report their result below).",
                flush=True,
            )
            for future in futures:
                future.cancel()
            for future in futures:
                if future.cancelled():
                    continue
                try:
                    results.append(future.result())
                except Exception:
                    pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    overall_elapsed = time.monotonic() - overall_start

    ok = [r for r in results if r.returncode == 0]
    warned = [r for r in results if r.returncode == 2]
    failed = [r for r in results if r.returncode not in (0, 2)]

    status_label = "Interrupted" if interrupted else "Sweep complete"
    print(f"\n[offline_pretrain_im_parallel] ===== {status_label} in {overall_elapsed:.1f}s "
          f"({len(results)}/{len(jobs)} jobs accounted for) =====")
    print(f"  ok:     {len(ok)}")
    print(f"  warn:   {len(warned)} (shard load/parse errors -- check logs, run is likely still usable)")
    print(f"  failed: {len(failed)}")
    if warned:
        print("  Warned jobs:")
        for r in warned:
            print(f"    {r.job.label}: log={r.log_path}")
    if failed:
        print("  Failed jobs:")
        for r in failed:
            print(f"    {r.job.label}: exit={r.returncode} log={r.log_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
