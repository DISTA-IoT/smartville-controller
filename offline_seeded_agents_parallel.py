#!/usr/bin/env python3
"""Multi-GPU parallel counterpart to offline_seeded_agents.py.

offline_seeded_agents.py runs the (seed, agent, ablation-mode) sweep as one
offline_replay.py subprocess after another, on whatever single device
--device defaults to. On a multi-GPU box that leaves 7 of 8 A100s idle the
whole time. This script schedules the exact same sweep across N GPUs
concurrently: a fixed-size pool of GPU "slots" (by default one job per GPU,
configurable via --jobs-per-gpu) pulls jobs off a shared queue, each job
pinned to its slot's GPU via offline_replay.py's --device cuda:<id>.

Still no GNS3/POX/live controller/other repo -- same as offline_seeded_agents.py,
just fanned out across local GPUs. Run it directly inside a checkout of this
repo on the multi-GPU machine, pointed at a recorded run directory:

    python3 offline_seeded_agents_parallel.py /data/tiger_data_collection/run_20250101_120000

GPU autodetection: if --gpus is not given, this script shells out to
`nvidia-smi --query-gpu=index --format=csv,noheader` to enumerate visible
GPUs. If that fails (no nvidia-smi, no GPUs visible) it aborts and asks you
to pass --gpus explicitly -- it will not silently fall back to CPU for a
sweep whose whole point is to use the GPUs.

Job scheduling: jobs are generated in the same (seed outermost, agent, mode
innermost) order as offline_seeded_agents.py, then submitted to a
ThreadPoolExecutor sized to the number of GPU slots (num_gpus *
--jobs-per-gpu). Each worker thread pops a free GPU id off a queue, runs one
`python3 offline_replay.py <run_dir> --device cuda:<id> ...` subprocess to
completion with its stdout/stderr captured to its own log file under
--logs-dir (concurrent subprocesses writing to one shared stdout would
otherwise interleave into an unreadable mess), then returns that GPU id to
the queue for the next queued job. This is a standard producer/consumer GPU
pool: whichever job's GPU frees up first claims the next queued job,
regardless of how long any individual run takes.

Failure handling differs from offline_seeded_agents.py's fail_loudly(): in a
multi-hour parallel grid, one bad config aborting every other in-flight GPU
job is usually not what you want. So a job that fails (offline_replay.py
exit code not in {0, 2}) is logged and recorded as failed, but the sweep
keeps going; the script's own exit code is 1 if any job failed (or had shard
errors), 0 if every job replayed cleanly, and a summary table of every job's
outcome and log path is printed at the end either way.

Ctrl-C: a single SIGINT cancels every not-yet-started queued job and prints
the summary right away, instead of hanging until the whole remaining grid
drains. Jobs whose subprocess was already running when you pressed Ctrl-C
are still killed outright by that same SIGINT (it hits the whole process
group); they're recorded as failed/cancelled rather than silently dropped.

Run-name / checkpoint conventions (wandb run name collapsed to the ablation
label for non-baseline modes via wb_run_name(), --no-save default,
--repetitions, --set-based config overrides) are identical to
offline_seeded_agents.py -- see that script's module docstring for the full
rationale. This script reuses its constants and per-job --set override /
run-naming logic directly (`import offline_seeded_agents`) rather than
re-deriving them, so the two stay in sync.
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

import offline_seeded_agents as seq


@dataclass(frozen=True)
class Job:
    seed: int
    agent: str
    mode: str

    @property
    def label(self) -> str:
        return f"{self.agent}-{self.mode}-seed{self.seed}"


@dataclass
class JobResult:
    job: Job
    gpu_id: int
    returncode: int
    elapsed_seconds: float
    log_path: Path


def detect_gpu_ids() -> list[int]:
    """
    Enumerates visible GPU indices via nvidia-smi. Returns [] (never raises)
    if nvidia-smi is missing or returns nothing parseable -- callers must
    decide what to do with an empty list (this script aborts rather than
    guessing a count).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []
    ids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            ids.append(int(line))
    return ids


def build_jobs(seeds: list[int], agents: list[str], modes: list[str]) -> list[Job]:
    # Seeds outermost, agents/modes innermost -- same order as
    # offline_seeded_agents.py, kept for log/console readability and so an
    # interrupted sweep's "first N jobs done" has the same interpretation
    # (full coverage of early seeds before later ones) even though actual
    # completion order under parallel scheduling will differ from submission order.
    jobs = []
    for seed in seeds:
        for agent in agents:
            for mode in modes:
                jobs.append(Job(seed=seed, agent=agent, mode=mode))
    return jobs


def run_job(
    job: Job,
    gpu_queue: "queue.Queue[int]",
    offline_replay_path: Path,
    run_dir: Path,
    cti_period: int,
    group_name: str,
    wandb_enabled: bool,
    passthrough_args: list[str],
    logs_dir: Path,
    print_lock: threading.Lock,
) -> JobResult:
    gpu_id = gpu_queue.get()
    try:
        overrides = [
            f"intrusion_detection.agent={job.agent}",
            f"intrusion_detection.seed={job.seed}",
            *seq.ablation_set_overrides(job.mode, cti_period),
            f"wandb.wb_group_name={group_name}",
        ]
        cmd = [
            sys.executable, str(offline_replay_path), str(run_dir),
            "--agency", "--device", f"cuda:{gpu_id}",
        ]
        if wandb_enabled:
            cmd += ["--wandb", "--wandb-run-name", seq.wb_run_name(job.agent, job.mode)]
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
            "Multi-GPU parallel version of offline_seeded_agents.py: same "
            "(seed, agent, ablation-mode) sweep over one recorded run, scheduled "
            "across local GPUs instead of run sequentially."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir", help="Path to a recorded run_<timestamp>/ directory (passed straight through to offline_replay.py).")
    parser.add_argument("--agents", nargs="+", default=seq.DEFAULT_AGENTS, help=f"Agent types to sweep (default: {seq.DEFAULT_AGENTS}).")
    parser.add_argument("--seeds", nargs="+", type=int, default=seq.DEFAULT_SEEDS, help=f"Seeds to run per agent (default: {seq.DEFAULT_SEEDS}).")
    parser.add_argument("--ablation-modes", nargs="+", default=seq.ABLATION_MODES, choices=seq.ABLATION_MODES, help=f"Ablation modes to sweep (default: {seq.ABLATION_MODES}).")
    parser.add_argument("--cti-period", type=int, default=seq.DEFAULT_CTI_PERIOD, help=f"intrusion_detection.cti_period for the 'periodic_cti' mode (default: {seq.DEFAULT_CTI_PERIOD}).")
    parser.add_argument("--wandb-group-name", default=seq.DEFAULT_WANDB_GROUP_NAME, help=f"wandb.wb_group_name shared by every run (default: {seq.DEFAULT_WANDB_GROUP_NAME!r}).")
    parser.add_argument("--no-wandb", action="store_true", help="Disable real wandb tracking for this sweep (default: enabled, --wandb-run-name set to the agent string).")
    parser.add_argument("--save", action="store_true", help="Enable model checkpoint saving (default: disabled -- see offline_seeded_agents.py docstring on checkpoint-name collisions).")
    parser.add_argument("--repetitions", type=int, default=None, help="Forwarded to offline_replay.py --repetitions (default: offline_replay.py's own default of 20).")
    parser.add_argument("--max-shards", type=int, default=None, help="Forwarded to offline_replay.py --max-shards (e.g. for a smoke-test sweep).")
    parser.add_argument("--max-samples", type=int, default=None, help="Forwarded to offline_replay.py --max-samples.")
    parser.add_argument("--load-pretrained", action="store_true", help="Forwarded to offline_replay.py --load-pretrained.")
    parser.add_argument("--pretrained-models-dir", default=None, help="Forwarded to offline_replay.py --pretrained-models-dir.")
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
             "<logs-dir>/<agent>-<mode>-seed<seed>.log -- check these for the actual training logs; "
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

    gpu_ids = args.gpus if args.gpus is not None else detect_gpu_ids()
    if not gpu_ids:
        print(
            "[offline_seeded_agents_parallel] No GPUs given and nvidia-smi-based autodetection "
            "found none. Pass --gpus explicitly (e.g. --gpus 0 1 2 3 4 5 6 7)."
        )
        return 1
    print(f"[offline_seeded_agents_parallel] Using GPUs: {gpu_ids} (jobs_per_gpu={args.jobs_per_gpu})")

    logs_dir = args.logs_dir or (Path("offline_sweep_logs") / args.wandb_group_name)
    logs_dir.mkdir(parents=True, exist_ok=True)

    passthrough_args: list[str] = ["--log-level", args.log_level]
    if args.pretrained_models_dir is not None:
        passthrough_args += ["--pretrained-models-dir", args.pretrained_models_dir]
    if args.load_pretrained:
        passthrough_args.append("--load-pretrained")
    if not args.save:
        passthrough_args.append("--no-save")
    if args.repetitions is not None:
        passthrough_args += ["--repetitions", str(args.repetitions)]
    if args.max_shards is not None:
        passthrough_args += ["--max-shards", str(args.max_shards)]
    if args.max_samples is not None:
        passthrough_args += ["--max-samples", str(args.max_samples)]

    jobs = build_jobs(args.seeds, args.agents, args.ablation_modes)
    slot_ids = [gpu_id for gpu_id in gpu_ids for _ in range(args.jobs_per_gpu)]
    print(
        f"[offline_seeded_agents_parallel] {len(jobs)} job(s) total, "
        f"{len(slot_ids)} concurrent slot(s) ({len(gpu_ids)} GPU(s) x {args.jobs_per_gpu} job(s)/GPU)."
    )

    if args.dry_run:
        print(f"[dry-run] logs_dir={logs_dir}")
        for job in jobs:
            overrides = [
                f"intrusion_detection.agent={job.agent}",
                f"intrusion_detection.seed={job.seed}",
                *seq.ablation_set_overrides(job.mode, args.cti_period),
                f"wandb.wb_group_name={args.wandb_group_name}",
            ]
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
                run_job, job, gpu_queue, offline_replay_path, run_dir, args.cti_period,
                args.wandb_group_name, not args.no_wandb, passthrough_args, logs_dir, print_lock,
            )
            for job in jobs
        ]
        try:
            for future in as_completed(futures):
                results.append(future.result())
        except KeyboardInterrupt:
            # Cancel every not-yet-started queued job so freed GPU slots stop
            # getting backfilled, instead of waiting out the whole remaining
            # grid (Python's default executor.shutdown(wait=True) does not
            # cancel pending futures, so without this Ctrl-C would appear to
            # hang until every already-submitted job finished). Jobs whose
            # subprocess is already running still get killed by the SIGINT
            # that just hit this whole process group, and still finish
            # recording their (non-zero) result below before we print the
            # summary -- only jobs that hadn't started a subprocess yet are
            # actually dropped.
            interrupted = True
            print(
                "\n[offline_seeded_agents_parallel] Interrupted -- cancelling queued jobs "
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
    print(f"\n[offline_seeded_agents_parallel] ===== {status_label} in {overall_elapsed:.1f}s "
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
