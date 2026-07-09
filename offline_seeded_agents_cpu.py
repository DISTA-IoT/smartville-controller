#!/usr/bin/env python3
"""CPU parallel counterpart to offline_seeded_agents_parallel.py.

offline_seeded_agents_parallel.py fans the (seed, agent, ablation-mode) sweep
out across N local GPUs: it insists on nvidia-smi-detectable GPUs and pins
every job to `--device cuda:<id>`. On a GPU-less box that script aborts by
design. This script runs the *exact same* sweep, purely on CPU: every job is
one `python3 offline_replay.py <run_dir> --device cpu ...` subprocess, and a
fixed-size pool of worker "slots" pulls jobs off a shared queue. There are no
GPUs involved and nvidia-smi is never consulted.

The DQN-family nets these runs exercise are small, so CPU replay is slower
than GPU but perfectly usable -- the win here is running many configs at once
instead of one after another (which is all offline_seeded_agents.py does).

    python3 offline_seeded_agents_cpu.py /data/tiger_data_collection/run_20250101_120000 \
        --num-workers 10 --threads-per-worker 5

Concurrency model -- worker slots, not GPUs:

- --num-workers N is how many offline_replay.py subprocesses run at once. It
  defaults to filling the machine: os.cpu_count() // --threads-per-worker when
  that flag is set (so the default of 50 cores / 5 threads = 10 workers), or
  os.cpu_count() when it isn't.

- --threads-per-worker T controls how much of the CPU each individual worker
  is allowed to grab, and this is the important knob. PyTorch/OpenMP/BLAS
  default every process to use *all* cores for intra-op parallelism, so N
  workers left unchecked spawn N*num_cores compute threads fighting over
  num_cores physical cores -- heavy oversubscription that usually runs slower
  than fewer workers. Setting T caps each worker at T threads (via
  OMP/MKL/OpenBLAS/NUMEXPR/VECLIB *_NUM_THREADS in that subprocess's env) and,
  when the machine has enough cores and `taskset` is available, pins each
  worker to its own contiguous bucket of T cores. That is what turns "50 CPUs"
  into "10 buckets of 5 CPUs each, one worker per bucket".

- If --threads-per-worker is NOT set, this script sets no thread limits and
  pins nothing: it just launches --num-workers subprocesses and lets the OS
  scheduler place all their threads across all cores as it sees fit. Use this
  when you'd rather let the scheduler load-balance than partition cores
  yourself; be aware many small workers will then oversubscribe the box.

Everything else is identical to offline_seeded_agents_parallel.py, which it
mirrors deliberately: the same (seed outermost, agent, mode innermost) job
order (build_jobs / seq.modes_for_agent), the same per-job --set override and
wandb run-naming logic (imported from offline_seeded_agents, not re-derived),
the same per-job log file under --logs-dir (concurrent subprocesses sharing one
stdout would interleave into noise), the same keep-going-on-failure policy (a
job whose offline_replay.py exits outside {0, 2} is recorded as failed but does
not abort the rest of the sweep; the script exits 1 if any job failed, 0
otherwise), the same single-SIGINT-cancels-queued-jobs Ctrl-C handling, the
same --dry-run, and the same end-of-run summary table. See
offline_seeded_agents.py's and offline_seeded_agents_parallel.py's module
docstrings for the full rationale behind the run-naming / --no-save /
--repetitions / --set conventions.
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import offline_seeded_agents as seq

# Env vars that cap a subprocess's intra-op / BLAS thread pools. Set together
# to --threads-per-worker so PyTorch, OpenMP and the various BLAS backends all
# honour the same per-worker budget instead of each defaulting to all cores.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


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
    slot_id: int
    returncode: int
    elapsed_seconds: float
    log_path: Path


def build_jobs(seeds: list[int], agents: list[str], modes: list[str], ablated_agents: list[str]) -> list[Job]:
    # Seeds outermost, agents/modes innermost -- same order as
    # offline_seeded_agents.py / offline_seeded_agents_parallel.py, so an
    # interrupted sweep's "first N jobs" still means full coverage of the early
    # seeds. Actual completion order under parallel scheduling differs from
    # submission order regardless. Agents not in ablated_agents only get
    # "baseline" -- see offline_seeded_agents.modes_for_agent().
    jobs = []
    for seed in seeds:
        for agent in agents:
            for mode in seq.modes_for_agent(agent, modes, ablated_agents):
                jobs.append(Job(seed=seed, agent=agent, mode=mode))
    return jobs


def build_core_buckets(num_workers: int, threads_per_worker: int) -> list[list[int]] | None:
    """
    Partition the machine's cores into one contiguous bucket of
    threads_per_worker cores per worker slot, for taskset affinity pinning.

    Returns None (caller falls back to thread-caps-only, no pinning) if the
    box can't be cleanly partitioned -- taskset missing, cpu_count unknown, or
    num_workers * threads_per_worker exceeding the available cores (in which
    case buckets would overlap and pinning would defeat its own purpose).
    """
    if shutil.which("taskset") is None:
        return None
    cpu_count = os.cpu_count()
    if not cpu_count or num_workers * threads_per_worker > cpu_count:
        return None
    return [list(range(i * threads_per_worker, (i + 1) * threads_per_worker)) for i in range(num_workers)]


def run_job(
    job: Job,
    slot_queue: "queue.Queue[int]",
    offline_replay_path: Path,
    run_dir: Path,
    cti_period: int,
    cti_confidence_threshold: float,
    hard_g2s: list[str],
    group_name: str,
    wandb_enabled: bool,
    passthrough_args: list[str],
    logs_dir: Path,
    print_lock: threading.Lock,
    threads_per_worker: int | None,
    core_buckets: list[list[int]] | None,
    postfix: str | None = None,
) -> JobResult:
    slot_id = slot_queue.get()
    try:
        overrides = [
            f"intrusion_detection.agent={job.agent}",
            f"intrusion_detection.seed={job.seed}",
            *seq.ablation_set_overrides(job.mode, cti_period, cti_confidence_threshold, hard_g2s),
            f"wandb.wb_group_name={group_name}",
        ]
        cmd = [
            sys.executable, str(offline_replay_path), str(run_dir),
            "--agency", "--device", "cpu",
        ]
        if wandb_enabled:
            cmd += ["--wandb", "--wandb-run-name", seq.wb_run_name(job.agent, job.mode, postfix)]
        for override in overrides:
            cmd += ["--set", override]
        cmd += passthrough_args

        # Per-worker CPU budget. Only touched when --threads-per-worker is set;
        # otherwise env is left as-is and nothing is pinned, so the OS
        # scheduler places every worker's threads across all cores itself.
        env = os.environ.copy()
        pin_desc = "unpinned"
        if threads_per_worker is not None:
            for var in THREAD_ENV_VARS:
                env[var] = str(threads_per_worker)
            if core_buckets is not None:
                cores = core_buckets[slot_id]
                cmd = ["taskset", "-c", f"{cores[0]}-{cores[-1]}"] + cmd
                pin_desc = f"cores {cores[0]}-{cores[-1]}"
            else:
                pin_desc = f"{threads_per_worker} threads"

        log_path = logs_dir / f"{job.label}.log"
        with print_lock:
            print(f"[start] {job.label} on slot {slot_id} ({pin_desc}) (log: {log_path})", flush=True)

        start = time.monotonic()
        with open(log_path, "w") as log_file:
            log_file.write(f"# command: {' '.join(cmd)}\n\n")
            log_file.flush()
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        elapsed = time.monotonic() - start

        with print_lock:
            status = "ok" if result.returncode == 0 else ("warn" if result.returncode == 2 else "FAIL")
            print(
                f"[{status}] {job.label} on slot {slot_id} finished in {elapsed:.1f}s "
                f"(exit={result.returncode})",
                flush=True,
            )
        return JobResult(job=job, slot_id=slot_id, returncode=result.returncode, elapsed_seconds=elapsed, log_path=log_path)
    finally:
        slot_queue.put(slot_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CPU parallel version of offline_seeded_agents.py: same "
            "(seed, agent, ablation-mode) sweep over one recorded run, scheduled "
            "across --num-workers CPU worker slots (each offline_replay.py --device cpu) "
            "instead of across GPUs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir", help="Path to a recorded run_<timestamp>/ directory (passed straight through to offline_replay.py).")
    parser.add_argument("--agents", nargs="+", default=seq.DEFAULT_AGENTS, help=f"Agent types to sweep (default: {seq.DEFAULT_AGENTS}).")
    parser.add_argument("--seeds", nargs="+", type=int, default=seq.DEFAULT_SEEDS, help=f"Seeds to run per agent (default: {seq.DEFAULT_SEEDS}).")
    parser.add_argument("--ablation-modes", nargs="+", default=seq.ABLATION_MODES, choices=seq.ABLATION_MODES, help=f"Ablation modes to sweep (default: {seq.ABLATION_MODES}). Only actually applied to agents listed in --ablated-agents.")
    parser.add_argument("--ablated-agents", nargs="+", default=seq.DEFAULT_ABLATED_AGENTS, help=f"Which agents (from --agents) to sweep across --ablation-modes (default: {seq.DEFAULT_ABLATED_AGENTS}); every other agent in --agents only runs 'baseline'.")
    parser.add_argument("--cti-period", type=int, default=seq.DEFAULT_CTI_PERIOD, help=f"intrusion_detection.cti_period for the 'periodic_cti' mode (default: {seq.DEFAULT_CTI_PERIOD}).")
    parser.add_argument("--cti-confidence-threshold", type=float, default=seq.DEFAULT_CTI_CONFIDENCE_THRESHOLD, help=f"intrusion_detection.cti_confidence_threshold for the 'fixed_threshold' mode (default: {seq.DEFAULT_CTI_CONFIDENCE_THRESHOLD}).")
    parser.add_argument("--hard-g2s", nargs="*", default=seq.DEFAULT_HARD_G2S, help=f"intrusion_detection.hard_g2s blocklist for the 'hard_g2s' mode: greedy CTI buys targeting these classes are remapped to a block, so only the off-list G2s are actually purchased (default: {seq.DEFAULT_HARD_G2S} -- the malicious G2s, leaving benign doorlock/echo purchasable). Pass --hard-g2s with no values to make 'hard_g2s' identical to plain 'greedy_cti'.")
    parser.add_argument("--wandb-group-name", default=seq.DEFAULT_WANDB_GROUP_NAME, help=f"wandb.wb_group_name shared by every run (default: {seq.DEFAULT_WANDB_GROUP_NAME!r}).")
    parser.add_argument("--no-wandb", action="store_true", help="Disable real wandb tracking for this sweep (default: enabled, --wandb-run-name set to the agent string).")
    parser.add_argument(
        "--postfix", default=None,
        help="Optional suffix appended to every wandb run name in this sweep, as "
             "'<wb_run_name(agent, mode)>-<postfix>'. Use this when re-running the same "
             "--wandb-group-name with different hyperparameters (e.g. --postfix lr1e-4): the "
             "group stays the same for wandb's 'group by name' aggregation, but each variant "
             "gets its own run name -- and its own checkpoint path -- instead of overwriting "
             "the previous sweep's runs of the same agent/mode.",
    )
    parser.add_argument("--save", action="store_true", help="Enable model checkpoint saving (default: disabled -- see offline_seeded_agents.py docstring on checkpoint-name collisions).")
    parser.add_argument("--repetitions", type=int, default=None, help="Forwarded to offline_replay.py --repetitions (default: offline_replay.py's own default of 20).")
    parser.add_argument("--max-shards", type=int, default=None, help="Forwarded to offline_replay.py --max-shards (e.g. for a smoke-test sweep).")
    parser.add_argument("--max-samples", type=int, default=None, help="Forwarded to offline_replay.py --max-samples.")
    parser.add_argument("--load-pretrained", action="store_true", help="Forwarded to offline_replay.py --load-pretrained.")
    parser.add_argument("--pretrained-models-dir", default=None, help="Forwarded to offline_replay.py --pretrained-models-dir.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Forwarded to offline_replay.py --log-level.")
    parser.add_argument("--offline-replay-path", type=Path, default=Path(__file__).resolve().parent / "offline_replay.py", help="Path to offline_replay.py (default: alongside this script).")

    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of offline_replay.py subprocesses to run concurrently. Default: fill the "
             "machine -- os.cpu_count() // --threads-per-worker if that flag is set (so 50 cores / "
             "5 threads = 10 workers), else os.cpu_count().",
    )
    parser.add_argument(
        "--threads-per-worker", type=int, default=None,
        help="Cap each worker at this many CPU threads (OMP/MKL/OpenBLAS/NUMEXPR/VECLIB *_NUM_THREADS) "
             "and, when the box has enough cores and `taskset` is present, pin each worker to its own "
             "contiguous bucket of this many cores -- e.g. --num-workers 10 --threads-per-worker 5 on a "
             "50-core box = 10 buckets of 5 cores. If omitted, no thread limits are set and nothing is "
             "pinned: --num-workers subprocesses launch and the OS scheduler places all their threads.",
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
        help="Print the planned worker/thread layout and full job list (with the command each job "
             "would run) and exit, without launching anything.",
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

    if args.threads_per_worker is not None and args.threads_per_worker < 1:
        print(f"--threads-per-worker must be >= 1 (got {args.threads_per_worker}).")
        return 1

    cpu_count = os.cpu_count() or 1
    if args.num_workers is not None:
        num_workers = args.num_workers
    elif args.threads_per_worker is not None:
        num_workers = max(1, cpu_count // args.threads_per_worker)
    else:
        num_workers = cpu_count
    if num_workers < 1:
        print(f"--num-workers must be >= 1 (got {num_workers}).")
        return 1

    core_buckets = None
    if args.threads_per_worker is not None:
        core_buckets = build_core_buckets(num_workers, args.threads_per_worker)
        if core_buckets is not None:
            pin_note = f"pinned to contiguous {args.threads_per_worker}-core buckets via taskset"
        else:
            pin_note = (
                f"capped at {args.threads_per_worker} threads each, not pinned "
                "(taskset unavailable or num_workers*threads_per_worker exceeds core count)"
            )
        print(f"[offline_seeded_agents_cpu] {num_workers} worker(s), {pin_note}. Detected {cpu_count} CPU(s).")
    else:
        print(
            f"[offline_seeded_agents_cpu] {num_workers} worker(s), no thread cap "
            f"(scheduler places all threads). Detected {cpu_count} CPU(s). "
            "Note: without --threads-per-worker each worker's PyTorch/BLAS may use all cores, "
            "so many workers will oversubscribe the machine."
        )

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

    jobs = build_jobs(args.seeds, args.agents, args.ablation_modes, args.ablated_agents)
    print(f"[offline_seeded_agents_cpu] {len(jobs)} job(s) total across {num_workers} concurrent worker slot(s).")

    if args.dry_run:
        print(f"[dry-run] logs_dir={logs_dir}")
        for job in jobs:
            overrides = [
                f"intrusion_detection.agent={job.agent}",
                f"intrusion_detection.seed={job.seed}",
                *seq.ablation_set_overrides(job.mode, args.cti_period, args.cti_confidence_threshold, args.hard_g2s),
                f"wandb.wb_group_name={args.wandb_group_name}",
            ]
            run_name = seq.wb_run_name(job.agent, job.mode, args.postfix)
            print(f"[dry-run] {job.label}: --device cpu --wandb-run-name {run_name} --set " + " --set ".join(overrides))
        return 0

    slot_queue: "queue.Queue[int]" = queue.Queue()
    for slot_id in range(num_workers):
        slot_queue.put(slot_id)
    print_lock = threading.Lock()

    overall_start = time.monotonic()
    results: list[JobResult] = []
    interrupted = False
    executor = ThreadPoolExecutor(max_workers=num_workers)
    try:
        futures = [
            executor.submit(
                run_job, job, slot_queue, offline_replay_path, run_dir, args.cti_period,
                args.cti_confidence_threshold, args.hard_g2s,
                args.wandb_group_name, not args.no_wandb, passthrough_args, logs_dir, print_lock,
                args.threads_per_worker, core_buckets,
                postfix=args.postfix,
            )
            for job in jobs
        ]
        try:
            for future in as_completed(futures):
                results.append(future.result())
        except KeyboardInterrupt:
            # Same policy as offline_seeded_agents_parallel.py: a single SIGINT
            # cancels every not-yet-started queued job so freed worker slots stop
            # getting backfilled, instead of waiting out the whole remaining grid.
            # Jobs whose subprocess is already running get killed by the same
            # SIGINT (it hit this whole process group) and still record their
            # non-zero result below; only jobs that hadn't started a subprocess
            # yet are actually dropped.
            interrupted = True
            print(
                "\n[offline_seeded_agents_cpu] Interrupted -- cancelling queued jobs "
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
    print(f"\n[offline_seeded_agents_cpu] ===== {status_label} in {overall_elapsed:.1f}s "
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
