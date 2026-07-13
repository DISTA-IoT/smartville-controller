#!/usr/bin/env python3
"""SIMBA parallel experiment launcher (wandb project: SIMBA).

Runs one simba_offline.py subprocess per (ablation-mode, agent, seed)
triple, round-robining over the given GPUs — same spirit as
offline_seeded_agents_parallel.py.

Example
-------
python simba_experiments.py pre_recorded_data/ \
    --gpus 0 2 3 4 5 6 \
    --ablation-modes drl no_epistemic greedy_cti \
    --agents dqn ddqn dueling_ddqn \
    --seeds 6 1 \
    --episodes 200 --wandb-project SIMBA

`--agents` is orthogonal to `--ablation-modes` (the modes gate the
epistemic action; the agents swap the value-learner). It defaults to just
`dqn`, so omitting it reproduces the original modes x seeds grid.

Anything after `--` is forwarded verbatim to every simba_offline.py run:
python simba_experiments.py data/ --seeds 1 2 -- --set gamma=0.997
"""

import argparse
import itertools
import os
import subprocess
import sys
import time


def parse_args():
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        split = argv.index('--')
        argv, extra = argv[:split], argv[split + 1:]

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='trace dir, forwarded to simba_offline.py')
    ap.add_argument('--gpus', nargs='*', default=[],
                    help='GPU ids to round-robin over; empty = CPU only')
    ap.add_argument('--jobs-per-gpu', type=int, default=1)
    ap.add_argument('--ablation-modes', nargs='+',
                    default=['drl', 'no_epistemic', 'greedy_cti'])
    ap.add_argument('--agents', nargs='+',
                    default=['dqn'], choices=['dqn', 'ddqn', 'dueling_ddqn'],
                    help='DM value-learners to grid over (orthogonal to the '
                         'ablation modes). Default just dqn.')
    ap.add_argument('--seeds', nargs='+', type=int, default=[777])
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--eval-episodes', type=int, default=5)
    ap.add_argument('--wandb-project', default='SIMBA')
    ap.add_argument('--no-wandb', action='store_true')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--hard-g2s', nargs='*', default=None,
                    help='forwarded to every run (CTI-sale blocklist)')
    ap.add_argument('--log-dir', default='runs_simba/logs')
    return ap.parse_args(argv), extra


def main():
    args, extra = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    runner = os.path.join(here, 'simba_offline.py')
    os.makedirs(args.log_dir, exist_ok=True)

    combos = list(itertools.product(args.ablation_modes, args.agents, args.seeds))
    slots = (args.gpus or [None]) * args.jobs_per_gpu
    running = {}   # popen -> (mode, agent, seed, slot, logfile)
    pending = list(combos)
    failures = []

    print(f'[simba-exp] {len(combos)} runs '
          f'({len(args.ablation_modes)} modes x {len(args.agents)} agents '
          f'x {len(args.seeds)} seeds), {len(slots)} parallel slots')

    def launch(mode, agent, seed, slot):
        # Mirror simba_offline.run_name_for: the 'dqn' agent keeps the
        # historical <mode>_seed<seed> name, the others insert the agent.
        name = f'{mode}_seed{seed}' if agent == 'dqn' \
            else f'{mode}_{agent}_seed{seed}'
        cmd = [sys.executable, runner, args.data,
               '--mode', mode, '--agent', agent, '--seed', str(seed),
               '--episodes', str(args.episodes),
               '--eval-episodes', str(args.eval_episodes),
               '--wandb-project', args.wandb_project,
               '--run-name', name]
        env = dict(os.environ)
        if slot is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(slot)
            cmd += ['--device', 'cuda:0']
        if args.no_wandb:
            cmd.append('--no-wandb')
        if args.synthetic:
            cmd.append('--synthetic')
        if args.hard_g2s is not None:
            cmd += ['--hard-g2s'] + args.hard_g2s
        cmd += extra
        logfile = open(os.path.join(args.log_dir, name + '.log'), 'w')
        print(f'[simba-exp] start {name} (gpu={slot if slot is not None else "cpu"})')
        proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                                env=env, cwd=here)
        running[proc] = (mode, agent, seed, slot, logfile)

    free = list(slots)
    while pending or running:
        while pending and free:
            mode, agent, seed = pending.pop(0)
            launch(mode, agent, seed, free.pop(0))
        time.sleep(2)
        for proc in [p for p in running if p.poll() is not None]:
            mode, agent, seed, slot, logfile = running.pop(proc)
            logfile.close()
            free.append(slot)
            status = 'ok' if proc.returncode == 0 else f'FAILED ({proc.returncode})'
            print(f'[simba-exp] done {mode}/{agent}_seed{seed}: {status}')
            if proc.returncode != 0:
                failures.append((mode, agent, seed))

    # ---------------------------------------------------------- summary
    print('\n[simba-exp] ===== eval summary =====')
    import json
    per_cell = {}
    for mode, agent, seed in combos:
        name = f'{mode}_seed{seed}' if agent == 'dqn' \
            else f'{mode}_{agent}_seed{seed}'
        path = os.path.join(here, 'runs_simba', name, 'results.json')
        if os.path.isfile(path):
            with open(path) as f:
                r = json.load(f)
            per_cell.setdefault((mode, agent), []).append(r['eval_mean_return'])
            print(f"  {mode:14s} {agent:12s} seed {seed:4d}  "
                  f"return {r['eval_mean_return']:9.1f}  "
                  f"buys {r['eval_mean_buys']:.1f}  "
                  f"{r['eval_buys_per_class']}")
    for (mode, agent), vals in per_cell.items():
        mean = sum(vals) / len(vals)
        print(f'  {mode:14s} {agent:12s} MEAN over {len(vals)} seeds: {mean:9.1f}')
    if failures:
        print(f'[simba-exp] failures: {failures}')
        sys.exit(1)


if __name__ == '__main__':
    main()
