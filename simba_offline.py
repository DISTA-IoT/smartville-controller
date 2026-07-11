#!/usr/bin/env python3
"""SIMBA offline runner — train + evaluate one (ablation, seed) pair on a
recorder-format trace, no GNS3 required.

Examples
--------
# synthetic smoke run (no data needed; trace is generated on the fly):
python simba_offline.py ./simba_synth --synthetic --mode drl --seed 6 --no-wandb

# pre-recorded traces, wandb project SIMBA:
python simba_offline.py pre_recorded_data/ --mode greedy_cti --seed 1 \
    --episodes 200 --wandb-project SIMBA --device cuda:0

Any SimbaConfig field can be overridden with --set, e.g.
    --set unknown_accept_discount=0.25 --set gamma=0.998 --set init_budget=500
"""

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simba.brain import SimbaBrain, ABLATIONS           # noqa: E402
from simba.config import SimbaConfig                    # noqa: E402
from simba.data import load_trace                       # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='trace dir (a run_*/ dir, or a dir of run_*/ '
                                 'dirs — newest wins); with --synthetic, the '
                                 'dir where the synthetic trace is generated')
    ap.add_argument('--mode', default='drl', choices=list(ABLATIONS))
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--episodes', type=int, default=120)
    ap.add_argument('--eval-episodes', type=int, default=5)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--synthetic', action='store_true',
                    help='generate a synthetic trace under DATA if none exists')
    ap.add_argument('--synth-ticks', type=int, default=200)
    ap.add_argument('--run-dir', default=None, help='explicit run_*/ dir')
    ap.add_argument('--max-shards', type=int, default=None)
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--out', default=None,
                    help='output dir for checkpoints/results '
                         '(default runs_simba/<run-name>)')
    ap.add_argument('--wandb-project', default='SIMBA')
    ap.add_argument('--wandb-group', default=None,
                    help='wandb group (default: the ablation mode)')
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--no-wandb', action='store_true')
    ap.add_argument('--prices-json', default=None,
                    help='JSON string or path overriding per-class CTI prices')
    ap.add_argument('--hard-g2s', nargs='*', default=None,
                    help='G2 classes whose CTI is never on sale (layered '
                         'on any mode; greedy_cti + this = oracle buyer)')
    ap.add_argument('--set', dest='overrides', action='append', default=[],
                    metavar='KEY=VALUE', help='override any SimbaConfig field')
    return ap.parse_args(argv)


def apply_overrides(cfg: SimbaConfig, overrides):
    types = {f.name: f.type for f in dataclasses.fields(SimbaConfig)}
    for item in overrides:
        key, _, raw = item.partition('=')
        if key not in types:
            raise SystemExit(f'--set: unknown config field {key!r}')
        current = getattr(cfg, key)
        if isinstance(current, bool):
            val = raw.lower() in ('1', 'true', 'yes')
        elif isinstance(current, int):
            val = int(raw)
        elif isinstance(current, float):
            val = float(raw)
        elif isinstance(current, (dict, list)):
            val = json.loads(raw)
        else:
            val = raw
        setattr(cfg, key, val)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(brain, trace, collect=None):
    """One pass over the trace; returns (env summary, aggregated tick info)."""
    brain.begin_episode(total_ticks=len(trace))
    agg = defaultdict(float)
    n_ticks = 0
    losses_im, losses_dm, cs_accs = [], [], []
    for t in range(len(trace)):
        flow, pkt, labels = trace.tick(t)
        info = brain.step_tick(flow, pkt, labels)
        n_ticks += 1
        for k in ('n_decisions', 'n_buys', 'tick_reward',
                  'ad_tp', 'ad_fp', 'ad_fn', 'ad_tn'):
            agg[k] += info.get(k) or 0
        if info.get('im_loss') is not None:
            losses_im.append(info['im_loss'])
        if info.get('dm_loss') is not None:
            losses_dm.append(info['dm_loss'])
        if info.get('cs_acc') is not None:
            cs_accs.append(info['cs_acc'])
        if brain.env.episode_ended(len(trace)):
            break
    summary = brain.env.summary()
    tp, fp, fn = agg['ad_tp'], agg['ad_fp'], agg['ad_fn']
    metrics = {
        'im_loss': float(np.mean(losses_im)) if losses_im else 0.0,
        'dm_loss': float(np.mean(losses_dm)) if losses_dm else 0.0,
        'cs_acc': float(np.mean(cs_accs)) if cs_accs else 0.0,
        'ad_recall': tp / max(1.0, tp + fn),
        'ad_precision': tp / max(1.0, tp + fp),
        'decisions': agg['n_decisions'],
    }
    if collect is not None:
        collect.append(summary['return'])
    return summary, metrics


def main(argv=None):
    args = parse_args(argv)

    if args.synthetic:
        has_run = os.path.isdir(args.data) and any(
            os.path.isfile(os.path.join(args.data, d, 'shards_index.jsonl'))
            for d in os.listdir(args.data))
        if not has_run:
            from simba.synth import write_synthetic_trace
            os.makedirs(args.data, exist_ok=True)
            write_synthetic_trace(args.data, n_ticks=args.synth_ticks,
                                  seed=args.seed)

    trace = load_trace(args.data, run_dir=args.run_dir,
                       max_shards=args.max_shards, max_samples=args.max_samples)

    cfg = SimbaConfig()
    cfg.apply_manifest(trace.manifest)
    cfg.ablation = args.mode
    cfg.seed = args.seed
    cfg.device = args.device
    if args.prices_json:
        raw = args.prices_json
        if os.path.isfile(raw):
            with open(raw) as f:
                raw = f.read()
        cfg.prices.update(json.loads(raw))
    if args.hard_g2s is not None:
        cfg.hard_g2s = list(args.hard_g2s)
    apply_overrides(cfg, args.overrides)

    run_name = args.run_name or f'{args.mode}_seed{args.seed}'
    out_dir = args.out or os.path.join('runs_simba', run_name)
    os.makedirs(out_dir, exist_ok=True)

    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project,
                            group=args.wandb_group or args.mode,
                            name=run_name,
                            config=dataclasses.asdict(cfg))
        except Exception as e:
            print(f'[simba] wandb disabled ({e})', file=sys.stderr)

    seed_everything(args.seed)
    brain = SimbaBrain(cfg)
    print(f'[simba] trace: {len(trace)} ticks / {trace.num_samples()} samples | '
          f'mode={args.mode} seed={args.seed} device={args.device}')
    t0 = time.time()
    brain.pretrain(trace)
    print(f'[simba] IM pretrained in {time.time()-t0:.1f}s  tau={brain.im.tau:.3f}')

    # ------------------------------------------------------------- training
    for ep in range(args.episodes):
        summary, metrics = run_episode(brain, trace)
        row = {f'train/{k}': v for k, v in {**summary, **metrics}.items()}
        row['train/epsilon'] = brain.agent.epsilon
        for _, label, _ in brain.env.buys:
            row[f'buys/{label}'] = row.get(f'buys/{label}', 0) + 1
        if wb:
            wb.log(row, step=ep)
        if ep % 5 == 0 or ep == args.episodes - 1:
            buys = ','.join(l for _, l, _ in brain.env.buys) or '-'
            print(f'[simba] ep {ep:4d}  return {summary["return"]:9.1f}  '
                  f'budget {summary["final_budget"]:9.1f}  '
                  f'buys [{buys}] wasted {summary["wasted_buys"]:.0f}  '
                  f'bankrupt {bool(summary["bankrupt"])}  '
                  f'eps {brain.agent.epsilon:.3f}  cs_acc {metrics["cs_acc"]:.2f}  '
                  f'ad_rec {metrics["ad_recall"]:.2f}')

    # ------------------------------------------------------------ evaluation
    brain.training = False
    eval_returns, eval_buys, eval_bankrupt = [], [], 0
    eval_buy_labels = defaultdict(int)
    for ep in range(args.eval_episodes):
        summary, metrics = run_episode(brain, trace, collect=eval_returns)
        eval_buys.append(summary['n_buys'])
        eval_bankrupt += int(summary['bankrupt'])
        for _, label, _ in brain.env.buys:
            eval_buy_labels[label] += 1
        print(f'[simba] EVAL ep {ep}  return {summary["return"]:9.1f}  '
              f'buys {summary["n_buys"]:.0f}  '
              f'bought [{",".join(l for _, l, _ in brain.env.buys) or "-"}]')
    brain.training = True

    results = {
        'mode': args.mode,
        'seed': args.seed,
        'eval_mean_return': float(np.mean(eval_returns)) if eval_returns else 0.0,
        'eval_std_return': float(np.std(eval_returns)) if eval_returns else 0.0,
        'eval_mean_buys': float(np.mean(eval_buys)) if eval_buys else 0.0,
        'eval_bankruptcies': eval_bankrupt,
        'eval_buys_per_class': dict(eval_buy_labels),
    }
    print(f'[simba] RESULT {json.dumps(results)}')
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    brain.agent.save(os.path.join(out_dir, 'dm_final.pt'))
    brain.im.save(os.path.join(out_dir, 'im_pretrained.pt'))
    if wb:
        for k, v in results.items():
            if not isinstance(v, dict):
                wb.summary[k] = v
        for label, n in eval_buy_labels.items():
            wb.summary[f'eval_buys/{label}'] = n
        wb.finish()
    return results


if __name__ == '__main__':
    main()
