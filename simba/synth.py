"""Synthetic trace generator (recorder format).

Adapted from the `write_synthetic_smoketest` cell of
tiger/analytics/tiger_data_analytics.ipynb, extended with what the SIMBA
game needs: many ticks, per-class per-tick Poisson traffic rates, and a
class geometry whose separability is controllable. The output directory
is byte-compatible with FlowDataRecorder captures, so swapping in real
pre-recorded traces requires no code change.

Default per-tick rates encode the intended economics:

    doorlock (benign G2)  rate 5  price  40   -> WORTH buying
    echo     (benign G2)  rate 1  price 400   -> NOT worth buying
    malicious G2s         various price 150   -> NOT worth buying
                                                 (blocking is free+correct)

Run directly:  python -m simba.synth OUT_DIR [--ticks 200] [--seed 777]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

# (class, group, role, per-tick rate)
DEFAULT_CLASS_SPECS = [
    ('hue',          'known', 'benign',    3.0),
    ('hakai',        'known', 'malicious', 2.0),
    ('torii',        'known', 'malicious', 1.0),
    ('okiru',        'g1',    'malicious', 1.0),
    ('cc_heartbeat', 'g1',    'malicious', 1.0),
    ('generic_ddos', 'g1',    'malicious', 1.0),
    ('doorlock',     'g2',    'benign',    5.0),
    ('echo',         'g2',    'benign',    1.0),
    ('mirai',        'g2',    'malicious', 4.0),
    ('gafgyt',       'g2',    'malicious', 2.0),
    ('hajime',       'g2',    'malicious', 1.0),
    ('h_scan',       'g2',    'malicious', 1.0),
    ('muhstik',      'g2',    'malicious', 1.0),
]

DEFAULT_PRICES_FOR_MANIFEST = {
    'doorlock': 40, 'echo': 400, 'mirai': 150, 'gafgyt': 150,
    'hajime': 150, 'h_scan': 150, 'muhstik': 150,
}


def write_synthetic_trace(out_dir,
                          n_ticks=200,
                          seed=777,
                          class_specs=None,
                          separability=1.0,
                          flows_per_sample=10,
                          packet_feat_dim=64,
                          shard_size=4000):
    """Writes a run_*/ directory under `out_dir`; returns its path."""
    specs = class_specs or DEFAULT_CLASS_SPECS
    rng = np.random.default_rng(seed)

    knowns = [c for c, g, _, _ in specs if g == 'known']
    g1s = [c for c, g, _, _ in specs if g == 'g1']
    g2s = [c for c, g, _, _ in specs if g == 'g2']
    benign = [c for c, _, r, _ in specs if r == 'benign']
    attack = [c for c, _, r, _ in specs if r == 'malicious']

    manifest = {
        'synthetic': True,
        'seed': seed,
        'flow_feat_dim': 4,
        'packet_feat_dim': packet_feat_dim,
        'use_packet_feats': True,
        'use_node_feats': False,
        'intrusion_detection': {'flows_per_sample': flows_per_sample,
                                'packets_per_sample': 1},
        'knowledge': {'Knowns': knowns, 'G1s': g1s, 'G2s': g2s,
                      'bening_patterns': benign, 'attack_patterns': attack},
        'rewards': {c: (1 if c in benign else -1) for c, _, _, _ in specs},
        'prices': DEFAULT_PRICES_FOR_MANIFEST,
        'class_rates': {c: rate for c, _, _, rate in specs},
    }

    # ---- per-class geometry (same spirit as the notebook cell) -----------
    # flow-feature mean vector on a circle, packet-byte template with a
    # class-dependent offset; `separability` scales class separation
    # against the sampling noise.
    n_classes = len(specs)
    geoms = {}
    for i, (c, group, role, _) in enumerate(specs):
        ang = i / n_classes * 2 * math.pi
        mal = role == 'malicious'
        fmean = np.array([
            (5e4 if mal else 8e3) * (1 + 0.6 * separability * math.cos(ang)),
            0.4 + 0.3 * separability * math.sin(ang),
            2 + 1.5 * separability * math.cos(ang),
            (120 if mal else 25) * (1 + 0.5 * separability * math.sin(ang)),
        ])
        tmpl = np.zeros(packet_feat_dim)
        tmpl[0] = 69
        tmpl[9] = 6 if mal else 17
        off = 40 * separability * math.sin(ang) + 15 * i
        tmpl = np.clip(tmpl + off + rng.integers(0, 120, packet_feat_dim), 0, 255)
        tmpl[12:20] = 0  # anonymised IP bytes
        geoms[c] = (fmean, tmpl)

    def make_sample(c):
        fmean, tmpl = geoms[c]
        w = np.zeros((flows_per_sample, 4))
        s0 = rng.integers(0, flows_per_sample)
        for t in range(s0, flows_per_sample):
            w[t] = fmean * ((t - s0 + 1) / (flows_per_sample - s0)) \
                 * rng.normal(1, 0.15, 4)
        pk = np.clip(tmpl + rng.normal(0, 10, packet_feat_dim), 0, 255)
        pk[12:20] = 0
        return np.clip(w, 0, None).astype(np.float32), pk.astype(np.float32)

    # ---- draw traffic tick by tick ---------------------------------------
    flows, pkts, labels, ticks = [], [], [], []
    for t in range(n_ticks):
        for c, _, _, rate in specs:
            n = int(rng.poisson(rate))
            if t < 2:  # every class shows up early so buffers see everyone
                n = max(n, 1)
            for _ in range(n):
                f, p = make_sample(c)
                flows.append(f)
                pkts.append(p)
                labels.append(c)
                ticks.append(t)

    # ---- write shards -----------------------------------------------------
    run = os.path.join(out_dir, 'run_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    index_lines = []
    n_total = len(labels)
    shard_id = 0
    lo = 0
    while lo < n_total:
        hi = min(lo + shard_size, n_total)
        # never split a tick across shards
        while hi < n_total and ticks[hi] == ticks[hi - 1]:
            hi += 1
        name = f'shard_{shard_id:06d}.pt'
        torch.save({
            'flow_features': torch.from_numpy(np.stack(flows[lo:hi])),
            'packet_features': torch.from_numpy(np.stack(pkts[lo:hi])).unsqueeze(1),
            'node_features': None,
            'element_classes': labels[lo:hi],
            'flow_ids': ['synthetic'] * (hi - lo),
            'tick': ticks[lo:hi],
        }, os.path.join(run, name))
        index_lines.append(json.dumps({
            'shard': name, 'num_samples': hi - lo,
            'tick_min': int(ticks[lo]), 'tick_max': int(ticks[hi - 1])}))
        shard_id += 1
        lo = hi

    with open(os.path.join(run, 'shards_index.jsonl'), 'w') as f:
        f.write('\n'.join(index_lines) + '\n')

    print(f'wrote {n_total} samples / {n_ticks} ticks -> {run}')
    return run


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out_dir')
    ap.add_argument('--ticks', type=int, default=200)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--separability', type=float, default=1.0)
    args = ap.parse_args()
    write_synthetic_trace(args.out_dir, n_ticks=args.ticks, seed=args.seed,
                          separability=args.separability)
