"""Loading recorder-format traces for offline replay.

A "run" directory is what the controller's FlowDataRecorder writes (and
what simba.synth generates):

    run_*/
      manifest.json        curriculum, feature dims, recording options
      shards_index.jsonl   one JSON line per shard
      shard_*.pt           {'flow_features': [N,T,4], 'packet_features':
                            [N,P,64] or None, 'element_classes': [str],
                            'tick': [int], ...}

Samples are grouped by their recorded `tick`: one tick == one batch of
flowstats arriving at the controller == one SimbaBrain.step_tick call.
"""

import json
import os
from typing import List, Optional

import numpy as np
import torch


class Trace:
    """A fully-loaded trace, indexable by tick."""

    def __init__(self, flow, pkt, labels, ticks, manifest):
        self.flow = flow            # float32 tensor [N, T, F]
        self.pkt = pkt              # float32 tensor [N, P, D] or None
        self.labels = labels        # np.array of str, [N]
        self.manifest = manifest

        order = np.argsort(ticks, kind='stable')
        self.flow = self.flow[order]
        if self.pkt is not None:
            self.pkt = self.pkt[order]
        self.labels = self.labels[order]
        ticks = np.asarray(ticks)[order]

        # tick id -> slice of samples
        self.tick_ids, starts = np.unique(ticks, return_index=True)
        self._bounds = list(zip(starts.tolist(),
                                np.append(starts[1:], len(ticks)).tolist()))

        # per-class sample indices (for pretraining pools / CTI delivery)
        self.class_indices = {}
        for c in np.unique(self.labels):
            self.class_indices[str(c)] = np.nonzero(self.labels == c)[0]

    def __len__(self):
        return len(self.tick_ids)

    def num_samples(self):
        return len(self.labels)

    def tick(self, i):
        """Returns (flow [n,T,F], pkt [n,P,D] or None, labels list[str])."""
        lo, hi = self._bounds[i]
        pkt = self.pkt[lo:hi] if self.pkt is not None else None
        return self.flow[lo:hi], pkt, list(self.labels[lo:hi])

    def sample_class(self, label: str, n: int, rng: np.random.Generator):
        """n random samples of a class (for pretraining / label delivery)."""
        idx = self.class_indices.get(label)
        if idx is None or len(idx) == 0:
            return None, None
        chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)
        pkt = self.pkt[chosen] if self.pkt is not None else None
        return self.flow[chosen], pkt


def resolve_run_dir(data_root: str, run_dir: Optional[str] = None) -> str:
    """`data_root` may be a run dir itself or a directory of run_*/ dirs
    (newest one wins) — same convention as the analytics notebook."""
    if run_dir:
        if not os.path.isfile(os.path.join(run_dir, 'shards_index.jsonl')):
            raise FileNotFoundError(f'{run_dir} has no shards_index.jsonl')
        return run_dir
    if os.path.isfile(os.path.join(data_root, 'shards_index.jsonl')):
        return data_root
    cands = [os.path.join(data_root, d) for d in sorted(os.listdir(data_root))]
    cands = [c for c in cands
             if os.path.isfile(os.path.join(c, 'shards_index.jsonl'))]
    if not cands:
        raise FileNotFoundError(
            f'No run dirs (with shards_index.jsonl) under {data_root}. '
            f'Generate one with simba/synth.py or point at a capture dir.')
    return max(cands, key=os.path.getmtime)


def load_trace(data_root: str,
               run_dir: Optional[str] = None,
               skip_first_shard: Optional[bool] = None,
               max_shards: Optional[int] = None,
               max_samples: Optional[int] = None) -> Trace:
    run = resolve_run_dir(data_root, run_dir)
    with open(os.path.join(run, 'manifest.json')) as f:
        manifest = json.load(f)

    with open(os.path.join(run, 'shards_index.jsonl')) as f:
        index = [json.loads(l) for l in f if l.strip()]
    # the first shard-index entry of a REAL capture is an initial
    # packet-repetition warm-up artefact (mirrors offline_replay.py);
    # synthetic traces have no such artefact.
    if skip_first_shard is None:
        skip_first_shard = not manifest.get('synthetic', False)
    if skip_first_shard and len(index) > 1:
        index = index[1:]
    if max_shards is not None:
        index = index[:max_shards]

    flow_parts, pkt_parts = [], []
    labels: List[str] = []
    ticks: List[int] = []
    have_pkt = True
    for entry in index:
        sd = torch.load(os.path.join(run, entry['shard']),
                        map_location='cpu', weights_only=False)
        flow_parts.append(sd['flow_features'].to(torch.float32))
        p = sd.get('packet_features')
        if p is None:
            have_pkt = False
        else:
            pkt_parts.append(p.to(torch.float32))
        labels += [str(c) for c in sd['element_classes']]
        ticks += [int(t) for t in sd['tick']]

    flow = torch.cat(flow_parts, dim=0)
    pkt = torch.cat(pkt_parts, dim=0) if (have_pkt and pkt_parts) else None
    labels_arr = np.array(labels)
    ticks_arr = np.array(ticks)
    if max_samples is not None and len(labels_arr) > max_samples:
        flow = flow[:max_samples]
        pkt = pkt[:max_samples] if pkt is not None else None
        labels_arr = labels_arr[:max_samples]
        ticks_arr = ticks_arr[:max_samples]

    return Trace(flow, pkt, labels_arr, ticks_arr, manifest)
