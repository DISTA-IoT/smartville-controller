"""SIMBA Inference Module (IM).

Deliberately minimal ASAP-style NIDS:

  * encoder      — GRU over the flowstats window + MLP over raw packet
                   bytes, concatenated and projected to `hidden_size`;
  * classifier   — prototypical: one prototype per *trainable* class
                   (Knowns + bought CTIs), logits = -squared distance;
  * anomaly det. — a sample is "unknown" iff its distance to the nearest
                   prototype exceeds a threshold tau calibrated on the
                   quantile of own-class distances;
  * clustering   — unknown samples of a tick are grouped by single-linkage
                   connected components with radius ~ tau.

Shadow replay buffers are kept for EVERY true class (the recorder gives
us ground-truth labels, exactly like the GNS3 traffic_dict does), but
only classes in the `trainable` set contribute prototypes and gradient
steps — buying a CTI simply adds its class to that set.
"""

import copy
import random
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess(flow: torch.Tensor, pkt: Optional[torch.Tensor]):
    """Fixed input transform: flowstats are heavy-tailed counters
    (log-compress), packet bytes live in [0,255] (scale to [0,1])."""
    flow = torch.log1p(torch.clamp(flow, min=0.0))
    if pkt is not None:
        pkt = pkt / 255.0
    return flow, pkt


class SimbaEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_size
        self.use_packet_feats = cfg.use_packet_feats
        self.gru = nn.GRU(cfg.flow_feat_dim, h, num_layers=1, batch_first=True)
        in_dim = h
        if cfg.use_packet_feats:
            self.pkt_mlp = nn.Sequential(
                nn.Linear(cfg.packet_feat_dim, h), nn.ReLU(),
                nn.Linear(h, h))
            in_dim += h
        self.head = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h))

    def forward(self, flow, pkt):
        # flow: [B, T, F]; pkt: [B, P, D] or None
        _, hn = self.gru(flow)
        z = hn[-1]
        if self.use_packet_feats and pkt is not None:
            z = torch.cat([z, self.pkt_mlp(pkt.mean(dim=1))], dim=-1)
        return self.head(z)


class SimbaInference:

    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.device)
        self.encoder = SimbaEncoder(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(),
                                          lr=cfg.im_learning_rate)
        self.buffers: Dict[str, deque] = {}   # true label -> deque[(flow,pkt)]
        self.trainable: set = set()
        self.prototypes: Dict[str, torch.Tensor] = {}
        self.tau: float = float('inf')
        # stationary input-space representation (standardised raw
        # features): used for unknown clustering and for the DM state, so
        # neither depends on the drifting learned embedding.
        self.feat_mu: Optional[torch.Tensor] = None
        self.feat_sd: Optional[torch.Tensor] = None
        self.cluster_radius: Optional[float] = None
        self._rng = random.Random(cfg.seed)
        self._snapshot = None

    # ------------------------------------------------------------ buffers
    def push(self, flow, pkt, labels: List[str]):
        for i, lab in enumerate(labels):
            buf = self.buffers.setdefault(lab, deque(maxlen=self.cfg.buffer_capacity))
            buf.append((flow[i], None if pkt is None else pkt[i]))

    def buffer_len(self, label: str) -> int:
        return len(self.buffers.get(label, ()))

    def _stack_from_buffer(self, label: str, n: int):
        buf = self.buffers[label]
        picks = self._rng.sample(range(len(buf)), min(n, len(buf)))
        flow = torch.stack([buf[i][0] for i in picks])
        pkt = None
        if self.cfg.use_packet_feats and buf[picks[0]][1] is not None:
            pkt = torch.stack([buf[i][1] for i in picks])
        return flow, pkt

    def _ready(self, classes) -> List[str]:
        need = max(self.cfg.min_samples_per_class,
                   self.cfg.im_support_size + self.cfg.im_query_size)
        return [c for c in sorted(classes) if self.buffer_len(c) >= need]

    def _ready_classes(self) -> List[str]:
        return self._ready(self.trainable)

    def _ready_g1s(self) -> List[str]:
        """G1 pseudo zero-days with enough shadow samples. Their labels are
        usable at training time (they supervise novelty detection) but they
        are never Known to the game."""
        return self._ready(c for c in self.cfg.g1s if c not in self.trainable)

    # ------------------------------------------------------------ encoding
    def embed(self, flow, pkt) -> torch.Tensor:
        flow, pkt = preprocess(flow.to(self.device),
                               None if pkt is None else pkt.to(self.device))
        return self.encoder(flow, pkt)

    def input_rep(self, flow, pkt) -> torch.Tensor:
        """Standardised raw-feature vector: latest flowstats row + window
        mean + mean packet bytes. Summaries (not the flattened window) so
        that a flow's window-fill phase doesn't scatter same-class
        samples. Stationary across the whole run (unlike the learned
        embedding), so the DM can anchor identities to it."""
        x = self._raw_rep(flow, pkt)
        if self.feat_mu is not None:
            x = (x - self.feat_mu) / self.feat_sd
        return x

    def _raw_rep(self, flow, pkt) -> torch.Tensor:
        flow, pkt = preprocess(flow, pkt)
        parts = [flow[:, -1, :], flow.mean(dim=1)]
        if pkt is not None:
            parts.append(pkt.mean(dim=1))
        return torch.cat(parts, dim=1)

    def input_rep_dim(self) -> int:
        d = 2 * self.cfg.flow_feat_dim
        if self.cfg.use_packet_feats:
            d += self.cfg.packet_feat_dim
        return d

    def fit_feature_stats(self):
        """Fit the input-space standardisation on the shadow buffers."""
        xs = []
        for c in self.buffers:
            if self.buffer_len(c) < 2:
                continue
            f, p = self._stack_from_buffer(c, self.cfg.proto_samples)
            xs.append(self._raw_rep(f, p))
        if not xs:
            return
        x = torch.cat(xs)
        self.feat_mu = x.mean(dim=0)
        self.feat_sd = x.std(dim=0) + 1e-6

    def calibrate_clustering(self):
        """Calibrate the unknown-clustering radius on the G1 pseudo
        zero-days, in input space: large enough to swallow a class's own
        spread, small enough to keep distinct classes apart."""
        g1s = self._ready_g1s()
        if self.feat_mu is None or not g1s:
            return
        within, cents = [], []
        for c in g1s:
            f, p = self._stack_from_buffer(c, self.cfg.proto_samples)
            x = self.input_rep(f, p)
            within.append(torch.quantile(torch.pdist(x), 0.9).item())
            cents.append(x.mean(dim=0))
        radius = 1.25 * max(within)
        if len(cents) > 1:
            between = torch.pdist(torch.stack(cents)).min().item()
            radius = min(radius, 0.6 * between)
        self.cluster_radius = radius * self.cfg.cluster_radius_factor

    # ------------------------------------------------------------ training
    def train_step(self) -> Optional[float]:
        """One prototypical-episode gradient step.

        Trainable classes AND G1 pseudo zero-days each form their own
        prototype in the episode's cross-entropy (so distinct unknown
        classes stay distinct — this is what makes clustering work); G1
        queries are additionally repelled from the *Known* prototypes by
        a margin, which is what the unknown threshold tau is calibrated
        against.
        """
        cfg = self.cfg
        known_classes = self._ready_classes()
        if len(known_classes) < 2:
            return None
        g1_classes = self._ready_g1s()
        classes = known_classes + g1_classes
        s, q = cfg.im_support_size, cfg.im_query_size

        flows, pkts = [], []
        for c in classes:
            f, p = self._stack_from_buffer(c, s + q)
            flows.append(f)
            pkts.append(p)
        flow = torch.cat(flows)
        pkt = torch.cat(pkts) if pkts[0] is not None else None

        self.encoder.train()
        z = self.embed(flow, pkt).view(len(classes), s + q, -1)
        protos = z[:, :s].mean(dim=1)                      # [C, H]
        queries = z[:, s:].reshape(len(classes) * q, -1)   # [C*q, H]
        logits = -torch.cdist(queries, protos) ** 2
        targets = torch.arange(len(classes), device=self.device) \
                       .repeat_interleave(q)
        loss = F.cross_entropy(logits, targets)

        if g1_classes and cfg.ad_repulsion_weight > 0:
            g1_queries = queries[len(known_classes) * q:]
            d_to_known = torch.cdist(g1_queries, protos[:len(known_classes)])
            d_min = d_to_known.min(dim=1).values
            loss = loss + cfg.ad_repulsion_weight * \
                F.relu(cfg.ad_repulsion_margin - d_min).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def calibrate(self):
        """Recompute prototypes and the unknown threshold tau.

        tau is chosen to best separate Known samples' nearest-prototype
        distances from the G1 pseudo zero-days' ones (balanced accuracy
        over candidate thresholds). Without G1 data it falls back to a
        quantile-of-own-distances rule.
        """
        self.encoder.eval()
        if self.feat_mu is None:
            self.fit_feature_stats()
        if self.cluster_radius is None:
            self.calibrate_clustering()
        classes = self._ready_classes()
        if not classes:
            return
        protos = {}
        embeds = {}
        for c in classes:
            f, p = self._stack_from_buffer(c, self.cfg.proto_samples)
            z = self.embed(f, p)
            protos[c] = z.mean(dim=0)
            embeds[c] = z
        self.prototypes = protos
        proto_mat = torch.stack([protos[c] for c in classes])

        known_d = torch.cat([
            torch.cdist(embeds[c], proto_mat).min(dim=1).values
            for c in classes])

        g1_d = []
        for c in self._ready_g1s():
            f, p = self._stack_from_buffer(c, self.cfg.proto_samples)
            z = self.embed(f, p)
            g1_d.append(torch.cdist(z, proto_mat).min(dim=1).values)

        if g1_d:
            g1_d = torch.cat(g1_d)
            cands = torch.cat([known_d, g1_d]).unique()
            bal_acc = ((known_d.unsqueeze(0) <= cands.unsqueeze(1)).float().mean(1)
                       + (g1_d.unsqueeze(0) > cands.unsqueeze(1)).float().mean(1)) / 2
            self.tau = cands[bal_acc.argmax()].item()
        else:
            self.tau = torch.quantile(known_d, self.cfg.ad_quantile).item() \
                * self.cfg.ad_margin

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def infer(self, flow, pkt):
        """Returns (z, pred_labels, min_dist, unknown_mask).

        pred_labels[i] is the nearest trainable prototype's class name (or
        None while no prototypes exist); unknown_mask[i] is True when the
        sample looks like nothing the IM knows.
        """
        self.encoder.eval()
        z = self.embed(flow, pkt)
        n = z.shape[0]
        if not self.prototypes:
            return z, [None] * n, torch.full((n,), float('inf')), \
                torch.ones(n, dtype=torch.bool)
        names = list(self.prototypes.keys())
        proto_mat = torch.stack([self.prototypes[c] for c in names])
        dists = torch.cdist(z, proto_mat)          # [n, C]
        min_dist, arg = dists.min(dim=1)
        preds = [names[a] for a in arg.tolist()]
        unknown = min_dist > self.tau
        return z, preds, min_dist, unknown

    @staticmethod
    def cluster(z: torch.Tensor, radius: float) -> torch.Tensor:
        """Single-linkage connected components under `radius` (euclid)."""
        n = z.shape[0]
        adj = torch.cdist(z, z) <= radius
        labels = torch.full((n,), -1, dtype=torch.long)
        current = 0
        for i in range(n):
            if labels[i] >= 0:
                continue
            stack = [i]
            labels[i] = current
            while stack:
                j = stack.pop()
                for k in torch.nonzero(adj[j]).flatten().tolist():
                    if labels[k] < 0:
                        labels[k] = current
                        stack.append(k)
            current += 1
        return labels

    # ------------------------------------------------------------ lifecycle
    def snapshot(self):
        """Freeze the pretrained state so every episode restarts from it."""
        self._snapshot = {
            'encoder': copy.deepcopy(self.encoder.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            'prototypes': {k: v.clone() for k, v in self.prototypes.items()},
            'tau': self.tau,
            'trainable': set(self.trainable),
            'buffers': {k: list(v) for k, v in self.buffers.items()},
            'feat_mu': self.feat_mu,
            'feat_sd': self.feat_sd,
            'cluster_radius': self.cluster_radius,
        }

    def restore(self):
        if self._snapshot is None:
            return
        s = self._snapshot
        self.encoder.load_state_dict(s['encoder'])
        self.optimizer.load_state_dict(s['optimizer'])
        self.prototypes = {k: v.clone() for k, v in s['prototypes'].items()}
        self.tau = s['tau']
        self.trainable = set(s['trainable'])
        self.buffers = {k: deque(v, maxlen=self.cfg.buffer_capacity)
                        for k, v in s['buffers'].items()}
        self.feat_mu = s['feat_mu']
        self.feat_sd = s['feat_sd']
        self.cluster_radius = s['cluster_radius']

    def save(self, path: str):
        torch.save({'encoder': self.encoder.state_dict(),
                    'tau': self.tau}, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(state['encoder'])
        self.tau = state.get('tau', float('inf'))
