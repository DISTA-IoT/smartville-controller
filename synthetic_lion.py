#!/usr/bin/env python3
"""
synthetic_lion.py — Synthetic LION: DDQN vs. DAI-P vs. Break-Even
on a 2-D Gaussian CTI curriculum-learning task.

Self-contained: no smartville imports.  Run with:
    python synthetic_lion.py [--episodes 300] [--seeds 5] [--no-plot]

Why DAI-P should beat DDQN here
---------------------------------
Unknown class A (malicious) overlaps in 2-D with Known Benign 0, so the
inference module initially classifies A-type flows as benign with HIGH
confidence (low anomaly score).  The agent therefore accepts them and bleeds
budget.  Once label-A is purchased, the inference module immediately adds
A's prototype to its known-class roster: A-type flows flip from
"high-confidence benign" to "high-confidence malicious A", producing the
LARGEST possible one-step change in the proprioceptive state
(known-class confidence, anomaly score, n_labels_bought, budget).
Because this transition is maximally surprising, the DAI-P transition model
carries a high residual prediction error—high perceptive-epistemic gain—for
the buy-label-A action, driving the agent to execute it early.

DDQN explores with Boltzmann sampling over Q-values and relies on
discovering the reward signal of label-A through trial and error.  It can
eventually converge, but it loses critical budget in the early episodes.

The Break-Even agent buys the label with the HIGHEST anomaly score first
(the natural heuristic).  Unknown A has the LOWEST anomaly score (it looks
benign), so the Break-Even agent consistently de-prioritises it—buying B or
C first—and suffers the same budget haemorrhage as naive DDQN.

Architecture overview
---------------------
    DataGenerator      7 2-D Gaussians (3 known + 4 unknown)
    InferenceModule    prototypical classifier + soft anomaly/cluster head
                       updates online via EMA; label buying adds a prototype
    SyntheticLIONEnv   budget episode (block=0 / accept=1 / buy-label=2)
    TwoStreamNet       shared backbone for Q-net / EFE-net  (5 extero + 6 proprio)
    TransitionNet      predicts next proprioceptive state from (state, action)
    DDQNAgent          Double-DQN with Boltzmann sampling (no epistemic gain)
    DAIPAgent          DAI-P: reward + perceptive-epistemic gain from transition MSE
    BreakEvenAgent     rule-based: buy highest-anomaly-score unlabelled cluster when affordable
"""

from __future__ import annotations

import argparse
import random
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Global hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # Gaussian data
    input_dim    = 2,
    hidden_dim   = 24,       # encoder output / prototype dimension
    n_known      = 3,
    n_unknown    = 4,
    proto_temp   = 3.0,      # softmax temperature for prototypical assignment

    # Episode
    # Budget is tight: total label cost = 5+6+4+4 = 19 > init_budget of 15.
    # The agent can afford at most 2-3 labels. It MUST choose which ones.
    # Break-Even always buys the highest-anomaly-score label first (= B),
    # spending 6 of 15 up front; then can only afford C (4) and is stuck with
    # 5 left—not enough for A (5 exact) if any A-flow has already drained budget.
    # DAI-P's epistemic gain pushes it to buy A first despite A's low anomaly score.
    max_steps    = 150,
    init_budget  = 15.0,
    min_budget   = 0.0,
    max_budget   = 40.0,

    # Label prices  (A, B, C, D)
    label_prices = [5.0, 6.0, 4.0, 4.0],

    # Rewards: (uninformed, informed) per action per class type
    # known-class flows always use informed rewards
    r_accept_benign_informed    =  2.0,
    r_block_benign_informed     = -2.5,
    r_accept_malicious_informed = -5.0,
    r_block_malicious_informed  =  2.5,

    # unknown flows before label is bought
    # A: overlapping-malicious, B: separate-malicious, C: separate-benign, D: near-malicious-benign
    r_accept_A_uninformed =  -7.0,   # agent is fooled: thinks it's benign, accepts
    r_block_A_uninformed  =   0.5,   # accidental good block
    r_accept_B_uninformed =  -3.5,   # anomaly visible, but accepting possible
    r_block_B_uninformed  =   1.5,   # anomaly detected and blocked
    r_accept_C_uninformed =   0.5,   # benign but looks anomalous → OK to accept
    r_block_C_uninformed  =  -1.5,   # false positive on benign
    r_accept_D_uninformed =   0.5,   # benign, near malicious → sometimes accepted
    r_block_D_uninformed  =  -1.0,   # false positive

    buy_invalid_penalty  = -0.3,

    # RL hyper-params
    gamma        = 0.95,
    lr           = 4e-4,
    batch_size   = 64,
    memory_size  = 6000,
    target_update = 20,       # steps between target-net hard updates
    temperature  = 2.0,       # Boltzmann softmax temperature
    min_memory_to_train = 128,

    # DAI-P
    epistemic_weight = 0.5,   # weight on perceptive-epistemic gain in EFE target

    # Inference module
    anomaly_threshold = 1.8,  # hidden-space L2 distance → anomaly if >
    proto_ema         = 0.08, # EMA coefficient for unknown prototype update

    # Pretraining
    pretrain_epochs      = 120,
    pretrain_n_per_class = 200,
    pretrain_lr          = 1e-3,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Data generator
# ──────────────────────────────────────────────────────────────────────────────

class DataGenerator:
    """
    Seven 2-D Gaussians:

    Known (pre-trained):
      K0 – benign,    centre (−2, 0),   σ=0.40
      K1 – benign,    centre ( 2, 0),   σ=0.40
      K2 – malicious, centre ( 0,−2),   σ=0.40

    Unknown (label must be bought):
      UA – malicious, centre (−1.5, 0.4), σ=0.35  ← overlaps K0!  HIGHEST epistemic value
      UB – malicious, centre ( 0,   3.0), σ=0.45  ← clearly anomalous
      UC – benign,    centre ( 4.0, 1.0), σ=0.45  ← clearly anomalous, false-positive risk
      UD – benign,    centre ( 2.4,−1.0), σ=0.35  ← near K2, misclassified-malicious risk
    """
    N_KNOWN   = 3
    N_UNKNOWN = 4

    KNOWN_MEANS  = np.array([[-2.0,  0.0], [ 2.0,  0.0], [ 0.0, -2.0]])
    KNOWN_STDS   = [0.40, 0.40, 0.40]
    KNOWN_TYPES  = ['benign', 'benign', 'malicious']

    UNKNOWN_MEANS = np.array([[-1.5,  0.4], [ 0.0,  3.0], [ 4.0,  1.0], [ 2.4, -1.0]])
    UNKNOWN_STDS  = [0.35, 0.45, 0.45, 0.35]
    UNKNOWN_TYPES = ['malicious', 'malicious', 'benign', 'benign']
    # Readable names for logging
    UNKNOWN_NAMES = ['A(mal,overlap)', 'B(mal,sep)', 'C(ben,sep)', 'D(ben,nearmal)']

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def sample(self, class_id: Optional[int] = None):
        """Returns (x, class_id, is_known, unknown_id, class_type)."""
        n_total = self.N_KNOWN + self.N_UNKNOWN
        if class_id is None:
            class_id = self.rng.randint(0, n_total)

        if class_id < self.N_KNOWN:
            k = class_id
            x = self.rng.normal(self.KNOWN_MEANS[k], self.KNOWN_STDS[k])
            return x, class_id, True, None, self.KNOWN_TYPES[k]
        else:
            u = class_id - self.N_KNOWN
            x = self.rng.normal(self.UNKNOWN_MEANS[u], self.UNKNOWN_STDS[u])
            return x, class_id, False, u, self.UNKNOWN_TYPES[u]

    def sample_known_batch(self, n_per_class: int = 200):
        """Returns (X, y) for supervised pretraining on known classes."""
        Xs, ys = [], []
        for k in range(self.N_KNOWN):
            xs = self.rng.normal(self.KNOWN_MEANS[k], self.KNOWN_STDS[k],
                                 size=(n_per_class, 2))
            Xs.append(xs)
            ys.extend([k] * n_per_class)
        return np.vstack(Xs), np.array(ys, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Inference module (fully differentiable)
# ──────────────────────────────────────────────────────────────────────────────

class InferenceModule(nn.Module):
    """
    Prototypical classifier with a differentiable soft-anomaly / cluster head.

    Known prototypes are supervised (pretrained).  When a CTI label is bought
    for unknown cluster uid, that cluster's EMA prototype is added to the
    known-prototype list so future flows are immediately classified correctly.

    Unknown cluster prototypes are maintained as EMA buffers (no backprop
    needed for them); the encoder IS trained during pretraining.

    Anomaly detection: a flow is anomalous if its L2 distance to the nearest
    known prototype exceeds `anomaly_threshold` in hidden space.
    Unknown clustering: differentiable soft-assignment via temperature softmax
    over distances to learnable unknown-cluster prototypes.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        d_in  = cfg['input_dim']
        d_h   = cfg['hidden_dim']
        n_k   = cfg['n_known']
        n_u   = cfg['n_unknown']
        self.proto_temp       = cfg['proto_temp']
        self.anomaly_threshold = cfg['anomaly_threshold']
        self.n_known_initial  = n_k
        self.n_unknown        = n_u

        # Encoder: 2-D → hidden
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_h * 2),
            nn.ReLU(),
            nn.Linear(d_h * 2, d_h),
        )

        # Known-class prototypes (supervised, grown when labels bought)
        self.known_protos = nn.Parameter(torch.randn(n_k, d_h) * 0.1)
        # Corresponding class types (extended when label bought)
        self.known_types_list: List[str] = []   # filled at env reset

        # Unknown-cluster EMA prototypes (not trained by backprop)
        self.register_buffer('unk_protos',  torch.randn(n_u, d_h) * 0.1)
        # Running stats for proprioceptive state
        self.register_buffer('unk_conf',    torch.zeros(n_u))
        self.register_buffer('unk_size',    torch.zeros(n_u))

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.encoder(x)

    def known_classify(self, h: torch.Tensor):
        """Returns (pred_class, confidence, distances) for known prototypes."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        dists = torch.cdist(h, self.known_protos)       # (B, n_known)
        probs = F.softmax(-dists * self.proto_temp, dim=-1)
        pred  = probs.argmax(-1)
        conf  = probs.max(-1).values
        return pred, conf, dists

    def anomaly_detect(self, dists_to_known: torch.Tensor):
        """Returns (anomaly_score, is_anomaly_mask)."""
        min_dist   = dists_to_known.min(-1).values
        is_anomaly = min_dist > self.anomaly_threshold
        return min_dist, is_anomaly

    def unk_cluster(self, h: torch.Tensor):
        """Returns (cluster_id, cluster_conf, soft_weights) using current unk_protos."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        dists   = torch.cdist(h, self.unk_protos)
        weights = F.softmax(-dists * self.proto_temp, dim=-1)
        cid     = weights.argmax(-1)
        cconf   = weights.max(-1).values
        return cid, cconf, weights

    @torch.no_grad()
    def update_unk_proto(self, h: torch.Tensor, cid: int, ema: float):
        self.unk_protos[cid] = (1 - ema) * self.unk_protos[cid] + ema * h.squeeze()
        self.unk_size[cid]   = min(self.unk_size[cid] + 1.0, 200.0)

    @torch.no_grad()
    def update_unk_conf(self, cid: int, conf: float):
        self.unk_conf[cid] = 0.9 * self.unk_conf[cid] + 0.1 * conf

    def add_known_proto(self, proto: torch.Tensor, class_type: str):
        """Promote an unknown cluster to known when its label is bought."""
        proto = proto.detach().unsqueeze(0)
        self.known_protos = nn.Parameter(
            torch.cat([self.known_protos.detach(), proto], dim=0)
        )
        self.known_types_list.append(class_type)

    @torch.no_grad()
    def init_unk_protos_with_data(self, data_gen: DataGenerator, n_samples: int = 30):
        """
        Warm-start unknown cluster protos using the pretrained encoder applied to
        samples drawn near each true unknown cluster centre.
        This avoids cold-start misassignment of flows at episode start while
        keeping the prototype positions random enough to require online updating.
        """
        for u in range(self.n_unknown):
            xs = np.random.normal(data_gen.UNKNOWN_MEANS[u],
                                  data_gen.UNKNOWN_STDS[u] * 1.5,
                                  size=(n_samples, 2))
            h = self.encode(torch.FloatTensor(xs))
            self.unk_protos[u] = h.mean(0)

    def reset_episode_state(self, original_n_known: int, data_gen: DataGenerator):
        """
        Roll back added prototypes + re-initialise unknown cluster protos.
        Called at the start of every episode.
        """
        # Trim known protos back to the original pretrained set
        with torch.no_grad():
            self.known_protos = nn.Parameter(
                self.known_protos.detach()[:original_n_known].clone()
            )
        self.known_types_list = list(data_gen.KNOWN_TYPES[:original_n_known])
        # Warm-start unknown protos so flows map to correct clusters from step 1
        self.init_unk_protos_with_data(data_gen)
        self.unk_conf.zero_()
        self.unk_size.zero_()

    def forward(self, x: torch.Tensor) -> dict:
        h                          = self.encode(x)
        pred_k, conf_k, dists_k   = self.known_classify(h)
        anorm_score, is_anom       = self.anomaly_detect(dists_k)
        cid, cconf, cweights       = self.unk_cluster(h)
        return dict(
            h=h, pred_k=pred_k, conf_k=conf_k, dists_k=dists_k,
            anorm_score=anorm_score, is_anom=is_anom,
            cid=cid, cconf=cconf, cweights=cweights,
        )


def pretrain_inference(inf: InferenceModule, dg: DataGenerator, cfg: dict) -> None:
    """Supervised prototypical loss on known classes."""
    opt = optim.Adam(inf.parameters(), lr=cfg['pretrain_lr'])
    inf.train()
    X_np, y_np = dg.sample_known_batch(cfg['pretrain_n_per_class'])
    X = torch.FloatTensor(X_np)
    y = torch.LongTensor(y_np)
    for _ in range(cfg['pretrain_epochs']):
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        h = inf.encode(X)
        _, _, dists = inf.known_classify(h)
        log_p = F.log_softmax(-dists * inf.proto_temp, dim=-1)
        loss  = F.nll_loss(log_p, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    inf.eval()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Environment
# ──────────────────────────────────────────────────────────────────────────────

# State layout: [extero(5), proprio(6)] = 11-D total
#   extero:  known_pred_norm, known_conf, anorm_score_norm, unk_proto_x_norm, unk_proto_y_norm
#   proprio: unk_cluster_conf, n_labels_bought_frac, budget_frac, label_price_frac,
#            time_frac, known_conf_again  (second known_conf useful for transition net)
STATE_DIM   = 11
PROPRIO_DIM = 6
ACTION_SIZE = 3   # 0=block, 1=accept, 2=buy-label


class SyntheticLIONEnv:
    """
    Episode environment.

    Each step one flow is sampled uniformly from all 7 classes, the inference
    module classifies/clusters it, and the agent chooses one of 3 actions.

    A key subtlety: when the agent buys a label for unknown cluster uid,
    the cluster's current EMA prototype is immediately promoted to a
    known prototype.  This creates a LARGE one-step change in the
    proprioceptive state: the known-class confidence and anomaly score of
    future flows from that cluster flip dramatically.  The DAI-P transition
    model's prediction error peaks for this transition, contributing high
    perceptive-epistemic gain and directing the agent to explore label
    purchases for the most confusing/surprising clusters first.
    """

    def __init__(self, data_gen: DataGenerator, inf_mod: InferenceModule, cfg: dict):
        self.dg     = data_gen
        self.inf    = inf_mod
        self.cfg    = cfg
        self.n_unk  = cfg['n_unknown']
        self.prices = cfg['label_prices']

        # Store pretrained known-proto count for episode reset
        self._pretrained_n_known = inf_mod.known_protos.shape[0]

    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        self.budget        = self.cfg['init_budget']
        self.t             = 0
        self.labels_bought = [False] * self.n_unk
        self.inf.reset_episode_state(self._pretrained_n_known, self.dg)
        self._step_flow()
        return self._state()

    # ------------------------------------------------------------------
    def _step_flow(self):
        """Sample next flow and run inference.  Caches result."""
        x, cid, is_k, uid, ctype = self.dg.sample()
        xt = torch.FloatTensor(x).unsqueeze(0)
        with torch.no_grad():
            r = self.inf(xt)

        # If is_known: use the (possibly extended) known-proto set
        # uid == None  → known class; otherwise unknown cluster
        self._flow = dict(
            x=x, class_id=cid, is_known=is_k,
            unknown_id=uid, class_type=ctype,
            h=r['h'],
            pred_k=r['pred_k'].item(),
            conf_k=r['conf_k'].item(),
            anorm_score=r['anorm_score'].item(),
            is_anom=r['is_anom'].item(),
            cid=r['cid'].item() if uid is not None else -1,
            cconf=r['cconf'].item() if uid is not None else r['conf_k'].item(),
        )
        # If unknown, update cluster EMA with this sample
        if uid is not None:
            cid_t = r['cid']
            self.inf.update_unk_proto(r['h'], cid_t.item(), self.cfg['proto_ema'])
            self.inf.update_unk_conf(cid_t.item(), r['cconf'].item())

    # ------------------------------------------------------------------
    def _state(self) -> torch.Tensor:
        f   = self._flow
        cfg = self.cfg
        n_u = self.n_unk

        # Exteroceptive
        known_pred_norm    = f['pred_k'] / max(self.inf.known_protos.shape[0] - 1, 1)
        known_conf_norm    = f['conf_k']                           # already 0-1
        anorm_norm         = min(f['anorm_score'] / 5.0, 1.0)

        if f['unknown_id'] is not None:
            up = self.inf.unk_protos[f['cid']].detach().numpy()
        else:
            up = self.inf.known_protos[f['pred_k']].detach().numpy()
        proto_x = float(up[0]) / 5.0
        proto_y = float(up[1]) / 5.0

        # Proprioceptive
        unk_cconf      = f['cconf']
        n_labels_frac  = sum(self.labels_bought) / n_u
        budget_frac    = self.budget / cfg['init_budget']
        uid            = f['unknown_id']
        if uid is not None and not self.labels_bought[uid]:
            price_frac = self.prices[uid] / cfg['init_budget']
        else:
            price_frac = 0.0
        time_frac      = self.t / cfg['max_steps']

        extero = [known_pred_norm, known_conf_norm, anorm_norm, proto_x, proto_y]
        proprio = [unk_cconf, n_labels_frac, budget_frac, price_frac, time_frac, known_conf_norm]

        return torch.FloatTensor(extero + proprio)

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        f   = self._flow
        cfg = self.cfg
        uid = f['unknown_id']

        reward = 0.0

        # ── Action 2: buy label ──────────────────────────────────────
        if action == 2:
            if f['is_known'] or uid is None or self.labels_bought[uid]:
                reward = cfg['buy_invalid_penalty']
            elif self.budget < self.prices[uid]:
                reward = cfg['buy_invalid_penalty']
            else:
                price  = self.prices[uid]
                reward = -price * 0.05          # tiny immediate cost signal
                self.budget -= price
                self.labels_bought[uid] = True
                # Promote cluster prototype → known
                self.inf.add_known_proto(
                    self.inf.unk_protos[uid],
                    self.dg.UNKNOWN_TYPES[uid]
                )

        # ── Action 0: block ──────────────────────────────────────────
        elif action == 0:
            if f['is_known']:
                ctype = self.inf.known_types_list[f['pred_k']]
                reward = (cfg['r_block_malicious_informed'] if ctype == 'malicious'
                          else cfg['r_block_benign_informed'])
            elif uid is not None and self.labels_bought[uid]:
                ctype = self.dg.UNKNOWN_TYPES[uid]
                reward = (cfg['r_block_malicious_informed'] if ctype == 'malicious'
                          else cfg['r_block_benign_informed'])
            else:
                # Uninformed block of unknown cluster
                key = f'r_block_{self.dg.UNKNOWN_NAMES[uid][0]}_uninformed'
                reward = cfg.get(key, 0.5)

        # ── Action 1: accept ─────────────────────────────────────────
        elif action == 1:
            if f['is_known']:
                ctype = self.inf.known_types_list[f['pred_k']]
                reward = (cfg['r_accept_benign_informed'] if ctype == 'benign'
                          else cfg['r_accept_malicious_informed'])
            elif uid is not None and self.labels_bought[uid]:
                ctype = self.dg.UNKNOWN_TYPES[uid]
                reward = (cfg['r_accept_benign_informed'] if ctype == 'benign'
                          else cfg['r_accept_malicious_informed'])
            else:
                key = f'r_accept_{self.dg.UNKNOWN_NAMES[uid][0]}_uninformed'
                reward = cfg.get(key, -1.0)

        self.budget += reward
        self.t      += 1

        done = (
            self.budget <= cfg['min_budget']
            or self.budget >= cfg['max_budget']
            or self.t     >= cfg['max_steps']
        )
        win  = self.budget >= cfg['max_budget']

        # Hard clamp budget so it doesn't drift wildly
        self.budget = max(cfg['min_budget'], min(self.budget, cfg['max_budget'] + 5.0))

        if not done:
            self._step_flow()

        info = dict(
            budget       = self.budget,
            win          = win,
            labels       = list(self.labels_bought),
            n_labels     = sum(self.labels_bought),
            reward       = reward,
        )
        return self._state(), reward, done, info


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Neural modules (two-stream: extero ‖ proprio)
# ──────────────────────────────────────────────────────────────────────────────

class TwoStreamNet(nn.Module):
    """
    Shared backbone for DQN / EFE-net.
    Input: state (STATE_DIM = 11) split into extero (5) and proprio (6).
    Output: logits for each action (ACTION_SIZE = 3).
    """
    def __init__(self, out_dim: int = ACTION_SIZE, hidden: int = 48):
        super().__init__()
        extero_dim  = STATE_DIM - PROPRIO_DIM   # 5
        proprio_dim = PROPRIO_DIM               # 6

        self.ext_fc1 = nn.Linear(extero_dim,  hidden)
        self.ext_fc2 = nn.Linear(hidden,      hidden // 2)
        self.pro_fc1 = nn.Linear(proprio_dim, hidden)
        self.pro_fc2 = nn.Linear(hidden,      hidden // 2 * 2)
        self.out     = nn.Linear(hidden // 2 + hidden // 2 * 2 // 2, out_dim)
        self._h2 = hidden // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        e = x[:, :STATE_DIM - PROPRIO_DIM]
        p = x[:, STATE_DIM - PROPRIO_DIM:]
        e = F.relu(self.ext_fc1(e))
        e = F.relu(self.ext_fc2(e))
        p = F.relu(self.pro_fc1(p))
        p = F.relu(self.pro_fc2(p))
        # take first half of p stream
        p = p[:, :self._h2]
        return self.out(torch.cat([e, p], dim=-1))


class TransitionNet(nn.Module):
    """
    Predicts the next PROPRIOCEPTIVE state given (state, action-one-hot).
    Used only by DAI-P.
    """
    def __init__(self, hidden: int = 48):
        super().__init__()
        in_dim = STATE_DIM + ACTION_SIZE
        self.net = nn.Sequential(
            nn.Linear(in_dim,  hidden),
            nn.ReLU(),
            nn.Linear(hidden,  hidden),
            nn.ReLU(),
            nn.Linear(hidden,  PROPRIO_DIM),
        )

    def forward(self, state: torch.Tensor, action_oh: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action_oh], dim=-1))


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Replay buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: List = [None] * capacity
        self.pos = 0
        self.size = 0

    def push(self, *transition):
        self.buf[self.pos] = tuple(t.detach().clone() if isinstance(t, torch.Tensor)
                                   else t for t in transition)
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        idx = random.sample(range(self.size), n)
        return zip(*[self.buf[i] for i in idx])

    def __len__(self):
        return self.size


# ──────────────────────────────────────────────────────────────────────────────
# 6.  DDQN agent
# ──────────────────────────────────────────────────────────────────────────────

class DDQNAgent:
    """
    Double-DQN with Boltzmann sampling (no epistemic gain).
    Identical exploration mechanism to the DDQN baseline in the paper.
    """

    def __init__(self, cfg: dict):
        self.gamma       = cfg['gamma']
        self.batch_size  = cfg['batch_size']
        self.temperature = cfg['temperature']
        self.target_upd  = cfg['target_update']
        self.min_mem     = cfg['min_memory_to_train']

        self.net    = TwoStreamNet()
        self.target = TwoStreamNet()
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()
        self.opt    = optim.Adam(self.net.parameters(), lr=cfg['lr'])
        self.buf    = ReplayBuffer(cfg['memory_size'])
        self._steps = 0

    def act(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            q     = self.net(state).squeeze()
            probs = F.softmax(self.temperature * q, dim=-1)
        return torch.multinomial(probs, 1).item()

    def push(self, s, a, r, s2, done):
        self.buf.push(s, torch.tensor(a), torch.tensor(r, dtype=torch.float32),
                      s2, torch.tensor(done, dtype=torch.bool))

    def train_step(self):
        if len(self.buf) < self.min_mem:
            return
        states, actions, rewards, next_states, dones = self.buf.sample(self.batch_size)

        S  = torch.stack(list(states))
        A  = torch.stack(list(actions))
        R  = torch.stack(list(rewards)).unsqueeze(1)
        S2 = torch.stack(list(next_states))
        D  = torch.stack(list(dones)).unsqueeze(1)

        with torch.no_grad():
            # DDQN: online net selects, target net evaluates
            a_next  = self.net(S2).argmax(1, keepdim=True)
            q_next  = self.target(S2).gather(1, a_next)
            targets_full = self.net(S).detach()
            targets_full[range(self.batch_size), A] = (
                R.squeeze() + self.gamma * q_next.squeeze() * ~D.squeeze()
            )

        self.net.train()
        loss = F.mse_loss(self.net(S), targets_full)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self._steps += 1
        if self._steps % self.target_upd == 0:
            self.target.load_state_dict(self.net.state_dict())


# ──────────────────────────────────────────────────────────────────────────────
# 7.  DAI-P agent  (perceptive-epistemic gain + pragmatic reward)
# ──────────────────────────────────────────────────────────────────────────────

class DAIPAgent:
    """
    DAI-P: the canonical value-based Active Inference agent from the paper.

    EFE target = reward  +  epistemic_weight * perceptive_epistemic_gain
                         +  γ * EFE(s', a*)

    Perceptive-epistemic gain = 0.5 * ||next_proprio - TransitionNet(s,a)||²

    The transition net is trained concurrently (supervised MSE on proprio
    next-state prediction).  Early in training its residual error is high for
    buy-label-A because A-type flows flip dramatically from "high-conf benign"
    to "high-conf malicious A" in the proprioceptive state—the most surprising
    one-step transition in this environment—so the EFE objective assigns higher
    intrinsic value to buy-label-A and purchases it earlier.
    """

    def __init__(self, cfg: dict):
        self.gamma        = cfg['gamma']
        self.batch_size   = cfg['batch_size']
        self.temperature  = cfg['temperature']
        self.target_upd   = cfg['target_update']
        self.min_mem      = cfg['min_memory_to_train']
        self.eps_w        = cfg['epistemic_weight']

        self.efe_net    = TwoStreamNet()           # critic (−EFE values)
        self.efe_target = TwoStreamNet()
        self.efe_target.load_state_dict(self.efe_net.state_dict())
        self.efe_target.eval()
        self.efe_opt    = optim.Adam(self.efe_net.parameters(), lr=cfg['lr'])

        self.trans_net  = TransitionNet()
        self.trans_opt  = optim.Adam(self.trans_net.parameters(), lr=cfg['lr'])

        self.buf    = ReplayBuffer(cfg['memory_size'])
        self._steps = 0
        self._eye   = torch.eye(ACTION_SIZE)

    def act(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            nefe  = self.efe_net(state).squeeze()
            probs = F.softmax(self.temperature * nefe, dim=-1)
        return torch.multinomial(probs, 1).item()

    def push(self, s, a, r, s2, done):
        self.buf.push(s, torch.tensor(a), torch.tensor(r, dtype=torch.float32),
                      s2, torch.tensor(done, dtype=torch.bool))

    def train_step(self):
        if len(self.buf) < self.min_mem:
            return
        states, actions, rewards, next_states, dones = self.buf.sample(self.batch_size)

        S   = torch.stack(list(states))
        A   = torch.stack(list(actions))
        R   = torch.stack(list(rewards)).unsqueeze(1)
        S2  = torch.stack(list(next_states))
        D   = torch.stack(list(dones)).unsqueeze(1)
        AOH = self._eye[A]                         # one-hot actions  (B, 3)

        next_proprio = S2[:, STATE_DIM - PROPRIO_DIM:]

        # ── 1. Compute perceptive epistemic gain (no gradient through trans_net here) ──
        with torch.no_grad():
            pred_next = self.trans_net(S, AOH)
            epist = 0.5 * ((next_proprio - pred_next) ** 2).sum(dim=1, keepdim=True)

        # ── 2. Build EFE targets ──────────────────────────────────────────────
        with torch.no_grad():
            # DDQN-style: online net selects next action
            a_next      = self.efe_net(S2).argmax(1, keepdim=True)
            q_next      = self.efe_target(S2).gather(1, a_next)
            efe_targets = self.efe_net(S).detach().clone()
            efe_targets[range(self.batch_size), A] = (
                R.squeeze()
                + self.eps_w * epist.squeeze()
                + self.gamma * q_next.squeeze() * ~D.squeeze()
            )

        # ── 3. Train EFE network ──────────────────────────────────────────────
        self.efe_net.train()
        efe_loss = F.mse_loss(self.efe_net(S), efe_targets)
        self.efe_opt.zero_grad()
        efe_loss.backward()
        self.efe_opt.step()

        # ── 4. Train transition network (supervised, next proprio) ───────────
        self.trans_net.train()
        pred_next2  = self.trans_net(S, AOH)
        trans_loss  = F.mse_loss(pred_next2, next_proprio)
        self.trans_opt.zero_grad()
        trans_loss.backward()
        self.trans_opt.step()

        self._steps += 1
        if self._steps % self.target_upd == 0:
            self.efe_target.load_state_dict(self.efe_net.state_dict())


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Break-Even rule-based agent
# ──────────────────────────────────────────────────────────────────────────────

class BreakEvenAgent:
    """
    Deterministic heuristic:
      – Accept flows that the inference module classifies as known-benign
        (or labelled-benign unknown).
      – Block flows classified as known-malicious / labelled-malicious unknown.
      – For unlabelled unknown clusters:
          • Buy the label if affordable and the cluster has the highest
            ANOMALY SCORE among available unlabelled clusters.
            (Rationale: high anomaly score → clearly anomalous → must be dangerous.)
          • Otherwise block if anomaly score > 0.5, else accept.

    This heuristic is deliberate: Unknown-A (the most dangerous cluster) has
    the LOWEST anomaly score because it overlaps with Known-Benign-0.
    The break-even agent therefore buys B's label first (highest anomaly score),
    then C or D (all cheaper), and reaches A last—if at all within budget.
    """

    def act(self, state: torch.Tensor, env: SyntheticLIONEnv) -> int:
        f   = env._flow
        uid = f['unknown_id']

        # Known or labelled-unknown flow: use inference prediction
        if f['is_known'] or (uid is not None and env.labels_bought[uid]):
            inf_type = env.inf.known_types_list[f['pred_k']]
            return 0 if inf_type == 'malicious' else 1

        # Unknown, not yet labelled
        if uid is None:
            return 1  # shouldn't happen

        # Check if we can and should buy a label
        avail_uids = [i for i in range(env.n_unk) if not env.labels_bought[i]]
        if avail_uids:
            # Pick the unlabelled cluster with HIGHEST anomaly score as proxy for danger
            scores = {i: env.inf.unk_conf[i].item() for i in avail_uids}
            # Use running average anomaly distance instead of cluster confidence
            unk_anom_scores = {}
            for i in avail_uids:
                # Proxy: distance of cluster's prototype to nearest known proto
                up = env.inf.unk_protos[i].unsqueeze(0)
                with torch.no_grad():
                    d = torch.cdist(up, env.inf.known_protos).min().item()
                unk_anom_scores[i] = d
            best_uid = max(unk_anom_scores, key=unk_anom_scores.__getitem__)
            price    = env.prices[best_uid]
            # Buy if the current flow belongs to the best cluster AND we can afford
            if uid == best_uid and env.budget >= price + 2.0:
                return 2

        # Default: block if anomaly score suggests danger, else accept
        if f['anorm_score'] > env.cfg['anomaly_threshold'] * 0.8:
            return 0    # block anomalous-looking flow
        else:
            return 1    # accept benign-looking flow (this is the trap for cluster A!)

    def push(self, *args):
        pass

    def train_step(self):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Training loop helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeStats:
    total_reward: float = 0.0
    n_steps:      int   = 0
    win:          bool  = False
    n_labels:     int   = 0
    labels:       List[bool] = field(default_factory=lambda: [False]*4)
    budget_final: float = 0.0


def run_episode(agent, env: SyntheticLIONEnv, train: bool = True) -> EpisodeStats:
    state = env.reset()
    stats = EpisodeStats()

    while True:
        if isinstance(agent, BreakEvenAgent):
            action = agent.act(state, env)
        else:
            action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        stats.total_reward += reward

        if train:
            agent.push(state, action, reward, next_state, done)
            agent.train_step()

        state = next_state
        if done:
            stats.n_steps     = env.t
            stats.win         = info['win']
            stats.n_labels    = info['n_labels']
            stats.labels      = list(info['labels'])
            stats.budget_final = info['budget']
            break

    return stats


def smooth(xs: List[float], w: int = 15) -> List[float]:
    out = []
    for i, x in enumerate(xs):
        lo = max(0, i - w)
        out.append(float(np.mean(xs[lo:i+1])))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 10. Main experiment
# ──────────────────────────────────────────────────────────────────────────────

AGENTS = ['DDQN', 'DAI-P', 'Break-Even']

def make_agent(name: str, cfg: dict):
    if name == 'DDQN':
        return DDQNAgent(cfg)
    elif name == 'DAI-P':
        return DAIPAgent(cfg)
    else:
        return BreakEvenAgent()


def run_experiment(n_episodes: int, seeds: List[int], cfg: dict, verbose: bool = True):
    """Run all agents across all seeds.  Returns dict of results per agent."""
    results = {name: {'rewards': [], 'wins': [], 'n_labels': [],
                      'label_A': [], 'budget': []}
               for name in AGENTS}

    for seed in seeds:
        if verbose:
            print(f"\n── Seed {seed} ──────────────────────────────────────────")

        # Build shared data generator and inference module (one per seed)
        dg  = DataGenerator(seed=seed)
        inf = InferenceModule(cfg)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        pretrain_inference(inf, dg, cfg)
        if verbose:
            print(f"  Pretraining done.")

        for agent_name in AGENTS:
            # Fresh agent + fresh env per (seed, agent) run
            torch.manual_seed(seed + 1000 * AGENTS.index(agent_name))
            random.seed(seed + 1000 * AGENTS.index(agent_name))
            np.random.seed(seed + 1000 * AGENTS.index(agent_name))

            agent = make_agent(agent_name, cfg)
            env   = SyntheticLIONEnv(dg, deepcopy(inf), cfg)

            ep_rewards, ep_wins, ep_labels, ep_labelA, ep_budget = [], [], [], [], []

            for ep in range(n_episodes):
                stats = run_episode(agent, env, train=True)
                ep_rewards.append(stats.total_reward)
                ep_wins.append(int(stats.win))
                ep_labels.append(stats.n_labels)
                ep_labelA.append(int(stats.labels[0]))  # Unknown A (index 0)
                ep_budget.append(stats.budget_final)

            results[agent_name]['rewards'].append(ep_rewards)
            results[agent_name]['wins'].append(ep_wins)
            results[agent_name]['n_labels'].append(ep_labels)
            results[agent_name]['label_A'].append(ep_labelA)
            results[agent_name]['budget'].append(ep_budget)

            if verbose:
                late = slice(-50, None)
                print(f"  {agent_name:10s}  "
                      f"mean_rew={np.mean(ep_rewards[late]):.2f}  "
                      f"win_rate={np.mean(ep_wins[late]):.2f}  "
                      f"label_A_rate={np.mean(ep_labelA[late]):.2f}  "
                      f"mean_labels={np.mean(ep_labels[late]):.2f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 11. Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, n_episodes: int, out_path: str = 'synthetic_lion_results.png'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plot.")
        return

    colours  = {'DDQN': '#2196F3', 'DAI-P': '#E91E63', 'Break-Even': '#4CAF50'}
    ls_map   = {'DDQN': '--',       'DAI-P': '-',       'Break-Even': ':'}
    smooth_w = max(1, n_episodes // 20)
    xs       = np.arange(n_episodes)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        'Synthetic LION — DDQN vs. DAI-P vs. Break-Even  '
        f'({len(next(iter(results.values()))["wins"])} seeds)',
        fontsize=13, fontweight='bold'
    )

    metrics = [
        ('rewards',  'Episode cumulative reward',  'Reward',   axes[0, 0]),
        ('wins',     'Win rate (rolling avg)',      'Win rate', axes[0, 1]),
        ('n_labels', 'Labels purchased / episode', '# labels', axes[0, 2]),
        ('label_A',  'Label-A purchased (frac.)',  'Label-A',  axes[1, 0]),
        ('budget',   'Final budget',               'Budget',   axes[1, 1]),
    ]

    for key, title, ylabel, ax in metrics:
        for name in AGENTS:
            arr = np.array(results[name][key], dtype=float)   # (seeds, episodes)
            mu  = arr.mean(0)
            se  = arr.std(0) / max(1, math.sqrt(arr.shape[0]))
            smu = np.array(smooth(list(mu),   smooth_w))
            sse = np.array(smooth(list(se),   smooth_w))
            ax.plot(xs, smu, label=name, color=colours[name],
                    lw=2.2, ls=ls_map[name])
            ax.fill_between(xs, smu - sse, smu + sse,
                            alpha=0.18, color=colours[name])
        # Mark the "early phase" boundary
        ax.axvline(50, color='grey', lw=0.8, ls='--', alpha=0.6)
        ax.text(52, ax.get_ylim()[0], 'ep 50', fontsize=7, color='grey')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Annotated explanation panel
    axes[1, 2].axis('off')
    txt = (
        "Experimental design\n"
        "───────────────────\n"
        "7 Gaussian classes:\n"
        "  K0,K1 benign  K2 malicious (known)\n"
        "  A malicious overlapping K0  ← TRAP\n"
        "  B malicious separated\n"
        "  C benign  separated\n"
        "  D benign  near K2\n\n"
        "Budget 15 < total label cost 19:\n"
        "  must choose ≤ 3 labels wisely.\n\n"
        "Uninformed A-flow accepted → −7\n"
        "After label-A: blocked     → +2.5\n\n"
        "Break-Even: buys highest-anomaly\n"
        "  cluster first (always = B).\n"
        "  A has LOW anomaly score.\n"
        "  → A purchased last or never.\n\n"
        "DAI-P epistemic gain:\n"
        "  buy-label-A causes largest\n"
        "  proprioceptive state flip\n"
        "  (known-conf benign→malicious)\n"
        "  → highest prediction error\n"
        "  → highest intrinsic value.\n"
        "  → faster early convergence.\n"
    )
    axes[1, 2].text(0.04, 0.97, txt, transform=axes[1, 2].transAxes,
                    fontsize=8.5, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"Plot saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 12. Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Synthetic LION experiment')
    parser.add_argument('--episodes', type=int, default=300,
                        help='Episodes per agent per seed (default 300)')
    parser.add_argument('--seeds',    type=int, default=5,
                        help='Number of random seeds (default 5)')
    parser.add_argument('--no-plot',  action='store_true',
                        help='Skip matplotlib output')
    parser.add_argument('--verbose',  action='store_true', default=True)
    args = parser.parse_args()

    print("=" * 65)
    print(" Synthetic LION experiment")
    print(f"  Episodes : {args.episodes}   Seeds : {args.seeds}")
    print("=" * 65)
    print()
    print("Data layout:")
    dg = DataGenerator()
    for k in range(dg.N_KNOWN):
        print(f"  Known  {k} ({dg.KNOWN_TYPES[k]:8s}): μ={dg.KNOWN_MEANS[k]}")
    for u in range(dg.N_UNKNOWN):
        print(f"  Unknown {dg.UNKNOWN_NAMES[u]:20s}: μ={dg.UNKNOWN_MEANS[u]}"
              f"  price={CFG['label_prices'][u]}")
    print()
    print("When is AI expected to win?")
    print("  Unknown-A looks BENIGN to the uninformed inference module")
    print("  → accepted → −7 per flow.  After buying label-A the module")
    print("    flips to 'malicious A' → blocked → +2.5 per flow.")
    print("  Buying label-A produces the LARGEST one-step proprioceptive")
    print("  state change → highest perceptive-epistemic gain in DAI-P.")
    print("  Break-Even de-prioritises A because its anomaly score is LOW.")
    print()

    seeds = list(range(args.seeds))
    results = run_experiment(args.episodes, seeds, CFG, verbose=args.verbose)

    # ── Summary tables ─────────────────────────────────────────────────────
    early = slice(0, 50)
    late  = slice(-50, None)
    header = (f"{'Agent':12s}  {'MeanRew':>9}  {'WinRate':>8}  "
              f"{'LabelA':>8}  {'±σ WinRate':>11}")

    for label, slc in [('EARLY (ep 1-50)', early), ('LATE (ep last 50)', late)]:
        print()
        print("─" * 65)
        print(f"  {label}")
        print("─" * 65)
        print(header)
        print("─" * 65)
        for name in AGENTS:
            rews   = np.array(results[name]['rewards'])[:, slc].mean()
            arr_w  = np.array(results[name]['wins'])[:, slc]
            wins   = arr_w.mean()
            wins_sd = arr_w.mean(axis=1).std()
            labelA = np.array(results[name]['label_A'])[:, slc].mean()
            print(f"{name:12s}  {rews:>9.2f}  {wins:>8.3f}  "
                  f"{labelA:>8.3f}  {wins_sd:>11.3f}")
        print("─" * 65)

    if not args.no_plot:
        plot_results(results, args.episodes)


if __name__ == '__main__':
    main()
