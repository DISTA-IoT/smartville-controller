"""SimbaBrain — the whole controller intelligence in one small class.

Per tick (== one batch of flowstats):
  1. every sample lands in its class's shadow buffer (ground-truth labels
     come from the recorder trace offline, or from traffic_dict in GNS3);
  2. the IM embeds the batch and splits it into groups: one group per
     predicted Known class + one cluster per connected component of
     unknown-looking samples;
  3. for each group the DM sees a state (group centroid + a small
     proprioceptive block) and picks accept / block / buy-CTI; the
     environment converts the outcome into a budget delta;
  4. the IM takes its online gradient step(s), the DM replays.

Transitions are chained decision-to-decision (the next_state of a
decision is the state of the following decision, across tick boundaries),
which keeps the MDP honest without TIGER's cloned-state tricks.

The same `step_tick` serves both modalities; `process_input(flows)` is
the POX-facing adapter with the same signature contract as
TigerBrain.process_input.
"""

import os
from collections import Counter
from typing import Dict, List, Optional

import torch

from .agent import DQNAgent
from .config import SimbaConfig, _as_flat_dict
from .environment import SimbaEnvironment
from .inference import SimbaInference

PROPRIO_DIM = 8   # + one known-flag per curriculum class (see _state)
ACCEPT, BLOCK, BUY = 0, 1, 2
ABLATIONS = ('drl', 'no_epistemic', 'greedy_cti')


class SimbaBrain:

    def __init__(self, cfg: SimbaConfig, logger=None):
        assert cfg.ablation in ABLATIONS, f'unknown ablation {cfg.ablation}'
        self.cfg = cfg
        self.logger = logger
        torch.manual_seed(cfg.seed)
        self.im = SimbaInference(cfg, logger)
        self.im.trainable = set(cfg.knowns)
        self.env = SimbaEnvironment(cfg, logger)
        # The DM sees the group's centroid in the *stationary* input space
        # (not the drifting learned embedding), plus a proprioceptive block
        # that ends with one known-flag PER curriculum class (canonical
        # order). The per-class flags are what make knowledge *selectively*
        # valuable: with only an aggregate knowns-count, V(s) cannot tell a
        # world where doorlock is Known from one where mirai is — the
        # continuation value of every buy collapses to the average value of
        # knowledge (mostly worthless), and the DQN rationally stops buying.
        self._class_order = sorted(cfg.all_classes())
        self.agent = DQNAgent(cfg, state_dim=self.im.input_rep_dim()
                              + PROPRIO_DIM + len(self._class_order))
        self.training = True          # False => greedy policy, no DM updates
        self.episode_count = 0
        self._total_ticks = cfg.max_episode_ticks or 1_000_000
        self._pending = None          # (state, action, scaled_reward)
        # POX-side attributes (tiger_server wires these into the
        # flowstats listener); set by from_tiger_config, unused offline.
        self.traffic_dict = None
        self.ips_containers = None
        if getattr(cfg, 'im_snapshot_path', '') and os.path.exists(cfg.im_snapshot_path):
            self.im.load(cfg.im_snapshot_path)
            self._log(f'IM weights loaded from {cfg.im_snapshot_path}')

    def _log(self, msg):
        if self.logger:
            self.logger.info(f'SIMBA: {msg}')

    # ---------------------------------------------------------------- setup
    @classmethod
    def from_tiger_config(cls, args: dict) -> 'SimbaBrain':
        """Build from the JSON config payload the dashboard POSTs to the
        controller's /initialize endpoint. Optional overrides live under
        an `simba:` block; everything else is mapped from the usual keys."""
        det = args.get('intrusion_detection', {})
        know = args.get('knowledge', {})
        d = {
            'device': args.get('device', 'cpu'),
            'use_packet_feats': bool(args.get('use_packet_feats', True)),
            'flow_feat_dim': int(det.get('flow_feat_dim', 4)),
            'packet_feat_dim': int(det.get('packet_feat_dim', 64)),
            'flows_per_sample': int(det.get('flows_per_sample', 10)),
            'packets_per_sample': int(det.get('packets_per_sample', 1)),
            'seed': int(det.get('seed', 777)),
            'knowns': list(know.get('Knowns', [])),
            'g1s': list(know.get('G1s', [])),
            'g2s': list(know.get('G2s', [])),
            'benign_classes': list(know.get('benign_patterns',
                                            know.get('bening_patterns', []))),
            'malicious_classes': list(know.get('attack_patterns', [])),
            'rewards': _as_flat_dict(args.get('rewards')),
            'prices': _as_flat_dict(args.get('prices')),
        }
        d.update(args.get('simba') or {})
        cfg = SimbaConfig.from_dict(d)
        if not cfg.max_episode_ticks:
            cfg.max_episode_ticks = int(det.get('max_episode_steps', 1000))
        brain = cls(cfg, logger=args.get('logger'))
        brain.traffic_dict = args.get('traffic_dict')
        brain.ips_containers = args.get('ips_containers')
        return brain

    def pretrain(self, trace, steps: Optional[int] = None):
        """Offline pretraining: seed the shadow buffers from the trace's
        per-class pools, train the encoder on the Knowns, calibrate,
        snapshot. Every episode restarts from this snapshot."""
        import numpy as np
        rng = np.random.default_rng(self.cfg.seed)
        for label in trace.class_indices:
            flow, pkt = trace.sample_class(label, self.cfg.buffer_capacity, rng)
            if flow is not None:
                self.im.push(flow, pkt, [label] * flow.shape[0])
        steps = steps or self.cfg.pretrain_steps
        for g in self.im.optimizer.param_groups:
            g['lr'] = self.cfg.pretrain_learning_rate
        losses = []
        for i in range(steps):
            loss = self.im.train_step()
            if loss is not None:
                losses.append(loss)
        for g in self.im.optimizer.param_groups:
            g['lr'] = self.cfg.im_learning_rate
        self.im.calibrate()
        self.im.snapshot()
        last = sum(losses[-20:]) / max(1, len(losses[-20:]))
        self._log(f'pretrained IM for {steps} steps '
                  f'(last-20 loss {last:.4f}, tau {self.im.tau:.3f})')

    def begin_episode(self, total_ticks: Optional[int] = None):
        """Reset environment + IM to the pretrained snapshot. The DM (its
        weights, replay memory and epsilon) persists across episodes."""
        self._flush_pending(done=True)
        self.env.reset()
        self.im.restore()
        self._total_ticks = total_ticks or self.cfg.max_episode_ticks or 1_000_000
        self.episode_count += 1

    # ------------------------------------------------------------ the tick
    def step_tick(self, flow: torch.Tensor, pkt: Optional[torch.Tensor],
                  labels: List[str]) -> Dict:
        """Process one batch of flow samples. Returns a metrics dict."""
        cfg = self.cfg
        info = {'n_samples': len(labels), 'n_decisions': 0, 'n_buys': 0,
                'tick_reward': 0.0, 'im_loss': None, 'dm_loss': None}
        if len(labels) == 0:
            self.env.ticks += 1
            return info

        self.im.push(flow, pkt, labels)
        z, preds, min_dist, unknown = self.im.infer(flow, pkt)
        min_dist, unknown = min_dist.cpu(), unknown.cpu()
        x_in = self.im.input_rep(flow, pkt)   # stationary DM representation

        # ---- inference metrics (vs ground truth) --------------------------
        truly_unknown = torch.tensor(
            [lab not in self.env.knowns for lab in labels])
        info['ad_tp'] = int((unknown & truly_unknown).sum())
        info['ad_fp'] = int((unknown & ~truly_unknown).sum())
        info['ad_fn'] = int((~unknown & truly_unknown).sum())
        info['ad_tn'] = int((~unknown & ~truly_unknown).sum())
        known_idx = (~unknown).nonzero().flatten().tolist()
        if known_idx:
            hits = sum(1 for i in known_idx if preds[i] == labels[i])
            info['cs_acc'] = hits / len(known_idx)

        # ---- build decision groups ----------------------------------------
        groups = []   # (member indices, is_unknown)
        by_class: Dict[str, List[int]] = {}
        for i in known_idx:
            by_class.setdefault(preds[i], []).append(i)
        for name in sorted(by_class):
            groups.append((by_class[name], False))
        unk_idx = unknown.nonzero().flatten().tolist()
        if unk_idx:
            radius = self.im.cluster_radius
            if radius is None:
                cl = torch.zeros(len(unk_idx), dtype=torch.long)
            else:
                cl = self.im.cluster(x_in[unk_idx], radius)
            for cid in range(int(cl.max().item()) + 1):
                members = [unk_idx[j] for j in
                           (cl == cid).nonzero().flatten().tolist()]
                groups.append((members, True))

        # ---- decide per group ---------------------------------------------
        for members, is_unknown in groups:
            state = self._state(x_in[members], is_unknown,
                                min_dist[members], members, labels)
            majority = Counter(labels[i] for i in members).most_common(1)[0][0]
            quote = self.env.quote(majority)
            action = self._decide(state, is_unknown, quote)

            outcome = self.env.group_outcome(
                action, [labels[i] for i in members])
            reward = outcome['total']
            if action == BUY:
                charged = self.env.buy(majority)
                reward -= charged
                info['n_buys'] += 1
                if majority in self.env.knowns:   # purchase succeeded
                    self.im.trainable.add(majority)
                    for _ in range(cfg.buy_train_burst):
                        self.im.train_step()
                    self.im.calibrate()
            self.env.apply(reward)
            info['tick_reward'] += reward
            info['n_decisions'] += 1

            terminal = cfg.bankrupt_terminates and self.env.bankrupt
            if self.training:
                scaled = reward * cfg.reward_scale
                if self._pending is not None:
                    s0, a0, r0 = self._pending
                    self.agent.remember(s0, a0, r0, state, done=False)
                self._pending = (state, action, scaled)
                if terminal:
                    self._flush_pending(done=True)
                dm_loss = self.agent.replay()
                if dm_loss is not None:
                    info['dm_loss'] = dm_loss
            if terminal:
                break

        # ---- online IM training -------------------------------------------
        for _ in range(cfg.im_train_steps_per_tick):
            im_loss = self.im.train_step()
            if im_loss is not None:
                info['im_loss'] = im_loss
        self.env.ticks += 1
        if self.env.ticks % cfg.calibrate_every_ticks == 0:
            self.im.calibrate()

        info['budget'] = self.env.budget
        info['epsilon'] = self.agent.epsilon
        return info

    # ------------------------------------------------------------ internals
    def _state(self, z_members, is_unknown, dists, members, labels):
        cfg = self.cfg
        centroid = z_members.mean(dim=0)
        majority = Counter(labels[i] for i in members).most_common(1)[0][0]
        quote = self.env.quote(majority)
        tau = self.im.tau if self.im.tau < float('inf') else 1.0
        dist_ratio = float(torch.clamp(dists.mean() / tau, 0, 3) / 3) \
            if len(members) else 0.0
        proprio = torch.tensor([
            min(len(members) / cfg.group_size_scale, 2.0),
            1.0 if is_unknown else 0.0,
            dist_ratio,
            (quote or 0.0) / cfg.price_scale,
            1.0 if quote is not None else 0.0,
            self.env.budget / cfg.budget_scale,
            len(self.env.knowns) / max(1, len(cfg.all_classes())),
            min(self.env.ticks / self._total_ticks, 1.0),
        ], dtype=centroid.dtype, device=centroid.device)
        known_flags = torch.tensor(
            [1.0 if c in self.env.knowns else 0.0 for c in self._class_order],
            dtype=centroid.dtype, device=centroid.device)
        return torch.cat([centroid, proprio, known_flags]).detach()

    def _decide(self, state, is_unknown, quote) -> int:
        """The epistemic action only exists for unknown clusters (as in
        TIGER): known-traffic groups are accept/block for every mode."""
        mode = self.cfg.ablation
        explore = self.training
        pragmatic = [ACCEPT, BLOCK]
        if not is_unknown or mode == 'no_epistemic':
            return self.agent.act(state, allowed=pragmatic, explore=explore)
        if mode == 'greedy_cti':
            affordable = (quote is not None and
                          self.env.budget - quote >= self.cfg.min_budget)
            if affordable:
                return BUY   # buy ASAP, guarding bankrupt buys
            return self.agent.act(state, allowed=pragmatic, explore=explore)
        return self.agent.act(state, explore=explore)

    def _flush_pending(self, done: bool):
        if self._pending is None:
            return
        s0, a0, r0 = self._pending
        self.agent.remember(s0, a0, r0, torch.zeros_like(s0), done=done)
        self._pending = None

    # ----------------------------------------------------- POX / GNS3 side
    def process_input(self, flows, node_feats=None):
        """Adapter for the controller's smart_check loop: same contract as
        TigerBrain.process_input. `node_feats` is accepted and ignored —
        SIMBA is traffic-only by design."""
        if not flows:
            return
        cfg = self.cfg
        flow_feats = torch.stack([f.get_flow_features() for f in flows])
        pkt_feats = None
        if cfg.use_packet_feats:
            pkt_feats = torch.stack([f.get_packet_features() for f in flows])
        labels = [f.element_class for f in flows]
        self.step_tick(flow_feats, pkt_feats, labels)

        if self.env.episode_ended(self._total_ticks):
            summary = self.env.summary()
            self._log(f'episode {self.episode_count} ended: {summary}')
            self.begin_episode()

    def get_profiling_stats_dict(self):
        return {}

    def shutdown(self):
        pass
