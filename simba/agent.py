"""SIMBA Decision Module: one plain DQN. Nothing else.

Vanilla ingredients only, on purpose:
  * MLP Q-network, 3 actions (accept / block / buy-CTI);
  * epsilon-greedy exploration with linear decay;
  * uniform ring replay buffer;
  * 1-step TD target from a hard-updated target network;
  * Huber loss + Adam + gradient-norm clip.

No PER, no n-step, no dueling, no double-Q, no soft updates, no
Boltzmann sampling. Ablations do not change the learner — they only
restrict/force the epistemic action at act time (see SimbaBrain).
"""

import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class DQNAgent:

    N_ACTIONS = 3

    def __init__(self, cfg, state_dim: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = QNet(state_dim, cfg.dm_hidden).to(self.device)
        self.target = QNet(state_dim, cfg.dm_hidden).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=cfg.dm_learning_rate)
        self.memory: List = [None] * cfg.replay_capacity
        self.mem_pos = 0
        self.mem_len = 0
        self.act_steps = 0
        self.train_steps = 0
        self.last_loss = 0.0
        self._rng = random.Random(cfg.seed + 1)

    # ------------------------------------------------------------ policy
    @property
    def epsilon(self) -> float:
        cfg = self.cfg
        frac = min(1.0, self.act_steps / max(1, cfg.eps_decay_steps))
        return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    def act(self, state: torch.Tensor,
            allowed: Optional[List[int]] = None,
            explore: bool = True) -> int:
        """Pick an action. `allowed` restricts the choice (ablations);
        `explore=False` gives the greedy policy (evaluation)."""
        allowed = allowed or list(range(self.N_ACTIONS))
        if explore:
            self.act_steps += 1
            if self._rng.random() < self.epsilon:
                w = self.cfg.explore_buy_weight
                weights = [w if a == 2 else (1.0 - w) / max(1, len(allowed) - (2 in allowed))
                           for a in allowed]
                return self._rng.choices(allowed, weights=weights, k=1)[0]
        with torch.no_grad():
            q = self.model(state.to(self.device).unsqueeze(0)).squeeze(0)
        mask = torch.full_like(q, float('-inf'))
        mask[allowed] = 0.0
        return int((q + mask).argmax().item())

    # ------------------------------------------------------------ memory
    def remember(self, state, action, reward, next_state, done):
        self.memory[self.mem_pos] = (
            state.detach().cpu(), int(action), float(reward),
            next_state.detach().cpu(), float(done))
        self.mem_pos = (self.mem_pos + 1) % self.cfg.replay_capacity
        self.mem_len = min(self.mem_len + 1, self.cfg.replay_capacity)

    # ------------------------------------------------------------ learning
    def replay(self) -> Optional[float]:
        cfg = self.cfg
        if self.mem_len < max(cfg.replay_batch_size, cfg.learn_start):
            return None
        batch = [self.memory[self._rng.randrange(self.mem_len)]
                 for _ in range(cfg.replay_batch_size)]
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            target = rewards + cfg.gamma * (1.0 - dones) * next_q
        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % cfg.target_update_freq == 0:
            self.target.load_state_dict(self.model.state_dict())
        self.last_loss = loss.item()
        return self.last_loss

    # ------------------------------------------------------------ persist
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.target.load_state_dict(state)
