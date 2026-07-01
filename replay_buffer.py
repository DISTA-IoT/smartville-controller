# This file is part of the "Smartville" project.
# Copyright (c) 2024 University of Insubria
# Licensed under the Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# For the full text of the license, visit:
# https://www.apache.org/licenses/LICENSE-2.0

# Smartville is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License 2.0 for more details.

# You should have received a copy of the Apache License 2.0
# along with Smartville. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

# Additional licensing information for third-party dependencies
# used in this file can be found in the accompanying `NOTICE` file.
import torch
import random
import warnings
import numpy as np
from collections import deque


class SumTree:
    """
    A binary tree data structure where the parent node is the sum of its children.
    Iterative implementation for better performance on CPU.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        while True:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def _retrieve(self, idx, s):
        while True:
            left = 2 * idx + 1
            right = left + 1

            if left >= len(self.tree):
                return idx

            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, seed=42):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.max_priority = 1.0
        random.seed(seed)
        np.random.seed(seed)

    def _get_priority(self, error):
        # Guard against a non-finite TD error (upstream NaN/inf, e.g. a diverged
        # encoder producing NaN states). A single inf/NaN priority would set
        # max_priority=inf, poison every subsequent push, drive tree.total() to
        # inf, and make sample()'s np.random.uniform raise "Range exceeds valid
        # bounds" -- a run-killing crash far from the real cause. Treat it as a
        # zero-error (lowest) priority and warn once so the divergence is
        # visible rather than silently swallowed.
        if not np.isfinite(error):
            if not getattr(self, '_warned_nonfinite_priority', False):
                warnings.warn(
                    "PrioritizedReplayBuffer: non-finite TD error encountered; "
                    "clamping its priority to the minimum. This indicates an "
                    "upstream NaN/inf (e.g. a diverged model producing NaN "
                    "states/Q-values) -- investigate the source.")
                self._warned_nonfinite_priority = True
            error = 0.0
        return (np.abs(error) + self.epsilon) ** self.alpha

    def push(self, sample):
        self.tree.add(self.max_priority, sample)

    def sample(self, n):
        batch = []
        idxs = []
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment])

        # Optimized vectorized segment sampling
        total_p = self.tree.total()
        segment = total_p / n
        s_vals = np.random.uniform(segment * np.arange(n), segment * np.arange(1, n + 1))

        for s in s_vals:
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / (total_p + 1e-10)
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= (is_weights.max() + 1e-10)

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)

        return (torch.stack(states),
                torch.tensor(actions),
                torch.tensor(rewards, dtype=torch.float32),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.bool),
                idxs,
                torch.tensor(is_weights, dtype=torch.float32))

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries


class Batch():
    """
    A comodity class to pass the elements of every batch back and forth...
    """
    def __init__(
            self,
            flow_features=None,
            packet_features=None,
            node_features=None,
            class_labels=None,
            zda_labels=None,
            test_zda_labels=None
            ):
        
        self.flow_features = flow_features
        self.packet_features = packet_features
        self.node_features = node_features
        self.class_labels = class_labels
        self.zda_labels = zda_labels
        self.test_zda_labels = test_zda_labels


class RawReplayBuffer():
    """
    This buffer is not using binary labels for zdas and test zdas,
    instead, it will ask the zda labellings to the dynamic curriculum in the caller.
    """
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.position = 0
        self.size = 0
        random.seed(seed)


    def push(self, flow_state, packet_state, node_state, label):
        self.buffer[self.position] = (flow_state, packet_state, node_state, label)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, num_of_samples):
        if self.size < num_of_samples:
            raise RuntimeError(f"Error during sampling replay buffer. Not enough samples: {self.size} < {num_of_samples}")

        indices = random.sample(range(self.size), num_of_samples)

        f_batch, p_batch, n_batch, l_batch = [], [], [], []

        for i in indices:
            f, p, n, l = self.buffer[i]
            f_batch.append(f)
            if p is not None: p_batch.append(p)
            if n is not None: n_batch.append(n)
            l_batch.append(l)
        
        return torch.cat(f_batch, 0), \
            (torch.cat(p_batch, 0) if p_batch else None), \
            (torch.cat(n_batch, 0) if n_batch else None), \
            torch.cat(l_batch, 0).unsqueeze(1)


    def __len__(self):
        return self.size