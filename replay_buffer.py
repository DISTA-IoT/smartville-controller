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
from collections import deque


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


class ReplayBuffer():
    
    def __init__(self, capacity, batch_size, seed):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = [None] * capacity
        self.position = 0
        self.size = 0
        random.seed(seed)


    def push(self, flow_state, packet_state, node_state, label, zda_label, test_zda_label):
        self.buffer[self.position] = (flow_state, packet_state, node_state, label, zda_label, test_zda_label)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, num_of_samples):
        if self.size < num_of_samples:
            raise RuntimeError(f"Not enough samples in buffer: {self.size} < {num_of_samples}")

        indices = random.sample(range(self.size), num_of_samples)
        
        f_batch, p_batch, n_batch, l_batch, zl_batch, tzl_batch = [], [], [], [], [], []

        for i in indices:
            f, p, n, l, zl, tzl = self.buffer[i]
            f_batch.append(f)
            if p is not None: p_batch.append(p)
            if n is not None: n_batch.append(n)
            l_batch.append(l)
            zl_batch.append(zl)
            tzl_batch.append(tzl)

        return torch.cat(f_batch, 0), \
               (torch.cat(p_batch, 0) if p_batch else None), \
               (torch.cat(n_batch, 0) if n_batch else None), \
               torch.cat(l_batch, 0), \
               torch.cat(zl_batch, 0), \
               torch.cat(tzl_batch, 0)

    def __len__(self):
        return self.size
    


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
            torch.cat(l_batch, 0)


    def __len__(self):
        return self.size