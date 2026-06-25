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

BENIGN = "Benign"

class CircularBuffer:
    def __init__(self, buffer_size=10, feature_size=4):
        self.buffer_size = buffer_size
        self.feature_size = feature_size
        self.buffer = torch.zeros(buffer_size, feature_size)
        self.is_full = False
        self.calls_to_add = 0
        self.curr_elements = 0


    def add(self, new_tensor):
        # Roll the buffer up by 1 along the first dimension
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
        # Add new tensor to the last row
        self.buffer[-1] = new_tensor
        self.calls_to_add += 1
        self.curr_elements += 1
        if self.calls_to_add >= self.buffer_size:
            self.is_full = True
            self.curr_elements = self.buffer_size

class Flow():
    def __init__(
        self, 
        source_ip, 
        dest_ip, 
        switch_output_port,
        flow_feat_dim,
        flows_per_sample,
        packet_feat_dim,
        packets_per_sample,
        replay_buffer_max_capacity):
        
        self.source_ip = source_ip
        self.dest_ip = dest_ip
        self.switch_output_port = switch_output_port
        self.flow_id = self.source_ip + "_" + self.dest_ip + "_" + str(self.switch_output_port)
        self.switch_input_port = None
        self.replay_buffer_max_capacity = replay_buffer_max_capacity
        self.flows_per_sample = flows_per_sample
        self.flow_feat_circular_buffer = CircularBuffer(
                            buffer_size=replay_buffer_max_capacity, 
                            feature_size=flow_feat_dim)
        self.packets_per_sample = packets_per_sample
        self.packet_feat_circular_buffer = CircularBuffer(
                            buffer_size=replay_buffer_max_capacity, 
                            feature_size=packet_feat_dim)
        self.node_feats = None
        self.element_class = BENIGN
        self.zda = False
        self.test_zda = False

        # Packets sampled from the switch during a sampling burst arrive faster
        # than flow-stats refresh (every flowstats_freq_secs). Rather than
        # overwriting them into a single "last packet" slot and discarding the
        # rest, we queue them here so every captured packet can be turned into
        # its own sample (paired with the same, still-valid, flow_feat window).
        self.pending_packet_feats = []


    def get_flow_features(self):
        return self.flow_feat_circular_buffer.buffer[-self.flows_per_sample:]

    def get_packet_features(self):
        return self.packet_feat_circular_buffer.buffer[-self.packets_per_sample:]

    def queue_packet_feature(self, packet_tensor):
        """Called once per packet captured during a sampling burst."""
        self.pending_packet_feats.append(packet_tensor)

    def drain_packet_feature_chunks(self, chunk_size):
        """
        Pops as many complete, in-arrival-order chunks of `chunk_size` packets
        as are currently queued, leaving any incomplete trailing chunk queued
        for the next call (so no captured packet is ever silently dropped).
        Returns a (possibly empty) list of [chunk_size, packet_feat_dim] tensors.
        """
        n_chunks = len(self.pending_packet_feats) // chunk_size
        chunks = []
        for _ in range(n_chunks):
            chunk = self.pending_packet_feats[:chunk_size]
            self.pending_packet_feats = self.pending_packet_feats[chunk_size:]
            chunks.append(torch.stack(chunk))
        return chunks