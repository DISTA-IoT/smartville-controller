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
        replay_buffer_max_capacity,
        max_pending_packet_feats=None):

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
        self.packet_count = 0

        # Packets sampled from the switch during a sampling burst arrive faster
        # than flow-stats refresh (every flowstats_freq_secs). Rather than
        # overwriting them into a single "last packet" slot and discarding the
        # rest, we queue them here so every captured packet can be turned into
        # its own sample (paired with the same, still-valid, flow_feat window).
        self.pending_packet_feats = []
        # Bounds how large pending_packet_feats can grow (e.g. a flow whose
        # consumer falls behind during a dense/fast-replayed burst). None
        # means unbounded. Once full, newly captured packets are dropped
        # (see queue_packet_feature) rather than queued indefinitely.
        self.max_pending_packet_feats = max_pending_packet_feats
        # High-water mark of len(pending_packet_feats) since the last time it
        # was read (see snapshot_pending_stats). Purely for monitoring: the
        # consumer (TigerBrain.process_input) drains this queue in a tight
        # loop, so its instantaneous length is almost always ~0 even while
        # packets are being captured in dense bursts. The peak preserves the
        # burst so a low-frequency observer (e.g. the dashboard polling every
        # few seconds) can still see it instead of only ever sampling zeros.
        self.pending_high_water_mark = 0


    def get_flow_features(self):
        return self.flow_feat_circular_buffer.buffer[-self.flows_per_sample:]

    def get_packet_features(self):
        return self.packet_feat_circular_buffer.buffer[-self.packets_per_sample:]

    def queue_packet_feature(self, packet_tensor):
        """
        Called once per packet captured during a sampling burst.
        Returns True if the packet was queued, False if it was dropped
        because pending_packet_feats is already at max_pending_packet_feats.
        """
        if self.max_pending_packet_feats is not None and \
           len(self.pending_packet_feats) >= self.max_pending_packet_feats:
            return False
        self.pending_packet_feats.append(packet_tensor)
        if len(self.pending_packet_feats) > self.pending_high_water_mark:
            self.pending_high_water_mark = len(self.pending_packet_feats)
        return True

    def snapshot_pending_stats(self, reset_peak=True):
        """
        Returns a small, JSON-friendly snapshot of this flow's pending-packet
        queue for monitoring (e.g. the dashboard utilisation gauges). Reads
        only, with no effect on the ML data path.

        `pending` is the instantaneous queue depth right now; `peak_pending`
        is the high-water mark since the previous snapshot -- the useful one,
        since bursts are usually drained between two low-frequency reads. It is
        max(deepest the queue got via arrivals since the last read, current
        depth), so it reports each capture burst exactly once and still
        reflects a standing/growing backlog. When `reset_peak` is True the mark
        is rearmed to 0 afterwards, so a burst that fully drains before the
        next read is not double-counted into that read's window.

        `capacity` is max_pending_packet_feats (may be None = unbounded).
        `utilization`/`peak_utilization` are fractions in [0, 1] of that
        capacity (None when capacity is None); captures past the cap are
        dropped rather than queued, so neither exceeds 1.0.
        """
        pending = len(self.pending_packet_feats)
        peak = max(self.pending_high_water_mark, pending)
        if reset_peak:
            self.pending_high_water_mark = 0
        capacity = self.max_pending_packet_feats
        util = (pending / capacity) if capacity else None
        peak_util = (peak / capacity) if capacity else None
        return {
            "flow_id": self.flow_id,
            "element_class": self.element_class,
            "pending": pending,
            "peak_pending": peak,
            "capacity": capacity,
            "utilization": util,
            "peak_utilization": peak_util,
            "packet_count": self.packet_count,
        }

    def drain_packet_feature_chunks(self, chunk_size, max_chunks=None):
        """
        Pops up to max_chunks complete, in-arrival-order chunks of
        `chunk_size` packets (all available chunks if max_chunks is None),
        leaving any remainder -- incomplete trailing chunk and/or chunks
        beyond the cap -- queued for the next call. This bounds how many
        rows a single flow can inject into one tick's batch (a flow with a
        deep backlog, e.g. mid-burst, drains it gradually over several ticks
        instead of all at once), while still guaranteeing every captured
        packet eventually becomes a sample.
        Returns a (possibly empty) list of [chunk_size, packet_feat_dim] tensors.
        """
        n_chunks = len(self.pending_packet_feats) // chunk_size
        if max_chunks is not None:
            n_chunks = min(n_chunks, max_chunks)
        chunks = []
        for _ in range(n_chunks):
            chunk = self.pending_packet_feats[:chunk_size]
            self.pending_packet_feats = self.pending_packet_feats[chunk_size:]
            chunks.append(torch.stack(chunk))
        return chunks