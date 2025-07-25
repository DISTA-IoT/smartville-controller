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
from pox.core import core
from pox.openflow.of_json import flow_stats_to_list
from smartController.flow import Flow, CircularBuffer
import torch
import torch.nn.functional as F
from pox.lib.packet.ipv4 import ipv4


BENIGN_SUFFIX = ' (Benign)'
class FlowLogger(object):
    

    def __init__(
      self,
      multi_class,
      packet_buffer_len,
      packet_feat_dim,
      anonymize_transport_ports,
      flow_feat_dim=4,
      flow_buff_len=10):

      """
      TODO: flows_dict should be one for each switch... or, equivalently, we should use one 
      switch_logger per switch.
      """
      self.flows_dict = {}
      self.packet_cache = {}
      self.logger_instance = core.getLogger()
      self.multi_class = multi_class
      self.packet_buffer_len = packet_buffer_len
      self.packet_feat_dim = packet_feat_dim
      self.anomyn_ports = anonymize_transport_ports
      self.flow_feat_dim = flow_feat_dim
      self.flow_buff_len = flow_buff_len


    def extract_flow_feature_tensor(self, flow):
       return torch.Tensor(
          [flow['byte_count'], 
            flow['duration_nsec'] / 10e9,
            flow['duration_sec'],
            flow['packet_count']]).to(torch.float32)


    def get_anonymized_copy(self, original_packet):
      # Create a new instance of the IPv4 packet
      new_ipv4_packet = ipv4(raw=original_packet.raw)
      new_ipv4_packet.srcip = '0.0.0.0'
      new_ipv4_packet.dstip = '0.0.0.0'
      if self.anomyn_ports:
         new_ipv4_packet.next.srcport = 0  # Set source port to 0
         new_ipv4_packet.next.dstport = 0  # Set destination port to 0
      return new_ipv4_packet


    def build_packet_tensor(self, packet):
        
        packet_copy = self.get_anonymized_copy(packet)
        # Old anonymization techniche: (only IP masking was verified, port masking corresponds to last two byte sequences and need verification)
        # packet_copy.raw = packet.raw[:12] + b'\x00\x00\x00\x00'  + b'\x00\x00\x00\x00' + b'\x00\x00' + b'\x00\x00' + packet.raw[24:]
        
        # These prints show that anonymization is working:
        # print(f" old srcip: {packet.srcip} old dstip: {packet.dstip} old srcport: {packet.next.srcport} old destport: {packet.next.dstport}")
        # print(f" new srcip: {packet_copy.srcip} new dstip: {packet_copy.dstip} new srcport: {packet_copy.next.srcport} new destport: {packet_copy.next.dstport}")

        # Extract the first self.packet_feat_dim bytes of the packet
        packet_data = packet_copy.raw[:self.packet_feat_dim]
        # Convert packet data to a tensor
        payload_data_tensor = torch.tensor([int(x) for x in packet_data], dtype=torch.float32)
        # Pad the array if it's less than self.packet_feat_dim bytes
        if payload_data_tensor.shape[0] < self.packet_feat_dim:
            payload_data_tensor = F.pad(payload_data_tensor, 
                                  (0, self.packet_feat_dim - payload_data_tensor.shape[0]), 
                                  mode='constant', value=0)

        return payload_data_tensor


    def cache_unprocessed_packets(self, src_ip, dst_ip, packet):
        """
        We need to add some packets among the features of flows to augment the perceptive field of our AI. 
        The packets that arrive at the controller, however, are by definition orphans of flow rules. 
        We cache them until the flow rules are available. Whenever flowstats arrive, we will
        query this cache memory to populate flow features with packet data.

        returns a flag indicating if the buffer is full of data.
        """
        partial_flow_id = str(src_ip) + "_" + str(dst_ip)
        packet_tensor = self.build_packet_tensor(packet=packet.next)

        if partial_flow_id in self.packet_cache.keys():
            # A tensor already exists:
            curr_packets_circ_buff = self.packet_cache[partial_flow_id]
            self.logger_instance.debug(f"Updated circular packet buffer for {partial_flow_id}") 
        else:
           # Create new circular buffer:
           curr_packets_circ_buff = CircularBuffer(
              buffer_size=self.packet_buffer_len, 
              feature_size=self.packet_feat_dim)
           self.logger_instance.debug(f"Created circular packet buffer for {partial_flow_id}") 


        curr_packets_circ_buff.add(packet_tensor)
        self.packet_cache[partial_flow_id] = curr_packets_circ_buff

        return curr_packets_circ_buff.is_full


    def process_received_flow(
          self, 
          flow,
          current_knowledge,
          traffic_dict,
          ips_containers):
        
        sender_ip_addr = flow['match']['nw_src'].split('/')[0]

        new_flow = Flow(
          source_ip=sender_ip_addr, 
          dest_ip=flow['match']['nw_dst'].split('/')[0], 
          switch_output_port=flow['actions'][1]['port'],
          flow_feat_dim=self.flow_feat_dim,
          flow_buff_len=self.flow_buff_len)
        
        # This is where our labelling takes place... 
        if sender_ip_addr in ips_containers:
            if ips_containers[sender_ip_addr] != 'pox-controller':
               hostname = ips_containers[sender_ip_addr]
               flow_info = traffic_dict[hostname]
                     
               new_flow.element_class = flow_info['pattern']
               # legacy labelling:
               # if flow_info['benign']: new_flow.element_class += BENIGN_SUFFIX

               new_flow.test_zda = flow_info['pattern'] in current_knowledge['G2s']
               new_flow.zda = new_flow.test_zda or flow_info['pattern'] in current_knowledge['G1s']
               
               flow_features = self.extract_flow_feature_tensor(flow=flow)

               if new_flow.flow_id in self.flows_dict.keys():
                  self.flows_dict[new_flow.flow_id].enrich_flow_features(flow_features)
               else:
                  new_flow.enrich_flow_features(flow_features)
                  self.flows_dict[new_flow.flow_id] = new_flow

               self.update_packet_buffer(new_flow)


    def update_packet_buffer(self, flow_object):
       """
       Attack the cached packets tensor to the flow entry in flows_dict
       """

       partial_flow_id = "_".join(flow_object.flow_id.split("_")[:-1])

       if partial_flow_id in self.packet_cache.keys():
          
          packets_buffer = self.packet_cache[partial_flow_id]
          del self.packet_cache[partial_flow_id]

          if self.flows_dict[flow_object.flow_id].packets_tensor == None:
             self.flows_dict[flow_object.flow_id].packets_tensor = packets_buffer
          else: 
             for single_packet_tensor in packets_buffer.buffer:
                self.flows_dict[flow_object.flow_id].packets_tensor.add(single_packet_tensor)

    
    def _handle_flowstats_received (self, event, current_knowledge, traffic_dict, ips_containers):
      self.logger_instance.debug("FlowStatsReceived")
      stats = flow_stats_to_list(event.stats)
      self.logger_instance.debug(f"Received {len(stats)} flow stats")
      for sender_flow in stats:
        self.process_received_flow(
           flow=sender_flow,
           current_knowledge=current_knowledge,
           traffic_dict=traffic_dict,
           ips_containers=ips_containers)

    def reset_all_flows_metadata(self):
       self.flows_dict = {}


    def reset_single_flow_metadata(self, flow_id):
       del self.flows_dict[flow_id]
