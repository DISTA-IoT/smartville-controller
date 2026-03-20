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
from smartController.flow import Flow
from smartController.attr_dict import AttrDict
import torch
import torch.nn.functional as F
from pox.lib.packet.ipv4 import ipv4

class FlowLogger(object):
    

    def __init__(
      self,
      wb_tracker=None,
      **kwargs,
      ):

      """
      TODO: flows_dict should be one for each switch... or, equivalently, we should use one 
      switch_logger per switch.
      """
      args = AttrDict(kwargs)
      self.wb_tracker = wb_tracker
      self.flows_dict = {}
      self.packet_buffers = {}
      self.logger_instance = core.getLogger()
      self.logger_instance.name = "FlowLogger"
      self.logger_instance.setLevel(kwargs.get("flow_logger_log_level").upper())
      self.replay_buffer_max_capacity = int(args.intrusion_detection.replay_buffer_max_capacity)
      self.packets_per_sample = int(args.intrusion_detection.packets_per_sample)
      self.packet_feat_dim = int(args.intrusion_detection.packet_feat_dim)
      self.anomyn_ports = args.intrusion_detection.anonymize_transport_ports
      self.flow_feat_dim = int(args.intrusion_detection.flow_feat_dim)
      self.flows_per_sample = int(args.intrusion_detection.flows_per_sample)
      self.use_packet_feats = args.intrusion_detection.use_packet_feats


    def reset(self):
       self.flows_dict = {}
       self.packet_buffers = {}


    def extract_flow_feature_tensor(self, flow, sender_ip_addr):
       
      if self.wb_tracker is not None:
       self.wb_tracker.wb_run.log(
          {
            f'flows/{sender_ip_addr}/byte_count': flow['byte_count'],
            f'flows/{sender_ip_addr}/duration_nsec': flow['duration_nsec'],
            f'flows/{sender_ip_addr}/duration_sec': flow['duration_sec'],
            f'flows/{sender_ip_addr}/packet_count': flow['packet_count']
          },
          step=self.wb_tracker.step_counter
       )

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


        for key in self.flows_dict.keys():
            
            if partial_flow_id in key:
               # We have at this moment a flow object that is interested in this packet.
         
               # Extract packet tensor
               packet_tensor = self.build_packet_tensor(packet=packet.next)

               # Add packet to the buffer
               self.flows_dict[key].packet_feat_circular_buffer.add(packet_tensor)

               self.logger_instance.debug(f"Updated packet buffer for {partial_flow_id}")



    def process_received_flow(
          self, 
          of_flowstats_obj,
          current_knowledge,
          traffic_dict,
          ips_containers):
        
      sender_ip_addr = of_flowstats_obj['match']['nw_src'].split('/')[0]
      dest_ip_addr = of_flowstats_obj['match']['nw_dst'].split('/')[0]

      if sender_ip_addr not in ips_containers:
         self.logger_instance.error(f"IP address {sender_ip_addr} not found in ips_containers!")
         return
      
      if ips_containers[sender_ip_addr] == 'pox-controller':
         return
      
      hostname = ips_containers[sender_ip_addr]
      if hostname not in traffic_dict.keys():
         self.logger_instance.error(f"Traffic from unknown host: {hostname}. Not labelling this flow.")
         return
      
      if traffic_dict[hostname]['dest_ip'] == dest_ip_addr and traffic_dict[hostname]['src_ip'] == sender_ip_addr:

         # This is a flow that we are interested in
         flow_id = sender_ip_addr + "_" + dest_ip_addr + "_" + str(of_flowstats_obj['actions'][1]['port'])
      
         if flow_id in self.flows_dict.keys():
            # Flow already exists
            flow = self.flows_dict[flow_id]
         else:
            # Create new flow object
            self.logger_instance.info(f"Creating new flow object: {flow_id}")
            flow = Flow(
               source_ip=sender_ip_addr, 
               dest_ip=dest_ip_addr, 
               switch_output_port=of_flowstats_obj['actions'][1]['port'],
               flow_feat_dim=self.flow_feat_dim,
               flows_per_sample=self.flows_per_sample,
               packet_feat_dim=self.packet_feat_dim,
               packets_per_sample=self.packets_per_sample,
               replay_buffer_max_capacity=self.replay_buffer_max_capacity)
            self.flows_dict[flow.flow_id] = flow


         # This is where our labelling takes place...
         self.logger_instance.debug(f"Updating labels for flow: {flow.flow_id}")
                  
         flow_info = traffic_dict[hostname]
         flow.element_class = flow_info['pattern']  # this is not changing over time in this version of Smartville
         flow.test_zda = flow_info['pattern'] in current_knowledge['G2s'] # these change
         flow.zda = flow.test_zda or flow_info['pattern'] in current_knowledge['G1s'] # these change
         
         # flow feature extraction ( packet feature circular buffer is updated asychonously...)
         curr_flow_stats_vec = self.extract_flow_feature_tensor(flow=of_flowstats_obj, sender_ip_addr=sender_ip_addr)
         # update the flow feature circular buffer
         flow.flow_feat_circular_buffer.add(curr_flow_stats_vec)

    
    def _handle_flowstats_received(self, event, current_knowledge, traffic_dict, ips_containers):
      self.logger_instance.debug("FlowStatsReceived")
      stats = flow_stats_to_list(event.stats)
      self.logger_instance.debug(f"Received {len(stats)} flow stats")
      for sender_flow in stats:
        self.process_received_flow(
           of_flowstats_obj=sender_flow,
           current_knowledge=current_knowledge,
           traffic_dict=traffic_dict,
           ips_containers=ips_containers)
      

    def reset_all_flows_metadata(self):
       self.flows_dict = {}


    def reset_single_flow_metadata(self, flow_id):
       del self.flows_dict[flow_id]
