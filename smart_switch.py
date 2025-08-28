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


"""
A Smart L3 switch.

THIS CODE IS AN EXTENSION OF  https://github.com/CPqD/RouteFlow/blob/master/pox/pox/forwarding/l3_learning.py


For each switch:
1) Keep a table that maps IP addresses to MAC addresses and switch ports.
   Stock this table using information from ARP and IP packets.
2) When you see an ARP query, try to answer it using information in the table
   from step 1.  If the info in the table is old, just flood the query.
3) Flood all other ARPs.
4) When you see an IP packet, if you know the destination port (because it's
   in the table from step 1), install a flow for it.
"""
from pox.core import core
from pox.lib.revent import EventMixin
from pox.lib.recoco import Timer
from pox.lib.packet.ipv4 import ipv4
from pox.lib.packet.arp import arp
import pox.openflow.libopenflow_01 as of
from pox.lib.packet.ethernet import ethernet, ETHER_BROADCAST
from pox.lib.addresses import EthAddr
import time
from smartController.entry import Entry
from collections import defaultdict

def dpid_to_mac (dpid):
  return EthAddr("%012x" % (dpid & 0xffFFffFFffFF,))
   

class ForwardingRule(object):
    def __init__(self, source_ip_addr, dest_ip_addr, dest_mac_addr, outgoing_port, dl_type):
      self.source_ip_addr = source_ip_addr
      self.dest_ip_addr = dest_ip_addr
      self.dest_mac_addr = dest_mac_addr
      self.outgoing_port = outgoing_port
      self.dl_type = dl_type
    
    
    def __eq__(self, other):
      if not isinstance(other, ForwardingRule):
        return False
      return (self.source_ip_addr == other.source_ip_addr and
              self.dest_ip_addr == other.dest_ip_addr and
              self.dest_mac_addr == other.dest_mac_addr and
              self.outgoing_port == other.outgoing_port and
              self.dl_type == other.dl_type)
    

class SmartSwitch(EventMixin):
  """
  For each switch:
  1) Keep a table that maps IP addresses to MAC addresses and switch ports.
    Stock this table using information from ARP and IP packets.
  2) When you see an ARP query, try to answer it using information in the table
    from step 1.  If the info in the table is old, just flood the query.
  3) Flood all other ARPs.
  4) When you see an IP packet, if you know the destination port (because it's
    in the table from step 1), install a flow for it.
   """
  def __init__ (
        self,
        flow_logger,
        **kwargs
        ):

    self.flow_idle_timeout = int(kwargs.get('flow_idle_timeout'))
    self.arp_timeout = int(kwargs.get('arp_timeout'))
    self.max_buffered_packets = int(kwargs.get('max_buffered_packets'))
    self.max_buffering_secs = int(kwargs.get('max_buffering_secs'))
    self.arp_req_exp_secs = int(kwargs.get('arp_req_exp_secs'))
    self.logger = kwargs.get('logger')
    # We use this to prevent ARP flooding
    # Key: (switch_id, ARPed_IP) Values: ARP request expire time
    self.recently_sent_ARPs = {}

    # self.unprocessed_flows is a dict where:
    # keys 2-tuples: (switch_id, dst_ip)
    # values: list of 4-tuples: [(expire_time, packet_id, input_port, scr_ip), ...]
    # It flow packets which we can't deliver because we don't know where they go.
    self.unprocessed_flows = {}

    # For each switch, we map destination IP addresses to Entries
    # (Entries are pairs of switch output ports and MAC addresses)
    self.arpTables = {}

    # This timer handles expiring stuff 
    # Doesnt seems having to do with time to live stuff
    self._expire_timer = Timer(5, self._handle_expiration, recurring=True)

    self.flow_logger = flow_logger
    self.openflow_packets_received = 0
    self.forwardingRules = defaultdict(list)
    self.logger.info(f"SmartSwitch initialized!!")


  def _handle_expiration(self):
    
    # Called by a timer so that we can remove old items.
    to_delete_flows = []

    for flow_metadata, packet_metadata_list in self.unprocessed_flows.items():
      switch_id, dest_ip_addr = flow_metadata

      if len(packet_metadata_list) == 0: 
         self.logger.debug("Flow %s expired", flow_metadata)
         to_delete_flows.append(flow_metadata)
      else: 
        for packet_metadata in list(packet_metadata_list):
          
          expires_at, packet_id, in_port, _ = packet_metadata

          if expires_at < time.time():
            # This packet is old. Remove it from the buffer.
            packet_metadata_list.remove(packet_metadata)
            # Tell this switch to drop such a packet:
            # To do that we simply send an action-empty openflow message
            # containing the buffer id and the input port of the switch.
            po = of.ofp_packet_out(buffer_id=packet_id, in_port = in_port)
            core.openflow.sendToDPID(switch_id, po)
            self.logger.debug(f"Expired packet {packet_id} for {flow_metadata}")

    # Remove empty flow entries from the unprocessed_flows dictionary
    for flow_metadata in to_delete_flows:
      del self.unprocessed_flows[flow_metadata]


  def _send_unprocessed_flows(self, switch_id, port, dest_mac_addr, dest_ip_addr):
    """
    Unprocessed flows are those we didn't know
    where to send at the time of arrival.  We may know now.  Try and see.
    """
    query_tuple = (switch_id, dest_ip_addr)
    if query_tuple in self.unprocessed_flows.keys():
      
      bucket = self.unprocessed_flows[query_tuple]    
      del self.unprocessed_flows[query_tuple]

      self.logger.debug(f"Sending {len(bucket)} buffered packets to {dest_ip_addr}")
      
      for _, packet_id, in_port, _ in bucket:
        po = of.ofp_packet_out(buffer_id=packet_id, in_port=in_port)
        po.actions.append(of.ofp_action_dl_addr.set_dst(dest_mac_addr))
        po.actions.append(of.ofp_action_output(port = port))
        core.openflow.sendToDPID(switch_id, po)


  def delete_ip_flow_matching_rules(self, dest_ip, connection):
      switch_id = connection.dpid

      msg = of.ofp_flow_mod(command=of.OFPFC_DELETE)
      msg.match.nw_dst = dest_ip
      msg.match.dl_type = ethernet.IP_TYPE
      connection.send(msg)

      to_delete_frs = []

      for fr in self.forwardingRules[switch_id]:
        if fr.dest_ip_addr == dest_ip:
          to_delete_frs.append(fr)

      for fr in to_delete_frs:
          self.forwardingRules[switch_id].remove(fr)

      self.logger.info(f"Switch {switch_id} will delete flow rules matching nw_dst={dest_ip}")


  def learn_or_update_arp_table(
        self, 
        ip_addr,
        mac_addr,
        port, 
        connection):
      
      switch_id = connection.dpid 

      if ip_addr in self.arpTables[switch_id] and \
        self.arpTables[switch_id][ip_addr] != (port, mac_addr):
            
            # Update switch_port/MAC info
            self.delete_ip_flow_matching_rules(
              dest_ip=ip_addr,
              connection=connection)

      # Learn switch_port/MAC info
      self.arpTables[switch_id][ip_addr] = Entry(
                                            port=port, 
                                            mac=mac_addr, 
                                            ARP_TIMEOUT=self.arp_timeout)
      
      self.logger.debug(f"Entry added/updated to switch {switch_id}'s internal arp table: "+\
                f"(port:{port} ip:{ip_addr})")
        

  def add_ip_to_ip_flow_matching_rule(self, 
                                 switch_id,
                                 source_ip_addr, 
                                 dest_ip_addr, 
                                 dest_mac_addr, 
                                 outgoing_port,
                                 connection,
                                 packet_id,
                                 type):
      
      forwarding_rule = ForwardingRule(
        source_ip_addr=source_ip_addr,
        dest_ip_addr=dest_ip_addr,
        dest_mac_addr=dest_mac_addr,
        outgoing_port=outgoing_port,
        dl_type=type)
      
      if forwarding_rule in self.forwardingRules[switch_id]:
        return

      actions = [of.ofp_action_dl_addr.set_dst(dest_mac_addr),
                of.ofp_action_output(port = outgoing_port)]

      match = of.ofp_match(
        dl_type = type, 
        nw_src = source_ip_addr,
        nw_dst = dest_ip_addr)

      msg = of.ofp_flow_mod(command=of.OFPFC_ADD,
                            idle_timeout=self.flow_idle_timeout,
                            hard_timeout=of.OFP_FLOW_PERMANENT,
                            buffer_id=packet_id,
                            actions=actions,
                            priority=100,   # <-- lower priority
                            match=match)
      
      # if self.add_flow_rule_message_to_buffer(msg, switch_id):
      connection.send(msg.pack())
      self.forwardingRules[switch_id].append(forwarding_rule)

      self.logger.info(f"Added new forwarding flow rule to: {switch_id}"+\
                f" source: {match.nw_src} dest: {match.nw_dst} outgoing port: {outgoing_port}")


  def send_sampling_rules_to_all(self, event):
    
    for stat_obj in event.stats:
      sampling_actions = stat_obj.actions + [of.ofp_action_output(port=of.OFPP_CONTROLLER)]
      sample_msg = of.ofp_flow_mod(
          command=of.OFPFC_ADD,
          idle_timeout=self.flow_idle_timeout,
          hard_timeout=1,   # expires after 1 second
          priority=200,     # <-- higher priority
          actions=sampling_actions,
          match=stat_obj.match
      )
      event.connection.send(sample_msg.pack())

    self.logger.debug(f"Sent {len(event.stats)} sampling rules to switch {event.dpid}")
     
  def build_and_send_ARP_request(
        self, 
        switch_id, 
        incomming_port,
        source_mac_addr,
        source_ip_addr,
        dest_ip_addr,
        connection):
      
      request = arp()
      request.hwtype = request.HW_TYPE_ETHERNET
      request.prototype = request.PROTO_TYPE_IP
      request.hwlen = 6
      request.protolen = request.protolen
      request.opcode = request.REQUEST
      request.hwdst = ETHER_BROADCAST
      request.protodst = dest_ip_addr
      request.hwsrc = source_mac_addr
      request.protosrc = source_ip_addr
      e = ethernet(type=ethernet.ARP_TYPE, src=source_mac_addr,
                    dst=ETHER_BROADCAST)
      e.set_payload(request)
      
      self.logger.debug(f"{switch_id}'s port {incomming_port} ARPing for {dest_ip_addr} on behalf of {source_ip_addr}")

      msg = of.ofp_packet_out()
      msg.data = e.pack()
      msg.actions.append(of.ofp_action_output(port = of.OFPP_FLOOD))
      msg.in_port = incomming_port
      connection.send(msg)


  def add_unprocessed_packet(self, switch_id,dst_ip,port,src_ip,buffer_id):
    self.logger.debug(f"Adding unprocessed packet for {dst_ip}")
    tuple_key = (switch_id, dst_ip)
    if tuple_key not in self.unprocessed_flows: 
      self.unprocessed_flows[tuple_key] = []
    packet_metadata_list = self.unprocessed_flows[tuple_key]
    packet_metadata = (time.time() + self.max_buffering_secs, 
                       buffer_id, 
                       port,
                       src_ip)
    packet_metadata_list.append(packet_metadata)
    while len(packet_metadata_list) > self.max_buffered_packets: 
       del packet_metadata_list[0]


  def handle_unknown_ip_packet(self, switch_id, incomming_port, packet_in_event):
    """
    First, track this buffer so that we can try to resend it later, when we will learn the destination.
    Second, ARP for the destination, which should ultimately result in it responding and us learning where it is
    """
    self.logger.warning(f"Switch {switch_id} received unknown IP packet from port {incomming_port}")

    packet = packet_in_event.parsed
    source_mac_addr = packet.src
    source_ip_addr = packet.next.srcip
    dest_ip_addr = packet.next.dstip
    
    self.add_unprocessed_packet(switch_id=switch_id,
                                dst_ip=dest_ip_addr,
                                port=incomming_port,
                                src_ip=source_ip_addr,
                                buffer_id=packet_in_event.ofp.buffer_id)

    # Expire things from our recently_sent_ARP list...
    self.recently_sent_ARPs = {k:v for k, v in self.recently_sent_ARPs.items() if v > time.time()}

    # Check if we've already ARPed recently
    if (switch_id, dest_ip_addr) in self.recently_sent_ARPs:
      # Oop, we've already done this one recently.
      return

    # Otherwise, ARP...
    self.recently_sent_ARPs[(switch_id, dest_ip_addr)] = time.time() + self.arp_req_exp_secs

    self.build_and_send_ARP_request(
        switch_id, 
        incomming_port,
        source_mac_addr,
        source_ip_addr,
        dest_ip_addr,
        connection=packet_in_event.connection)
    
  
  def try_creating_flow_rule(self, switch_id, incomming_port, packet_in_event):
      
      packet = packet_in_event.parsed
      source_ip_addr = packet.next.srcip
      dest_ip_addr = packet.next.dstip

      if dest_ip_addr in self.arpTables[switch_id]:
          # destination address is present in the arp table
          # get mac and out port
          outgoing_port = self.arpTables[switch_id][dest_ip_addr].port

          if outgoing_port != incomming_port:
              
              dest_mac_addr = self.arpTables[switch_id][dest_ip_addr].mac

              self.add_ip_to_ip_flow_matching_rule(
                                switch_id,
                                source_ip_addr, 
                                dest_ip_addr, 
                                dest_mac_addr, 
                                outgoing_port,
                                connection=packet_in_event.connection,
                                packet_id=packet_in_event.ofp.buffer_id,
                                type=packet.type)

      else:
          self.handle_unknown_ip_packet(switch_id, incomming_port, packet_in_event)


  def handle_ipv4_packet_in(self, switch_id, incomming_port, packet_in_event):
      
      try:
        packet = packet_in_event.parsed  # DNS parsing error occurs during this step
      except:
        # If the parsing fails, just skip this packet without raising an exception
        self.logger.warning(f'error while parsing ipv4 packet_in_event: {packet_in_event}') 
        return

      self.logger.debug("IPV4 DETECTED - SWITCH: %i ON PORT: %i IP SENDER: %s IP RECEIVER %s", 
                switch_id,
                incomming_port,
                packet.next.srcip,
                packet.next.dstip)
      
      # Save packet for inference purposes...
      self.flow_logger.cache_unprocessed_packets(
          src_ip=packet.next.srcip,
          dst_ip=packet.next.dstip,
          packet=packet)
      
      # Send any waiting packets for that ip
      self._send_unprocessed_flows(
         switch_id, 
         incomming_port, 
         dest_mac_addr=packet.src,
         dest_ip_addr=packet.next.srcip)

      self.learn_or_update_arp_table(ip_addr=packet.next.srcip,
                                     mac_addr=packet.src,
                                     port=incomming_port, 
                                     connection=packet_in_event.connection)

      self.try_creating_flow_rule(switch_id, 
                                    incomming_port, 
                                    packet_in_event)


  def send_arp_response(
        self, 
        connection,
        l2_packet,
        l3_packet,
        outgoing_port):
      
      switch_id = connection.dpid

      arp_response = arp()
      arp_response.hwtype = l3_packet.hwtype
      arp_response.prototype = l3_packet.prototype
      arp_response.hwlen = l3_packet.hwlen
      arp_response.protolen = l3_packet.protolen
      arp_response.opcode = arp.REPLY
      arp_response.hwdst = l3_packet.hwsrc
      arp_response.protodst = l3_packet.protosrc
      arp_response.protosrc = l3_packet.protodst
      arp_response.hwsrc = self.arpTables[switch_id][l3_packet.protodst].mac

      ethernet_wrapper = ethernet(type=l2_packet.type, 
                   src=dpid_to_mac(switch_id),
                    dst=l3_packet.hwsrc)
      
      ethernet_wrapper.set_payload(arp_response)

      self.logger.debug(f"ARP ANSWER from switch {switch_id}: ADDRESS:{arp_response.protosrc}")

      msg = of.ofp_packet_out()
      msg.data = ethernet_wrapper.pack()
      msg.actions.append(of.ofp_action_output(port =of.OFPP_IN_PORT))
      msg.in_port = outgoing_port
      connection.send(msg)
      
      
  def handle_arp_packet_in(self, switch_id, incomming_port, packet_in_event):
      
      try:
        packet = packet_in_event.parsed  # DNS parsing error occurs during this step
      except:
        # If the parsing fails, just skip this packet without raising an exception
        self.logger.warning(f'error while parsing arp packet_in_event: {packet_in_event}') 
        return
      
      packet = packet_in_event.parsed
      inner_packet = packet.next

      arp_operation = ''
      if inner_packet.opcode == arp.REQUEST: arp_operation = 'request'
      elif inner_packet.opcode == arp.REPLY: arp_operation = 'reply'
      else: arp_operation = 'op_'+ str(inner_packet.opcode)

      self.logger.debug(f"ARP {arp_operation} received: SWITCH: {switch_id} IN PORT:{incomming_port} ARP FROM: {inner_packet.protosrc} TO {inner_packet.protodst}")

      if inner_packet.prototype == arp.PROTO_TYPE_IP and \
        inner_packet.hwtype == arp.HW_TYPE_ETHERNET and \
          inner_packet.protosrc != 0:

          self.learn_or_update_arp_table(ip_addr=inner_packet.protosrc,
                                     mac_addr=packet.src,
                                     port=incomming_port, 
                                     connection=packet_in_event.connection)
          
          # Send any waiting packets...
          self._send_unprocessed_flows(
              switch_id, 
              incomming_port, 
              dest_mac_addr=packet.src,
              dest_ip_addr=inner_packet.protosrc)

          if inner_packet.opcode == arp.REQUEST and \
            inner_packet.protodst in self.arpTables[switch_id] and \
              not self.arpTables[switch_id][inner_packet.protodst].isExpired():
                '''
                An ARP request has been received, the corresponding 
                switch has the answer, and such an answer is not expired
                '''

                self.send_arp_response(connection=packet_in_event.connection,
                                       l2_packet=packet,
                                       l3_packet=inner_packet,
                                       outgoing_port=incomming_port)

                return

      # Didn't know how to answer or otherwise handle the received ARP, so just flood it
      self.logger.debug(f"Flooding ARP {arp_operation} Switch: {switch_id} IN_PORT: {incomming_port} from:{inner_packet.protosrc} to:{inner_packet.protodst}")

      msg = of.ofp_packet_out(
         in_port = incomming_port, 
         data = packet_in_event.ofp,
         action = of.ofp_action_output(port = of.OFPP_FLOOD))
      
      packet_in_event.connection.send(msg)


  def _handle_openflow_PacketIn(self, event):
    self.logger.debug('handling openflow packet_in_event')
    self.openflow_packets_received += 1
    switch_id = event.connection.dpid
    incomming_port = event.port
    try:
      packet = event.parsed  # DNS parsing error occurs during this step
    except:
      # If the parsing fails, just skip this packet without raising an exception
      self.logger.warning(f'error while parsing openflow packet_in_event: {event}') 
      return 
       
    if not packet.parsed:
      self.logger.warning(f"switch {switch_id}, port {incomming_port}: ignoring unparsed packet")
      return
  
    if switch_id not in self.arpTables:
      # New switch -- create an empty table
      self.logger.info(f"New switch detected - creating empty flow table with id {switch_id}")
      self.arpTables[switch_id] = {}
      
    if packet.type == ethernet.LLDP_TYPE:
      #Ignore lldp packets
      return

    if isinstance(packet.next, ipv4):
      
        self.handle_ipv4_packet_in(
          switch_id=switch_id,
          incomming_port=incomming_port,
          packet_in_event=event)

    elif isinstance(packet.next, arp):
        self.handle_arp_packet_in(
          switch_id=switch_id,
          incomming_port=incomming_port,
          packet_in_event=event
        )
      
