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
from pox.lib.packet.ethernet import ethernet, ETHER_BROADCAST
from pox.lib.packet.ipv4 import ipv4
from pox.lib.packet.arp import arp
from pox.lib.recoco import Timer
from pox.lib.util import str_to_bool
import pox.openflow.libopenflow_01 as of
from pox.lib.revent import *
import time
from pox.lib.recoco import Timer
from pox.openflow.of_json import *
from pox.lib.addresses import EthAddr
from smartController.entry import Entry
from smartController.flowlogger_new import FlowLogger
from smartController.tiger_brain import TigerBrain
from smartController.metricslogger import MetricsLogger
from collections import defaultdict
import requests
from fastapi import FastAPI
import uvicorn


log = core.getLogger()
log.name = "TigerServer"
openflow_connection = None  # openflow connection to switch is stored here
FLOWSTATS_FREQ_SECS = None  # Interval in which the FLOW stats request is triggered
honeypots = None
attackers = None
rewards = None
knowledge = None
container_ips = None


def dpid_to_mac (dpid):
  return EthAddr("%012x" % (dpid & 0xffFFffFFffFF,))
   

def _handle_ConnectionUp (event):
  global openflow_connection
  openflow_connection=event.connection
  log.info("Connection is UP")
  # Request stats periodically
  # Timer(FLOWSTATS_FREQ_SECS, requests_stats, recurring=True)


def requests_stats():
  for connection in core.openflow._connections.values():
    connection.send(of.ofp_stats_request(body=of.ofp_flow_stats_request()))
    connection.send(of.ofp_stats_request(body=of.ofp_port_stats_request()))
  log.debug("Sent %i flow/port stats request(s)", len(core.openflow._connections))



def launch(**kwargs):     
    global app
    
    app = FastAPI(title="TigerServer API", description="API for ML experiments")

    # attach handlers to listeners
    core.openflow.addListenerByName(
      "ConnectionUp", 
      _handle_ConnectionUp)
    

    @app.get("/")
    async def root():
        log.info("Root endpoint called")
        return {"msg": "Hello World from the TigerServer!"}
    

    @app.post("/curricula")
    async def set_curricula(curricula: dict):
        global honeypots, attackers, rewards, knowledge, container_ips

        log.info(f"Curricula received:")
        
        def pprint(obj):
          for key, value in obj.items():
              if isinstance(value, dict):
                  pprint(value)
              else:
                log.info(f"{key}: {value}")

        pprint(curricula)
        honeypots = curricula.get("honeypots", [])
        attackers = curricula.get("attackers", [])
        rewards = curricula.get("rewards", {})
        knowledge = curricula.get("knowledge", {})
        container_ips = curricula.get("container_ips", {})

        # Here you would typically process the curricula
        return {"msg": "Curricula updated successfully", "status_code": 200}


    @app.post("/flowlogger")
    async def set_flowlogger(flowlogger: dict):
        global flow_logger
        log.info(f"Flowlogger received:")
        flowlogger = FlowLogger(
            flowlogger.get("multi_class", False),
            flowlogger.get("packet_buffer_len", 0),
            flowlogger.get("packet_feat_dim", {}),
            flowlogger.get("packet_cache", {}),
            flowlogger.get("anonymize_transport_ports", True),
            flowlogger.get("flow_feat_dim", 4),
            flowlogger.get("flow_buff_len", {})
        )
        return {"msg": "Flowlogger initialized successfully", "status_code": 200}
    

    log.info("TigerServer API is starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
