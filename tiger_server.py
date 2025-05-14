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
from pox.lib.util import str_to_bool
import pox.openflow.libopenflow_01 as of
from pox.lib.addresses import EthAddr
from smartController.flowlogger_new import FlowLogger
from smartController.tiger_brain_new import TigerBrain
from smartController.metricslogger import MetricsLogger
from smartController.smart_switch import SmartSwitch
from fastapi import FastAPI
import uvicorn
from pox.lib.recoco import Timer
import logging
import threading
import os

logger = core.getLogger()
logger.name = "TigerServer"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
app_thread = None  # Thread for the FastAPI server
app = None  # FastAPI app instance
args = None
openflow_connection = None  # openflow connection to switch is stored here
FLOWSTATS_FREQ_SECS = None  # Interval in which the FLOW stats request is triggered
traffic_dict = None
rewards = None
init_knowledge = None
current_knowledge = None
smart_switch = None
container_ips = None
flow_logger = None
metrics_logger = None
controller_brain = None

def dpid_to_mac (dpid):
  return EthAddr("%012x" % (dpid & 0xffFFffFFffFF,))
   

def requests_stats():
  for connection in core.openflow._connections.values():
    connection.send(of.ofp_stats_request(body=of.ofp_flow_stats_request()))
    connection.send(of.ofp_stats_request(body=of.ofp_port_stats_request()))
  logger.debug("Sent %i flow/port stats request(s)", len(core.openflow._connections))
  

def pprint(obj):
    for key, value in obj.items():
        if isinstance(value, dict):
            pprint(value)
        else:
          logger.debug(f"{key}: {value}")


def run_server():
  global app
  # Start the FastAPI server
  uvicorn.run(app, host="0.0.0.0", port=8000)


def _handle_ConnectionUp (event):
      global openflow_connection, app_thread
      openflow_connection=event.connection
      logger.info("Connection is UP")

      if app_thread is None or not app_thread.is_alive():
        logger.info("TigerServer API is starting...")
        app_thread = threading.Thread(target=run_server, daemon=True)
        app_thread.start()
     

def get_switching_args():

  switching_args = {
    'flow_idle_timeout' : os.getenv('flow_idle_timeout'),
    'arp_timeout' : os.getenv('arp_timeout'),
    'max_buffered_packets' : os.getenv('max_buffered_packets'),
    'max_buffering_secs' : os.getenv('max_buffering_secs'),
    'arp_req_exp_secs' : os.getenv('arp_req_exp_secs'),
    'logger' :logger
    }

  return switching_args



def smart_check():
  global current_knowledge, args

  epistemic_updates = controller_brain.process_input(
    flows=list(flow_logger.flows_dict.values()),
    node_feats=(metrics_logger.metrics_dict if args['intrusion_detection']['node_features'] else None))
  
  if epistemic_updates is not None:
      
      current_knowledge = epistemic_updates['current_knowledge']
      discovered_attack = epistemic_updates['new_label']

      if 'reset' in epistemic_updates:
          logger.info(f'Curricula reset taken out')
      elif discovered_attack is not None:
          logger.info(f'Epistemic updates taken out: {discovered_attack} is no more an unknown attack.')


def launch(**kwargs):     
    global app, app_thread, openflow_connection, smart_switch, init_knowledge, current_knowledge
    global flow_logger, metrics_logger, controller_brain, FLOWSTATS_FREQ_SECS, args

    # Registering Switch component:
    smart_switch = SmartSwitch(
      **get_switching_args()
      )
    
    core.register("smart_switch", smart_switch) 
    core.listen_to_dependencies(smart_switch)
    
    app = FastAPI(title="TigerServer API", description="API for ML experiments")
    
    @app.get("/")
    async def root():
        logger.info("Root endpoint called")
        return {"msg": "Hello World from the TigerServer!"}
    

    @app.post("/initialize")
    async def initialize(kwargs: dict):
        global traffic_dict, rewards, init_knowledge, container_ips, current_knowledge
        global flow_logger, metrics_logger, controller_brain, smart_switch
        global FLOWSTATS_FREQ_SECS, args

        logger.info(f"Initialisation command received")

        pprint(kwargs)

        args = kwargs
        traffic_dict = kwargs.get("traffic_dict", [])
        rewards = kwargs.get("rewards", {})
        init_knowledge = kwargs.get("knowledge", {})
        current_knowledge = init_knowledge.copy()
        container_ips = kwargs.get("container_ips", {})
        ips_containers = kwargs.get("ips_containers", {})
        

        intrusion_detection_args = kwargs.get("intrusion_detection", {})
        intrusion_detection_args['container_ips'] = container_ips
        intrusion_detection_args['ips_containers'] = ips_containers
        intrusion_detection_args['traffic_dict'] = traffic_dict
        intrusion_detection_args['rewards'] = rewards
        intrusion_detection_args['knowledge'] = init_knowledge
        intrusion_detection_args['logger'] = logger

        flow_logger = FlowLogger(
            intrusion_detection_args.get("multi_class", False),
            intrusion_detection_args.get("packet_buffer_len", 0),
            intrusion_detection_args.get("packet_feat_dim", 64),
            intrusion_detection_args.get("anonymize_transport_ports", True),
            intrusion_detection_args.get("flow_feat_dim", 4),
            intrusion_detection_args.get("flow_buff_len", 10)
        )

        if intrusion_detection_args.get("node_features", False):
            metrics_logger = MetricsLogger(
              server_addr = "192.168.1.1:9092",
              max_conn_retries = int(intrusion_detection_args.get('max_kafka_conn_retries', 5)),
              metric_buffer_len = int(intrusion_detection_args.get('metric_buffer_len', 10)),
              grafana_user=intrusion_detection_args.get('grafana_user', 'admin'), 
              grafana_pass=intrusion_detection_args.get('grafana_password', 'admin'),
              )
            

        # The controllerBrain holds the ML functionalities.
        controller_brain = TigerBrain(
            eval=eval,
            flow_feat_dim=intrusion_detection_args.get("flow_feat_dim", 4),
            packet_feat_dim=intrusion_detection_args.get("packet_feat_dim", 64),
            dropout=intrusion_detection_args.get("packet_feat_dim", 0.6),
            multi_class=str_to_bool(intrusion_detection_args.get('multi_class', True)), 
            init_k_shot=int(intrusion_detection_args.get('init_k_shot', 5)),
            replay_buffer_batch_size=int(intrusion_detection_args.get('batch_size', 20)),
            kernel_regression=str_to_bool(intrusion_detection_args.get('kernel_regression', True)),
            device=intrusion_detection_args.get('device', 'cpu'),
            seed=int(intrusion_detection_args.get('seed', 777)),
            debug=str_to_bool(intrusion_detection_args.get('ai_debug', False)),
            wb_track=str_to_bool(intrusion_detection_args.get('wb_tracking', False)),
            wb_project_name=intrusion_detection_args.get('wb_project_name', 'TIGER'),
            wb_run_name=intrusion_detection_args.get('wb_run_name', f"my_run"),
            report_step_freq=int(intrusion_detection_args.get('report_step_freq',50)),
            kwargs=intrusion_detection_args)
        

        FLOWSTATS_FREQ_SECS = int(intrusion_detection_args.get("flowstats_freq_secs", 5))
        
        if FLOWSTATS_FREQ_SECS > 0:
          core.openflow.addListenerByName(
            "FlowStatsReceived", 
            lambda event: flow_logger._handle_flowstats_received(
              event, 
              current_knowledge,
              traffic_dict,
              ips_containers))
    
          # Request stats periodically
          Timer(FLOWSTATS_FREQ_SECS, requests_stats, recurring=True)
          # Periodic Training and Inference
          Timer(intrusion_detection_args['inference_freq_secs'], smart_check, recurring=True) 

        return {"msg": "TigerController initialized successfully", "status_code": 200}
    

    core.openflow.addListenerByName(
        "ConnectionUp", 
        _handle_ConnectionUp)
