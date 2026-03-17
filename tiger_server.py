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
import pox.openflow.libopenflow_01 as of
from pox.lib.addresses import EthAddr

from smartController.flowlogger_new import FlowLogger
from smartController.tiger_brain_new import TigerBrain
from smartController.metricslogger import MetricsLogger
from smartController.smart_switch import SmartSwitch
from smartController.wandb_tracker import WandBTracker

import subprocess
from fastapi import FastAPI
import uvicorn
import threading
import os
import atexit
import signal
from threading import Lock
import time
import logging 




SUPPRESSED_ENDPOINTS = [
   '/check_zookeeper', 
   '/check_kafka',
   '/check_prometheus',
   '/check_grafana',
   '/metrics',
   '/echo'
 ]

class SuppressEndpointFilter(logging.Filter):
    def filter(self, record):
        # Check if the log record has the necessary arguments (for Uvicorn access logs)
        if record.args and len(record.args) >= 3:
            # record.args[2] contains the path (including query parameters)
            path = record.args[2]
            # Check if the path is in the list of suppressed endpoints
            if path in SUPPRESSED_ENDPOINTS:
                return False  # Suppress this log entry
        return True  # Allow other log entries

# Get the Uvicorn access logger and add the filter
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(SuppressEndpointFilter())


logger = core.getLogger()
logger.name = "SmartvilleController"

app_thread = None  # Thread for the FastAPI server
app = None  # FastAPI app instance
echo_app = None      # FastAPI instance for the parallel echo microservice
echo_thread = None   # Thread for the echo server
args = None
openflow_connection = None  # openflow connection to switch is stored here
FLOWSTATS_FREQ_SECS = None  # Interval in which the FLOW stats request is triggered
traffic_dict = None
rewards = None
smart_switch = None
container_ips = None
flow_logger = None
metrics_logger = None
controller_brain = None
stop_tiger_threads = True
flowstats_req_thread = None
inference_thread = None

tiger_lock = Lock()

def dpid_to_mac (dpid):
  return EthAddr("%012x" % (dpid & 0xffFFffFFffFF,))
   

def periodically_requests_stats(period):
  
  while not stop_tiger_threads:
    
    with tiger_lock:
      connections_snapshot = list(core.openflow._connections.values())[:]
    
    for connection in connections_snapshot:
      connection.send(of.ofp_stats_request(body=of.ofp_flow_stats_request()))
      connection.send(of.ofp_stats_request(body=of.ofp_port_stats_request()))
    
    logger.debug("Sent %i flow/port stats request(s)", len(core.openflow._connections))
    
    # Sleep in small increments to react quickly to shutdown ---
    elapsed = 0
    while elapsed < period and not stop_tiger_threads:
        time.sleep(0.1)
        elapsed += 0.1



def pprint(obj):
    for key, value in obj.items():
        if isinstance(value, dict):
            pprint(value)
        else:
          logger.debug(f"{key}: {value}")


def run_server():
  global app
  # Start the FastAPI server
  try:
        port = int(os.environ.get("SERVER_PORT"))
  except Exception as e:
      print(f"Error parsing SERVER_PORT env var: {e}")
      assert False

  uvicorn.run(app, host="0.0.0.0", port=port)


def run_echo_server():
    """Parallel lightweight echo microservice on the specific IP/port."""
    global echo_app

    try:
       internal_ip = os.environ.get("INTERNAL_IP")
       echo_port = int(os.environ.get("ECHO_PORT"))
    except Exception as e:
       print(f"Error parsing env vars: {e}")
       assert False
    uvicorn.run(echo_app, host=internal_ip, port=echo_port)


def _handle_ConnectionUp (event):
      global openflow_connection, app_thread, echo_thread
      openflow_connection=event.connection
      logger.info("Connection is UP")

      if app_thread is None or not app_thread.is_alive():
        logger.info("SmartSwitch API is starting...")
        app_thread = threading.Thread(target=run_server, daemon=True)
        app_thread.start()

        # Start the parallel echo microservice (runs on 192.168.1.1:7778)
        if echo_thread is None or not echo_thread.is_alive():
          logger.info("Parallel Echo microservice is starting...")
          echo_thread = threading.Thread(target=run_echo_server, daemon=True)
          echo_thread.start()
     

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
  global args

  logger.info("Starting SmartSwitch inference loop")
  inference_count = 0
  
  while not stop_tiger_threads:

    inference_count += 1

    with tiger_lock:
      
      controller_brain.process_input(
        flows=list(flow_logger.flows_dict.values()),
        node_feats=(metrics_logger.metrics_dict if args['health_monitoring'] else None))
      

      if inference_count % 100 == 0:
        for key in flow_logger.flows_dict.keys():
              logger.info(f"Packets seen for {key}: {flow_logger.flows_dict[key].packet_feat_circular_buffer.calls_to_add}")


def add_ip_to_no_proxy_env_var(monitor_ip):
  no_proxy = os.environ.get('no_proxy', '')
  if no_proxy != '':
      no_proxy += ','
  no_proxy += monitor_ip
  os.environ['no_proxy'] = no_proxy
  logger.info(f"Fixed no_proxy to {no_proxy}")
    


def shutdown_process():
  global stop_tiger_threads, inference_thread, flowstats_req_thread, metrics_logger, controller_brain

  logger.info("Shutdown command received")
  
  stop_tiger_threads = True
  if inference_thread is not None:
    inference_thread.join()
  if flowstats_req_thread is not None:
    flowstats_req_thread.join()

  if metrics_logger is not None:
    metrics_logger.shutdown()
  metrics_logger = None

  if controller_brain is not None:
    controller_brain.shutdown()
  controller_brain = None
   


def launch(**kwargs):     
    global app, app_thread, openflow_connection, smart_switch, echo_app
    global flow_logger, metrics_logger, controller_brain, FLOWSTATS_FREQ_SECS, args
    
    app = FastAPI(title="SmartSwitch API", description="API for ML experiments")
    echo_app = FastAPI(title="SmartSwitch Echo API", description="API for internal overhead tracking")

    @app.get("/")
    async def root():
        logger.info("Root endpoint called")
        return {"msg": "Hello World from the SmartSwitch!"}
    

    @app.post("/stop")
    async def shutdown():
        
        shutdown_process()

        return {"status_code": 200, "msg": "SmartSwitch is stopped"}
    

    @app.post("/sync_wandb")
    async def sync_wandb():
        root_dir = '/pox/pox/smartController/wandb'
        folders = [f.path for f in os.scandir(root_dir) if f.is_dir() and 'run-' in f.path]
        # Loop through each folder and sync it with wandb
        for folder in folders:
            print(f"Syncing {folder}")
            subprocess.run(['wandb', 'sync', folder])
        return {"status_code": 200, "msg": "Wandb is synced"}

    def cleanup():
      logger.info("Cleaning up before exit")
      return shutdown_process()

    def handle_sigterm(signum, frame):
      cleanup()
      os._exit(0)  # Force exit


    @echo_app.get("/echo")
    def echo_target():
        """
        Application-layer echo.
        Simulates a lightweight microservice response.
        """
        return {"status": "ok", "timestamp": time.time()}


    @app.post("/initialize")
    async def initialize(kwargs: dict):
        global traffic_dict, rewards, container_ips, stop_tiger_threads
        global flow_logger, metrics_logger, controller_brain, smart_switch, wb_tracker
        global FLOWSTATS_FREQ_SECS, args, flowstats_req_thread, inference_thread

        try:
          logger.setLevel(kwargs.get("smart_controller_log_level").upper())
          logger.info(f"Initialisation command received")

          pprint(kwargs)

          add_ip_to_no_proxy_env_var(kwargs.get("monitor_ip"))

          args = kwargs
          args['logger'] = logger
          
          intrusion_detection_args = kwargs.get("intrusion_detection", {})
          intrusion_detection_args['rewards'] = kwargs.get("rewards", {})
          intrusion_detection_args['knowledge'] = kwargs.get("knowledge", {})
          intrusion_detection_args['logger'] = logger
          intrusion_detection_args['models'] = kwargs.get("models", {})
        except Exception as e:
          logger.error(f"Error parsing initialisation command: {e}")
          shutdown_process()
          return {"status_code": 500, "msg": f"Error parsing initialisation command: {e}"}
        
        wb_tracker = None
        try:
          wb_tracker = WandBTracker(args)
        except Exception as e:
          logger.error(f"Error initialising wandb tracker: {e}")
          return {"status_code": 500, "msg": f"Error initialising wandb tracker: {e}"}


        try:
          flow_logger = FlowLogger(**args)
        except Exception as e:
          logger.error(f"Error initialising flow logger: {e}")
          shutdown_process()
          return {"status_code": 500, "msg": f"Error initialising flow logger: {e}"}

        try:
          if args['health_monitoring']:
              metrics_logger = MetricsLogger(args, wb_tracker=wb_tracker)
          else:
             logger.info("Metrics logger is not enabled")
        except Exception as e:
          logger.error(f"Error creating metrics logger: {e}")
          return {"status_code": 500, "msg": f"Error initialising metrics logger: {e}"} 

        try:
          # The controllerBrain holds the ML functionalities.
          controller_brain = TigerBrain(args, wb_tracker = wb_tracker)
        except Exception as e:
          logger.error(f"Error creating controller brain: {e}")
          shutdown_process()
          return {"status_code": 500, "msg": f"Error creating controller brain: {e}"}

        
        try:
          if not core.hasComponent("smart_switch"):

            switch_args = get_switching_args()
            switch_args.update(args)
          
            # Registering Switch component:
            smart_switch = SmartSwitch(
              flow_logger=flow_logger,
              **switch_args
              )
            core.register("smart_switch", smart_switch) 
            core.listen_to_dependencies(smart_switch)
          
          else:
            logger.info("SmartSwitch already registered")
            smart_switch = core.components["smart_switch"]
            smart_switch.flow_logger = flow_logger # we need to update the flow logger instance attached to the SmartSwitch
            smart_switch.initialize()

        except Exception as e:
            logger.error(f"Error creating SmartSwitch: {e}")
            shutdown_process()
            return {"status_code": 500, "msg": f"Error creating SmartSwitch: {e}"}
        

        try:
           if metrics_logger is not None:
              metrics_logger.init()
        except Exception as e:
          logger.error(f"Error initialising metrics logger: {e}")
          shutdown_process()
          return {"status_code": 500, "msg": f"Error initialising metrics logger: {e}"}


        FLOWSTATS_FREQ_SECS = float(intrusion_detection_args["flowstats_freq_secs"])
        
        if FLOWSTATS_FREQ_SECS > 0:
          core.openflow.addListenerByName(
            "FlowStatsReceived", 
            lambda event: flow_logger._handle_flowstats_received(
              event, 
              controller_brain.env.current_knowledge,
              controller_brain.traffic_dict,
              controller_brain.ips_containers))
          
          if args['intrusion_detection']['resample_packets']:
              logger.info("Enabling packet resampling")
              core.openflow.addListenerByName(
                "FlowStatsReceived", 
                lambda event: smart_switch.send_sampling_rules_to_all(
                  event))
          
          flowstats_req_thread = threading.Thread(
            target=periodically_requests_stats,
            args=(FLOWSTATS_FREQ_SECS,),
            daemon=True
          )

          inference_thread = threading.Thread(
            target=smart_check,
            daemon=True
          )

          stop_tiger_threads = False
          flowstats_req_thread.start()
          inference_thread.start()

        return {"msg": "SmartSwitch initialized successfully", "status_code": 200}
    

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)


    core.openflow.addListenerByName(
        "ConnectionUp", 
        _handle_ConnectionUp)