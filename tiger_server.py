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
from fastapi.responses import JSONResponse
import uvicorn
import threading
import os
import atexit
import signal
import traceback
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
flowstats_listener = None
# Set by smart_check() if the inference loop dies on an uncaught exception.
# A daemon thread crashing leaves the FastAPI process up and answering
# requests, so without this the experiment looks "alive" for the rest of an
# unattended multi-hour run while doing nothing. /health surfaces it so the
# CLI driver can detect and abort instead of waiting out the full duration.
controller_crash_info = None

tiger_lock = Lock()

def dpid_to_mac (dpid):
  return EthAddr("%012x" % (dpid & 0xffFFffFFffFF,))


def notify_wandb_alert(title, text, level="ERROR"):
  """
  Best-effort W&B alert (emails/Slacks the user if Alerts are configured on
  the account). Must never itself raise, since it's called from error
  handlers and a crashing inference loop.
  """
  global wb_tracker
  try:
    if wb_tracker is None or getattr(wb_tracker, "wb_run", None) is None:
      return
    import wandb
    alert_level = getattr(wandb.AlertLevel, level, wandb.AlertLevel.ERROR)
    wb_tracker.wb_run.alert(title=title, text=text, level=alert_level)
  except Exception as alert_exc:
    logger.warning(f"Failed to send W&B alert ({title}): {alert_exc}")


def _init_error(msg):
  """
  Logs, fires a W&B alert (if a run is active), and returns a JSONResponse
  carrying the real HTTP 500 status code. Plain dict returns from a FastAPI
  route default to HTTP 200 regardless of any "status_code" key inside the
  body, which previously made callers (the dashboard, dash_cli.py) see
  /initialize failures as successes.
  """
  logger.error(msg)
  notify_wandb_alert(title="TIGER initialize failed", text=msg)
  return JSONResponse(status_code=500, content={"status_code": 500, "msg": msg})


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

        """
        # Start the parallel echo microservice (runs on internal net)
        if echo_thread is None or not echo_thread.is_alive():
          logger.info("Parallel Echo microservice is starting...")
          echo_thread = threading.Thread(target=run_echo_server, daemon=True)
          echo_thread.start()
        """

def smart_check():
  global args, wb_tracker, stop_tiger_threads, controller_crash_info

  logger.info("Starting SmartSwitch inference loop")
  check_count = 0

  INFERENCE_LOOP_MIN_PERIOD = 0.05  # seconds
  while not stop_tiger_threads:
    loop_start = time.time()
    check_count += 1

    try:
      with tiger_lock:

        controller_brain.process_input(
          flows=list(flow_logger.flows_dict.values()),
          node_feats=(metrics_logger.metrics_dict if args['health_monitoring'] else None))


        if check_count % 100 == 0:
          report_dict = {}
          for key, flow in list(flow_logger.flows_dict.items()):
            report_dict[f'packetcounts/{key}'] = flow.packet_count

          profiling_metrics = controller_brain.get_profiling_stats_dict()
          report_dict.update(profiling_metrics)

          if controller_brain.data_recorder is not None:
            report_dict.update(controller_brain.data_recorder.get_status_dict())
            logger.debug(f"[DataRecorder] status: {controller_brain.data_recorder.get_status_dict()}")

          wb_tracker.wb_run.log(report_dict, step=wb_tracker.step_counter)

    except Exception as e:
      error_text = f"{e}\n{traceback.format_exc()}"
      logger.error(f"Inference loop crashed at check #{check_count}: {error_text}")
      controller_crash_info = {
        "error": str(e),
        "traceback": traceback.format_exc(),
        "check_count": check_count,
        "timestamp": time.time(),
      }
      # Stop both background threads rather than spinning on the same
      # exception (or silently doing nothing) for the rest of the run.
      stop_tiger_threads = True
      notify_wandb_alert(
        title="TIGER inference loop crashed",
        text=f"smart_check() died at check #{check_count}: {error_text}",
      )
      break

    # sleep when there has not been an inference so that OVS can recconnect.
    time.sleep(max(0.0, INFERENCE_LOOP_MIN_PERIOD - (time.time() - loop_start)))





def add_ip_to_no_proxy_env_var(monitor_ip):
  no_proxy = os.environ.get('no_proxy', '')
  if no_proxy != '':
      no_proxy += ','
  no_proxy += monitor_ip
  os.environ['no_proxy'] = no_proxy
  logger.info(f"Fixed no_proxy to {no_proxy}")
    

def _remove_flowstats_listeners():
  """
  Remove any previously registered FlowStatsReceived listeners.

  This must be called both on /stop and at the top of /initialize.
  If skipped, every re-initialization stacks a new listener on top of
  the surviving old one, so buffers keep being populated by callbacks
  from the previous experiment even when traffic is idle.
  """
  global flowstats_listener

  if flowstats_listener is not None:
    try:
      removed = core.openflow.removeListener(flowstats_listener)
      if removed: logger.info("Removed flowstats listener")
      else: logger.warning("Could not remove flowstats listener")
    except Exception as e:
      logger.error(f"Could not remove flowstats listener: {e}")
    flowstats_listener = None


def shutdown_process():
  global stop_tiger_threads, inference_thread, flowstats_req_thread, metrics_logger, controller_brain
  global flow_logger, smart_switch

  logger.info("Shutdown command received")
  
  stop_tiger_threads = True

  if inference_thread is not None:
    inference_thread.join(timeout=5)
    inference_thread = None
  if flowstats_req_thread is not None:
    flowstats_req_thread.join(timeout=5)
    flowstats_req_thread = None

  # detach FlowStats listeners so no stale callbacks fire after stop.
  _remove_flowstats_listeners()


  # Pause the SmartSwitch PacketIn handler.
  # POX does not support unregistering components, so the _handle_openflow_
  # PacketIn listener stays alive forever.  Setting paused=True makes it
  # return immediately, stopping cache_unprocessed_packets() calls and
  # preventing any further buffer growth between experiments.
  if smart_switch is not None:
    smart_switch.paused = True
    logger.info("SmartSwitch paused Flowlogging...")

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
    # echo_app = FastAPI(title="SmartSwitch Echo API", description="API for internal overhead tracking")

    @app.get("/")
    async def root():
        logger.info("Root endpoint called")
        return {"msg": "Hello World from the SmartSwitch!"}
    

    @app.post("/stop")
    async def shutdown():

        shutdown_process()

        return {"status_code": 200, "msg": "SmartSwitch is stopped"}


    @app.get("/health")
    async def health():
        """
        Lets a sweep/driver script detect a crashed-but-still-running
        experiment (e.g. the background inference thread died on an
        uncaught exception) without waiting out the full run duration, and
        re-confirms which seed/agent the currently running experiment was
        actually initialized with.
        """
        if controller_crash_info is not None:
          return JSONResponse(status_code=500, content={
            "status_code": 500,
            "status": "crashed",
            "crash_info": controller_crash_info,
          })

        return {
          "status_code": 200,
          "status": "ok" if controller_brain is not None else "not_initialized",
          "applied_seed": controller_brain.seed if controller_brain is not None else None,
          "applied_agent": args['intrusion_detection'].get('agent') if args is not None else None,
        }



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

    """
    @echo_app.get("/echo")
    def echo_target():
        
        # Application-layer echo.
        # Simulates a lightweight microservice response.
        
        return {"status": "ok", "timestamp": time.time()}
    """

    @app.post("/initialize")
    async def initialize(init_controller_args: dict):
        global traffic_dict, rewards, container_ips, stop_tiger_threads
        global flow_logger, metrics_logger, controller_brain, smart_switch, wb_tracker
        global FLOWSTATS_FREQ_SECS, args, flowstats_req_thread, inference_thread
        global flowstats_listener, controller_crash_info

        # Reset before anything else so an early failure (or a stale handle
        # from a previous experiment) never gets misattributed: notify_wandb_alert
        # below should only ever fire against *this* request's run, not a
        # leftover one.
        wb_tracker = None
        controller_crash_info = None

        try:
          logger.setLevel(init_controller_args.get("smart_controller_log_level").upper())
          logger.info(f"Initialisation command received")

          add_ip_to_no_proxy_env_var(init_controller_args.get("monitor_ip"))

          args = init_controller_args
          args['logger'] = logger


        except Exception as e:
          shutdown_process()
          return _init_error(f"Error parsing initialisation command: {e}")

        try:
          wb_tracker = WandBTracker(args)
        except Exception as e:
          return _init_error(f"Error initialising wandb tracker: {e}")


        try:
          flow_logger = FlowLogger(wb_tracker=wb_tracker, **args)
        except Exception as e:
          shutdown_process()
          return _init_error(f"Error initialising flow logger: {e}")

        try:
          if args['health_monitoring']:
              metrics_logger = MetricsLogger(args, wb_tracker=wb_tracker)
          else:
             logger.info("Metrics logger is not enabled")
        except Exception as e:
          return _init_error(f"Error initialising metrics logger: {e}")

        try:
          # The controllerBrain holds the ML functionalities.
          controller_brain = TigerBrain(args, wb_tracker = wb_tracker)
        except Exception as e:
          shutdown_process()
          return _init_error(f"Error creating controller brain: {e}")


        try:
          if not core.hasComponent("smart_switch"):

            # Registering Switch component:
            smart_switch = SmartSwitch(
              flow_logger=flow_logger,
              wb_tracker=wb_tracker,
              **args
              )
            core.register("smart_switch", smart_switch)
            core.listen_to_dependencies(smart_switch)

          else:
            logger.info("SmartSwitch already registered — re-initialising")
            smart_switch = core.components["smart_switch"]
            smart_switch.initialize(
              flow_logger=flow_logger,
              wb_tracker=wb_tracker,
              **args
            )
            logger.info("SmartSwitch re-initialised!")

        except Exception as e:
            shutdown_process()
            return _init_error(f"Error creating SmartSwitch: {e}")


        try:
           if metrics_logger is not None:
              metrics_logger.init()
        except Exception as e:
          shutdown_process()
          return _init_error(f"Error initialising metrics logger: {e}")


        FLOWSTATS_FREQ_SECS = float(args['intrusion_detection']["flowstats_freq_secs"])
        
        if FLOWSTATS_FREQ_SECS > 0:

          # remove any surviving listeners from a previous experiment before adding fresh ones.
          _remove_flowstats_listeners()

          flowstats_listener = core.openflow.addListenerByName(
            "FlowStatsReceived", 
            lambda event: flow_logger._handle_flowstats_received(
              event, 
              controller_brain.env.current_knowledge,
              controller_brain.traffic_dict,
              controller_brain.ips_containers))
          
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

        return {
          "msg": "SmartSwitch initialized successfully",
          "status_code": 200,
          # Echoed back so callers (dash_cli.py / sweep scripts) can assert
          # the seed/agent they sent were the ones actually applied, instead
          # of just trusting that the request went through.
          "applied_seed": controller_brain.seed,
          "applied_agent": args['intrusion_detection'].get('agent'),
        }
    

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)


    core.openflow.addListenerByName(
        "ConnectionUp", 
        _handle_ConnectionUp)