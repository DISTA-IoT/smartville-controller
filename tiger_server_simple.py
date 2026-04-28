# This file is part of the "Smartville" project.
# Simplified tiger_server for scientific ablation of epistemic actions.

import os
import torch
import numpy as np
import threading
import time
import logging
from fastapi import FastAPI
import uvicorn
import signal
import atexit
from threading import Lock

from smartController.wandb_tracker import WandBTracker
from smartController.tiger_brain_simple import TigerBrainSimple
from smartController.attr_dict import AttrDict

# Mock classes to replace POX and real flow logging
class MockFlow:
    def __init__(self, flow_features, packet_features, element_class):
        self.flow_features = flow_features
        self.packet_features = packet_features
        self.element_class = element_class
        self.dest_ip = "1.2.3.4" # Dummy IP

    def get_flow_features(self):
        return self.flow_features

    def get_packet_features(self):
        return self.packet_features

class SimpleSmartSwitch:
    def __init__(self, brain, args):
        self.brain = brain
        self.args = args
        self.running = False
        self.dim = int(args['intrusion_detection'].get('blob_dim', 100))
        self.separability = float(args['intrusion_detection'].get('separability', 0.1))

        # Initialize cluster centers
        self.centers = {}
        knowledge = args['knowledge']
        all_classes = knowledge['Knowns'] + knowledge['G1s'] + knowledge['G2s']

        benign_center = np.zeros(self.dim)

        for i, cls_name in enumerate(all_classes):
            np.random.seed(i)
            direction = np.random.randn(self.dim)
            direction /= np.linalg.norm(direction)

            if cls_name in ['echo', 'doorlock', 'hue']: # Benign patterns
                distance = 0.5 * self.separability
            elif cls_name in knowledge['G2s']:
                distance = 1.0 * self.separability
            else:
                distance = 5.0 * self.separability

            center = benign_center + direction * distance
            self.centers[cls_name] = torch.tensor(center, dtype=torch.float32)

    def generate_batch(self):
        flows = []
        knowledge = self.args['knowledge']
        all_classes = knowledge['Knowns'] + knowledge['G1s'] + knowledge['G2s']
        selected_classes = np.random.choice(all_classes, size=min(len(all_classes), 5), replace=False)

        seq_len_flow = int(self.args['intrusion_detection']['flows_per_sample'])
        seq_len_packet = int(self.args['intrusion_detection']['packets_per_sample'])
        packet_dim = int(self.args['intrusion_detection']['packet_feat_dim'])

        for cls_name in selected_classes:
            center = self.centers[cls_name]
            num_flows = np.random.randint(1, 5)
            for _ in range(num_flows):
                noise = torch.randn(seq_len_flow, self.dim) * 0.05
                flow_feat = center.unsqueeze(0).repeat(seq_len_flow, 1) + noise
                packet_feat = torch.randn(seq_len_packet, packet_dim) * 0.01
                flows.append(MockFlow(flow_feat, packet_feat, cls_name))
        return flows

logger = logging.getLogger("SmartvilleControllerSimple")
app_thread = None
app = None
args = None
controller_brain = None
wb_tracker = None
smart_switch = None
stop_tiger_threads = True
inference_thread = None

tiger_lock = Lock()

def run_server():
    global app
    try:
        port = int(os.environ.get("SERVER_PORT"))
    except Exception as e:
        logger.error(f"Error parsing SERVER_PORT: {e}")
        return
    uvicorn.run(app, host="0.0.0.0", port=port)

def smart_check():
    global args, wb_tracker, controller_brain, smart_switch
    logger.info("Starting Simple SmartSwitch inference loop")

    while not stop_tiger_threads:
        with tiger_lock:
            if smart_switch and controller_brain:
                flows = smart_switch.generate_batch()
                node_feats = {}
                if args.get('health_monitoring'):
                    node_feats["1.2.3.4"] = {m: 0.5 for m in args['health']['probe_metrics']}

                try:
                    controller_brain.process_input(flows, node_feats)
                except Exception as e:
                    logger.error(f"Error in brain.process_input: {e}")

        time.sleep(float(args['intrusion_detection'].get('flowstats_freq_secs', 1.0)))

def shutdown_process():
    global stop_tiger_threads, inference_thread, controller_brain
    stop_tiger_threads = True
    if inference_thread:
        inference_thread.join(timeout=5)
    if controller_brain:
        controller_brain.shutdown()

def launch(**kwargs):
    global app, app_thread, openflow_connection, smart_switch
    global controller_brain, args

    app = FastAPI(title="SmartSwitch API (Simple)")

    @app.post("/initialize")
    async def initialize(init_controller_args: dict):
        global args, controller_brain, wb_tracker, smart_switch, stop_tiger_threads, inference_thread

        try:
            logger.info("Initialization command received")
            args = init_controller_args
            args['logger'] = logger

            wb_tracker = WandBTracker(args)
            controller_brain = TigerBrainSimple(args, wb_tracker=wb_tracker)
            smart_switch = SimpleSmartSwitch(controller_brain, args)

            stop_tiger_threads = False
            inference_thread = threading.Thread(target=smart_check, daemon=True)
            inference_thread.start()

            return {"msg": "Simple SmartSwitch initialized successfully", "status_code": 200}
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            return {"status_code": 500, "msg": f"Error: {e}"}

    @app.post("/stop")
    async def shutdown():
        shutdown_process()
        return {"status_code": 200, "msg": "Simple SmartSwitch stopped"}

    @app.post("/sync_wandb")
    async def sync_wandb():
        return {"status_code": 200, "msg": "Wandb sync triggered (mocked)"}

    logger.info("Simple SmartSwitch is starting...")
    if app_thread is None or not app_thread.is_alive():
        app_thread = threading.Thread(target=run_server, daemon=True)
        app_thread.start()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Mocking SERVER_PORT if not present for standalone run
    if "SERVER_PORT" not in os.environ:
        os.environ["SERVER_PORT"] = "8000"
    launch()
    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
