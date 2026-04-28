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

# We'll use the real components where possible, or their simplified versions
from smartController.wandb_tracker import WandBTracker

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
        self.thread = None
        self.lock = Lock()

        self.dim = int(args['intrusion_detection'].get('blob_dim', 100)) # Default to 100 as requested
        self.separability = float(args['intrusion_detection'].get('separability', 0.1)) # Low separability makes it hard
        self.batch_size = int(args['intrusion_detection'].get('batch_size', 32))

        # Initialize cluster centers
        self.centers = {}
        knowledge = args['knowledge']
        all_classes = knowledge['Knowns'] + knowledge['G1s'] + knowledge['G2s']

        # The user's desired logic:
        # unknown anomalies (G2s) that are very hard to distinguish from normal traffic
        # without the specific CTI-label-driven training should be perfect.

        # Let's define a "Benign" center
        benign_center = np.zeros(self.dim)

        for i, cls_name in enumerate(all_classes):
            np.random.seed(i)
            # Random direction
            direction = np.random.randn(self.dim)
            direction /= np.linalg.norm(direction)

            if cls_name in ['echo', 'doorlock', 'hue']: # Benign patterns from user
                distance = 0.5 * self.separability
            elif cls_name in knowledge['G2s']:
                # G2s are close to benign area
                distance = 1.0 * self.separability
            else:
                # Knowns and G1s are further away
                distance = 5.0 * self.separability

            center = benign_center + direction * distance
            self.centers[cls_name] = torch.tensor(center, dtype=torch.float32)

    def generate_batch(self):
        flows = []
        knowledge = self.args['knowledge']
        all_classes = knowledge['Knowns'] + knowledge['G1s'] + knowledge['G2s']

        # Sample 5 random classes for this "window"
        selected_classes = np.random.choice(all_classes, size=min(len(all_classes), 5), replace=False)

        seq_len_flow = int(self.args['intrusion_detection']['flows_per_sample'])
        seq_len_packet = int(self.args['intrusion_detection']['packets_per_sample'])
        packet_dim = int(self.args['intrusion_detection']['packet_feat_dim'])

        for cls_name in selected_classes:
            center = self.centers[cls_name]
            num_flows = np.random.randint(1, 5)
            for _ in range(num_flows):
                # Flow features (seq_len, dim)
                noise = torch.randn(seq_len_flow, self.dim) * 0.05
                flow_feat = center.unsqueeze(0).repeat(seq_len_flow, 1) + noise

                # Packet features (dummy)
                packet_feat = torch.randn(seq_len_packet, packet_dim) * 0.01

                flows.append(MockFlow(flow_feat, packet_feat, cls_name))

        return flows

    def loop(self):
        self.args['logger'].info("Starting SimpleSmartSwitch loop")
        while self.running:
            flows = self.generate_batch()

            # Mock node feats
            node_feats = {}
            if self.args.get('health_monitoring'):
                node_feats["1.2.3.4"] = {m: 0.5 for m in self.args['health']['probe_metrics']}

            try:
                self.brain.process_input(flows, node_feats)
            except Exception as e:
                self.args['logger'].error(f"Error in brain.process_input: {e}")

            time.sleep(float(self.args['intrusion_detection'].get('flowstats_freq_secs', 1.0)))

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

app = FastAPI(title="Simple SmartSwitch API")
controller_brain = None
wb_tracker = None
smart_switch = None
args = None

@app.get("/")
async def root():
    return {"msg": "Hello World from the Simple SmartSwitch!"}

@app.post("/initialize")
async def initialize(init_controller_args: dict):
    global controller_brain, wb_tracker, smart_switch, args

    try:
        args = init_controller_args
        # Setup logger
        logger = logging.getLogger("SimpleSmartSwitch")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        args['logger'] = logger

        logger.info("Initializing Simple SmartSwitch...")

        wb_tracker = WandBTracker(args)

        from smartController.tiger_brain_simple import TigerBrainSimple
        brain_class = TigerBrainSimple

        controller_brain = brain_class(args, wb_tracker=wb_tracker)

        smart_switch = SimpleSmartSwitch(controller_brain, args)
        smart_switch.start()

        logger.info("Initialization complete.")
        return {"msg": "Simple SmartSwitch initialized successfully", "status_code": 200}
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        if 'logger' in locals():
            logger.error(f"Initialization failed: {error_msg}")
        return {"status_code": 500, "msg": f"Error: {e}", "trace": error_msg}

@app.post("/stop")
async def shutdown():
    global smart_switch, controller_brain
    if smart_switch:
        smart_switch.stop()
    if controller_brain:
        controller_brain.shutdown()
    return {"status_code": 200, "msg": "Simple SmartSwitch stopped"}

@app.post("/sync_wandb")
async def sync_wandb():
    # Keep consistent with real server
    return {"status_code": 200, "msg": "Wandb sync triggered (mocked)"}

def cleanup():
    if smart_switch:
        smart_switch.stop()
    if controller_brain:
        controller_brain.shutdown()

atexit.register(cleanup)

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
