import sys
import os

# Support 'smartController' prefix regardless of how the directory is named
# or where the script is run from.
current_dir = os.path.dirname(os.path.abspath(__file__))
if 'smartController' not in sys.modules:
    from types import ModuleType
    smart_controller_module = ModuleType('smartController')
    smart_controller_module.__path__ = [current_dir]
    sys.modules['smartController'] = smart_controller_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import time
import threading
import os
import copy
from fastapi import FastAPI
import uvicorn
from typing import Dict, List, Any
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Smartville imports
from smartController.attr_dict import AttrDict
from smartController.tiger_agents import (
    ValueLearningAgent, DAIP_Agent, DAIA_Agent, DAIF_Agent, DAISA_Agent, PPO_Agent, A2C_Agent
)
from smartController.neural_modules import (
    MLP, MulticlassPrototypicalClassifier, ConfidenceDecoder, KernelRegressionLoss,
    DistKernelRegressor, DotProdKernelRegressor
)
from smartController.replay_buffer import RawReplayBuffer, Batch
from smartController.wandb_tracker import WandBTracker
from smartController.label_encoder import DynamicLabelEncoder
from smartController.brain_utils import (
    get_clusters, efficient_cm, efficient_os_cm, get_balanced_accuracy,
    INFERENCE, TRAINING, EVALUATION, AGENT, OS_ACC, OS_LOSS, CS_ACC, CS_LOSS,
    KR_ARI, KR_NMI, KR_LOSS, ANOMALY_BALANCE
)
from smartController.tiger_reporter import TigerReporter

def compute_uncertainty(logits, use_energy_score=True):
    if logits.shape[0] == 0:
        return torch.tensor([], device=logits.device)
    if use_energy_score:
        # energy(x) = -log( sum_c( exp(logit_c(x)) ) )
        # Higher energy = more uncertain.
        return -torch.logsumexp(logits, dim=1)
    else:
        # MSP-ratio: confidence = log(mean_max_logit / mean_non_max_logits)
        # higher confidence = lower uncertainty.
        max_logits, _ = logits.max(dim=1)
        if logits.shape[1] > 1:
            mean_non_max = (logits.sum(dim=1) - max_logits) / (logits.shape[1] - 1)
            confidence = torch.log(max_logits / (mean_non_max + 1e-10) + 1e-10)
            return -confidence
        else:
            return -max_logits

class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob, device='cpu', kr_type='dist', kr_heads=8):
        super(SimpleMLPClassifier, self).__init__()
        self.device = device
        # Use the existing MLP class from neural_modules.py
        # It's a 2-layer MLP: input_size -> intermediate -> hidden_size
        self.encoder = MLP(input_size, hidden_size, dropout_prob)

        kernel_regressor_class = DistKernelRegressor if kr_type == 'dist' else DotProdKernelRegressor
        self.kernel_regressor = kernel_regressor_class({
            'device': self.device,
            'dropout': dropout_prob,
            'n_heads': kr_heads,
            'in_features': hidden_size,
            'out_features': hidden_size
        })
        self.classifier = MulticlassPrototypicalClassifier(device=self.device)

    def forward(self, x, labels, curr_known_class_count, query_mask):
        # x shape: (batch_size, blob_feat_dim)
        # Prototypical classifier and kernel regressor expect embeddings
        embeddings = self.encoder(x)
        hiddens, predicted_kernel = self.kernel_regressor(embeddings)
        logits = self.classifier(hiddens, labels, curr_known_class_count, query_mask)
        return logits, hiddens, predicted_kernel

class SyntheticEnvironment:
    def __init__(self, kwargs: AttrDict):
        self.kwargs = kwargs
        self.intrusion_detection_kwargs = kwargs.intrusion_detection

        self.init_budget = float(self.intrusion_detection_kwargs.tiger_init_budget)
        self.min_budget = float(self.intrusion_detection_kwargs.min_budget)
        self.max_budget = float(self.intrusion_detection_kwargs.max_budget)
        self.current_budget = self.init_budget

        self.flow_rewards_dict = {key: float(value) for key, value in kwargs.rewards.items()}
        self.init_knowledge = {k: list(v) if isinstance(v, list) else v for k, v in kwargs.knowledge.items()}

        self.blob_feat_dim = int(self.intrusion_detection_kwargs.flow_feat_dim)
        self.seed = int(self.intrusion_detection_kwargs.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.max_episode_steps = int(self.intrusion_detection_kwargs.max_episode_steps)
        self.cti_price_factor = float(self.intrusion_detection_kwargs.cti_price_factor)
        self.useless_epistemic_penalty = float(self.intrusion_detection_kwargs.useless_epistemic_penalty)

        self.blob_drift_rate = float(self.intrusion_detection_kwargs.get("blob_drift_rate", 0.1))
        self.blob_std_growth = float(self.intrusion_detection_kwargs.get("blob_std_growth", 0.01))

        self.current_knowledge = copy.deepcopy(self.init_knowledge)
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_budgets = []
        self.epistemic_actions = 0

        self.class_means = {}
        self.class_stds = {}
        self.g2_drift_vectors = {}

        self.device = kwargs.device
        self.reset()

    def reset(self):
        self.current_budget = self.init_budget
        self.current_knowledge = copy.deepcopy(self.init_knowledge)
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_budgets = []
        self.epistemic_actions = 0

        all_classes = self.init_knowledge['Knowns'] + self.init_knowledge['G1s'] + self.init_knowledge['G2s']

        # Initialize means in [-5, 5]^blob_feat_dim
        for cls in all_classes:
            self.class_means[cls] = torch.rand(self.blob_feat_dim) * 10.0 - 5.0
            self.class_stds[cls] = 0.5

        # Setup drift for G2s relative to nearest G1
        g1_means = torch.stack([self.class_means[cls] for cls in self.init_knowledge['G1s']])
        for g2_cls in self.init_knowledge['G2s']:
            g2_mean = self.class_means[g2_cls]
            dists = torch.norm(g1_means - g2_mean, dim=1)
            nearest_g1_idx = torch.argmin(dists)
            nearest_g1_mean = g1_means[nearest_g1_idx]

            # Drift vector: from G1 mean to G2 mean, normalized
            drift_vec = g2_mean - nearest_g1_mean
            if torch.norm(drift_vec) > 0:
                drift_vec = drift_vec / torch.norm(drift_vec)
            else:
                drift_vec = torch.randn(self.blob_feat_dim)
                drift_vec /= torch.norm(drift_vec)

            self.g2_drift_vectors[g2_cls] = drift_vec

        self.epistemic_actions_available = 1 if len(self.current_knowledge['G2s']) > 0 else 0

    def step_environment(self):
        self.steps_done += 1
        # Apply drift and std growth
        for cls in self.init_knowledge['G2s']:
            self.class_means[cls] += self.g2_drift_vectors[cls] * self.blob_drift_rate

        for cls in self.class_stds:
            self.class_stds[cls] += self.blob_std_growth

    def sample_batch(self, batch_size: int):
        all_classes = self.init_knowledge['Knowns'] + self.init_knowledge['G1s'] + self.init_knowledge['G2s']
        samples_per_class = max(1, batch_size // len(all_classes))

        features = []
        labels = []
        zda_labels = []

        for cls in all_classes:
            mean = self.class_means[cls]
            std = self.class_stds[cls]

            cls_samples = torch.randn(samples_per_class, self.blob_feat_dim) * std + mean
            features.append(cls_samples)
            labels.extend([cls] * samples_per_class)

            is_zda = cls in self.current_knowledge['G1s'] or cls in self.current_knowledge['G2s']
            zda_labels.extend([float(is_zda)] * samples_per_class)

        return {
            'features': torch.cat(features, dim=0).to(self.device),
            'labels': labels,
            'zda_labels': torch.tensor(zda_labels, device=self.device).unsqueeze(1)
        }

    def has_episode_ended(self):
        if self.current_budget < self.min_budget \
                or self.steps_done >= self.max_episode_steps \
                or self.current_budget > self.max_budget:
            return True
        return False

    def perform_epistemic_action(self, cluster_idx: int, detected_clusters: List[str]):
        """
        cluster_idx: index i in [0, max_clusters-1]
        detected_clusters: list of ground truth class labels for each detected cluster
        """
        self.epistemic_actions += 1
        price_paid = 0
        updated_label = None

        if cluster_idx < len(detected_clusters):
            target_cls = detected_clusters[cluster_idx]
            if target_cls in self.current_knowledge['G2s']:
                self.current_knowledge['G2s'].remove(target_cls)
                self.current_knowledge['Knowns'].append(target_cls)
                price_paid = abs(self.flow_rewards_dict[target_cls] * self.cti_price_factor)
                updated_label = target_cls
                # update epistemic actions availability
                self.epistemic_actions_available = 1 if len(self.current_knowledge['G2s']) > 0 else 0
            else:
                # target_cls is G1 or already Known (Known because it was benign or already bought)
                price_paid = self.useless_epistemic_penalty
        else:
            # cluster_idx points to a padded cluster
            price_paid = self.useless_epistemic_penalty

        self.current_budget -= price_paid
        return {
            'updated_label': updated_label,
            'price_paid': price_paid
        }

    def get_cti_price(self, cls_name):
        return abs(self.flow_rewards_dict.get(cls_name, 0) * self.cti_price_factor)

    def calculate_step_reward(self, action: int, predicted_anomalous_clusters_info: List[Dict[str, Any]]):
        """
        action: 0 (block all), 1 (pass all), 2+i (CTI cluster i)
        predicted_anomalous_clusters_info: list of dicts for each detected anomalous cluster
        """
        total_reward = 0
        num_clusters = len(predicted_anomalous_clusters_info)

        if action == 0: # Block all
            if num_clusters == 0:
                self.current_budget += 0 # No change
                return 0, None
            for cluster in predicted_anomalous_clusters_info:
                # Correctly blocking an anomalous cluster: +abs(mean reward of members)
                # Incorrectly blocking a benign cluster: -abs(mean reward of members) (penalty)
                if cluster['is_actually_anomalous']:
                    total_reward += abs(cluster['mean_gt_reward'])
                else:
                    total_reward -= abs(cluster['mean_gt_reward'])
            total_reward /= num_clusters

        elif action == 1: # Pass all
            if num_clusters == 0:
                self.current_budget += 0 # No change
                return 0, None
            for cluster in predicted_anomalous_clusters_info:
                # Pass: mean reward of members (negative for attacks, positive for benign)
                total_reward += cluster['mean_gt_reward']
            total_reward /= num_clusters

        elif action >= 2: # CTI cluster i
            cluster_idx = action - 2
            # Reward calculation for epistemic action
            detected_gt_labels = [c['dominant_gt_label'] for c in predicted_anomalous_clusters_info]
            result = self.perform_epistemic_action(cluster_idx, detected_gt_labels)
            total_reward = -result['price_paid']
            # update budget is done inside perform_epistemic_action, but we return it for consistent logging
            return total_reward, result['updated_label']

        self.current_budget += total_reward
        return total_reward, None

def assembly_state_vector(cluster_infos: List[Dict[str, Any]], max_clusters: int, h_dim: int,
                          num_anom: int, num_known: int,
                          mean_ad_conf: float, mean_cs_conf: float,
                          epistemic_avail: int, budget: float, device: str):
    """
    [cluster_0_centroid (h_dim)] + [cluster_0_uncertainty (1)] + [cluster_0_dist_to_known (1)] + [cluster_0_size (1)]
    + ... (repeated for max_clusters clusters)
    + [global_proprioceptive (6)]
    """
    state_components = []

    # Per-cluster signals
    for i in range(max_clusters):
        if i < len(cluster_infos):
            c = cluster_infos[i]
            state_components.append(c['centroid']) # tensor (h_dim,)
            state_components.append(torch.tensor([c['uncertainty']], device=device))
            state_components.append(torch.tensor([c['dist_to_known']], device=device))
            state_components.append(torch.tensor([c['size_norm']], device=device))
        else:
            state_components.append(torch.zeros(h_dim, device=device))
            state_components.append(torch.zeros(1, device=device))
            state_components.append(torch.zeros(1, device=device))
            state_components.append(torch.zeros(1, device=device))

    # Global proprioceptive signals (6 scalars)
    global_signals = torch.tensor([
        float(num_known),
        float(num_anom),
        float(mean_ad_conf),
        float(mean_cs_conf),
        float(epistemic_avail),
        float(budget)
    ], device=device)

    state_components.append(global_signals)

    return torch.cat(state_components)

app = FastAPI(title="SimpleTigerServer", description="Synthetic TIGER sandbox")

# Global state
simulation_thread = None
stop_flag = threading.Event()
config = None
env = None
agent = None
classifier = None
confidence_decoder = None
kr_criterion = None
cs_criterion = None
os_criterion = None
optimizer = None
encoder = None
wb_tracker = None
reporter = None

@app.get("/")
async def root():
    return {"msg": "SimpleTigerServer is running", "status_code": 200}

@app.post("/initialize")
async def initialize(kwargs: dict):
    global config, env, agent, classifier, confidence_decoder, simulation_thread, stop_flag
    global kr_criterion, cs_criterion, os_criterion, optimizer, encoder, wb_tracker, reporter

    # 1. Parse kwargs
    args = AttrDict(kwargs)
    if 'intrusion_detection' not in args:
        return {"status_code": 400, "msg": "Missing intrusion_detection in kwargs"}

    id_args = args.intrusion_detection
    device = args.get('device', 'cpu')
    args.device = device
    h_dim = int(args.neural_modules.hidden_size)

    # 2. Build SyntheticEnvironment
    env = SyntheticEnvironment(args)

    # 3. Build inference module
    dropout = float(args.neural_modules.dropout)
    classifier = SimpleMLPClassifier(
        input_size=env.blob_feat_dim,
        hidden_size=h_dim,
        dropout_prob=dropout,
        device=device,
        kr_type=id_args.get('kr_type', 'dist'),
        kr_heads=int(id_args.get('kr_heads', 8))
    ).to(device)

    confidence_decoder = ConfidenceDecoder(device=device).to(device)

    # Criteria and Optimizer
    os_criterion = nn.BCEWithLogitsLoss().to(device)
    cs_criterion = nn.CrossEntropyLoss().to(device)
    kr_criterion = KernelRegressionLoss(
        repulsive_weigth=float(id_args.repulsive_weight),
        attractive_weigth=float(id_args.attractive_weight),
        device=device
    ).to(device)

    params = list(classifier.parameters()) + list(confidence_decoder.parameters())
    optimizer = optim.Adam(params, lr=float(id_args.learning_rate))

    encoder = DynamicLabelEncoder()
    # Pre-fit encoder with all possible classes to ensure consistent codes
    all_init_classes = env.init_knowledge['Knowns'] + env.init_knowledge['G1s'] + env.init_knowledge['G2s']
    encoder.fit(all_init_classes)

    # 4. Load pretrained weights if pretrained_inference=True
    pretrained_inference = id_args.get('pretrained_inference', False)
    if pretrained_inference:
        models_dir = id_args.get('pretrained_models_dir', 'tiger_models/')
        classifier_path = os.path.join(models_dir, f"simple_classifier_{id_args.agent}.pt")
        decoder_path = os.path.join(models_dir, f"simple_decoder_{id_args.agent}.pt")
        if os.path.exists(classifier_path):
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        if os.path.exists(decoder_path):
            confidence_decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    # 5. Build management agent
    max_clusters = int(id_args.get('max_clusters', len(env.init_knowledge['G2s'])))
    if max_clusters > 8: max_clusters = 8
    id_args.max_clusters = max_clusters

    state_size = max_clusters * (h_dim + 3) + 6
    action_size = 2 + max_clusters

    id_args.state_size = state_size
    id_args.action_size = action_size
    # batch_size is used as replay_buff_batch_size in agents
    id_args.replay_buff_batch_size = int(id_args.batch_size)
    # Compatibility with some agents that might look for replay_batch_size
    id_args.replay_batch_size = id_args.replay_buff_batch_size

    # Ensure all required agent kwargs are present
    agent_type = id_args.agent
    agent_mapping = {
        'DQN': ValueLearningAgent, 'DDQN': ValueLearningAgent,
        'DuelingDQN': ValueLearningAgent, 'DuelingDDQN': ValueLearningAgent,
        'DAI_P': DAIP_Agent, 'DAI_A': DAIA_Agent, 'DAI_SA': DAISA_Agent,
        'DAI_F': DAIF_Agent, 'PPO': PPO_Agent, 'A2C': A2C_Agent
    }

    if id_args.agency:
        agent = agent_mapping[agent_type](args)
    else:
        agent = None

    # WandB
    if args.wandb.wb_tracking:
        wb_tracker = WandBTracker(args)
        reporter = TigerReporter(kwargs, wb_tracker.wb_run, encoder, args.logger, id_args.seed)
    else:
        wb_tracker = None
        reporter = None

    # 6. Start simulation loop
    # If pretrained_inference=True and weights are found, skip the warmup entirely.
    if pretrained_inference:
        models_dir = id_args.get('pretrained_models_dir', 'tiger_models/')
        classifier_path = os.path.join(models_dir, f"simple_classifier_{id_args.agent}.pt")
        if os.path.exists(classifier_path):
            # Skip warmup
            id_args.intelligence_episode_steps = 0

    stop_flag.clear()
    simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
    simulation_thread.start()

    return {"msg": "SimpleTigerServer initialized", "status_code": 200}

@app.post("/stop")
async def stop():
    global stop_flag, simulation_thread, wb_tracker
    stop_flag.set()
    if simulation_thread:
        simulation_thread.join(timeout=5)
    if wb_tracker:
        wb_tracker.shutdown()
    return {"msg": "SimpleTigerServer stopped", "status_code": 200}

def simulation_loop():
    global stop_flag, env, agent, classifier, confidence_decoder, optimizer
    global kr_criterion, cs_criterion, os_criterion, encoder, wb_tracker, reporter

    id_args = env.intrusion_detection_kwargs
    step_freq = float(id_args.get('inference_freq_secs', 0.1))
    intelligence_warmup = int(id_args.get('intelligence_episode_steps', 0))
    report_freq = int(id_args.get('report_step_freq', 10))
    use_energy = id_args.get('use_energy_score', True)
    h_dim = int(env.kwargs.neural_modules.hidden_size)
    max_clusters = id_args.max_clusters
    save_models = id_args.get('save_models', False)
    models_dir = id_args.get('pretrained_models_dir', 'tiger_models/')

    best_cs_acc = 0
    best_ad_acc = 0
    best_kr_ari = 0
    max_observed_dist = 1.0

    epistemic_actions_count = 0
    epistemic_costs_sum = 0
    accepted_rewards_sum = 0
    blocked_rewards_sum = 0
    num_episodes_steps = 0

    global_step = 0

    while not stop_flag.is_set():
        # 1. Sample batch
        batch_data = env.sample_batch(env.kwargs.intrusion_detection.batch_size)
        features = batch_data['features']
        gt_labels = batch_data['labels']
        zda_labels = batch_data['zda_labels']

        encoded_labels = encoder.transform(gt_labels).to(env.device)

        # 2. Run inference
        # SimpleMLPClassifier.forward(x, labels, curr_known_class_count, query_mask)
        # Sandbox uses full batch for now (query_mask all True) or k-shot logic?
        # Let's follow TigerBrain: query_mask separates support and query.
        # But Gaussian blobs don't strictly need k-shot.
        # However, PrototypicalClassifier NEEDS support to build centroids.

        k_shot = int(id_args.k_shot)
        batch_size = int(id_args.batch_size)
        num_classes = len(set(gt_labels))
        # TigerBrain logic for query mask
        N = features.shape[0] // batch_size
        M = batch_size
        query_mask = torch.ones(features.shape[0], device=env.device).to(torch.bool)
        # This is tricky because sample_batch gives balanced classes.
        # Let's just use first k_shot of each class as support.
        samples_per_class = features.shape[0] // num_classes
        for i in range(num_classes):
            query_mask[i*samples_per_class : i*samples_per_class + k_shot] = False

        known_classes = env.current_knowledge['Knowns']
        num_known = len(known_classes)

        logits, hiddens, predicted_kernel = classifier(features, encoded_labels.unsqueeze(1), num_known, query_mask)

        # Subset everything to query part to match logits/clusters
        query_features = features[query_mask]
        query_gt_labels = [gt_labels[i] for i, m in enumerate(query_mask) if m]
        query_zda_labels = zda_labels[query_mask]
        query_hiddens = hiddens[query_mask]
        query_kernel = predicted_kernel[query_mask][:, query_mask]

        # 3. Detect anomalous clusters
        # predicted_clusters will be [N_query]
        predicted_clusters = get_clusters(query_kernel)
        num_predicted_clusters = predicted_clusters.max().item() + 1

        # 4. Compute per-cluster uncertainty signals
        # Logits are [N_query, num_known]
        # AD predictions from confidence decoder
        zda_preds_logits = confidence_decoder(logits) # [N_query, 1]

        # Cluster-level info assembly
        cluster_infos = []

        # known prototypes (centroids)
        # classifier.classifier.get_centroids(...)
        support_hiddens = classifier.encoder(features[~query_mask])
        support_labels = encoded_labels[~query_mask]
        # one-hot labels for support
        support_oh = torch.zeros(support_labels.size(0), num_known, device=env.device)
        # only labels that are in Knowns
        # This is a bit complex because encoded_labels includes G1/G2
        # TigerBrain filters them out or handles them.

        # simplified: get centroids of all classes for distance calculations
        all_centroids = []
        for i in range(num_known):
            cls_name = known_classes[i]
            cls_code = encoder.transform([cls_name])[0]
            mask = support_labels == cls_code
            if mask.any():
                all_centroids.append(support_hiddens[mask].mean(0))
        if all_centroids:
            known_prototypes = torch.stack(all_centroids)
        else:
            known_prototypes = None

        raw_dists = []
        for c_idx in range(int(num_predicted_clusters)):
            c_mask = predicted_clusters == c_idx
            if not c_mask.any(): continue

            c_hiddens = query_hiddens[c_mask]
            c_centroid = c_hiddens.mean(0)
            c_logits = logits[c_mask]

            uncertainty = compute_uncertainty(c_logits, use_energy).mean().item()

            # Distance to nearest known prototype
            dist_to_known = 0.0
            if known_prototypes is not None:
                dists = torch.norm(known_prototypes - c_centroid, dim=1)
                dist_to_known = dists.min().item()
            raw_dists.append(dist_to_known)

            size_norm = c_mask.float().sum().item() / query_mask.float().sum().item()

            # Ground truth for reward
            c_gt_labels = [query_gt_labels[i] for i, m in enumerate(c_mask) if m]
            dominant_gt_label = max(set(c_gt_labels), key=c_gt_labels.count)
            is_actually_anomalous = dominant_gt_label in env.current_knowledge['G1s'] or \
                                    dominant_gt_label in env.current_knowledge['G2s']
            mean_gt_reward = np.mean([env.flow_rewards_dict[l] for l in c_gt_labels])

            cluster_infos.append({
                'centroid': c_centroid,
                'uncertainty': uncertainty,
                'dist_to_known': dist_to_known, # placeholder, normalized below
                'size_norm': size_norm,
                'dominant_gt_label': dominant_gt_label,
                'is_actually_anomalous': is_actually_anomalous,
                'mean_gt_reward': mean_gt_reward
            })

        # Distance normalization across the whole step
        if raw_dists:
            step_max_dist = max(raw_dists)
            if step_max_dist > max_observed_dist:
                max_observed_dist = step_max_dist
            for i, info in enumerate(cluster_infos):
                info['dist_to_known'] /= max_observed_dist

        # Sort clusters by uncertainty (descending) to give agent priority
        cluster_infos.sort(key=lambda x: x['uncertainty'], reverse=True)

        # 5. Assemble state vector
        mean_ad_conf = zda_preds_logits.mean().item()
        # Mean CS confidence (max logit mean?)
        mean_cs_conf = logits.max(dim=1)[0].mean().item()

        state = assembly_state_vector(
            cluster_infos, max_clusters, h_dim,
            num_anom=int((zda_preds_logits > 0.5).sum().item()),
            num_known=num_known,
            mean_ad_conf=mean_ad_conf,
            mean_cs_conf=mean_cs_conf,
            epistemic_avail=env.epistemic_actions_available,
            budget=env.current_budget,
            device=env.device
        )

        # 6. Step decision (Warmup / Agency)
        action = None
        reward = 0
        if id_args.agency and global_step >= intelligence_warmup:
            action = agent.act(state)
            reward, updated_label = env.calculate_step_reward(action, cluster_infos)

            # Cumulative metrics for reporting
            num_episodes_steps += 1
            if action >= 2:
                epistemic_actions_count += 1
                epistemic_costs_sum += reward
            elif action == 0:
                accepted_rewards_sum += reward
            elif action == 1:
                blocked_rewards_sum += reward

            # 7. Train management agent
            # Update state after action (budget changed)
            next_state = assembly_state_vector(
                cluster_infos, max_clusters, h_dim,
                num_anom=int((zda_preds_logits > 0.5).sum().item()),
                num_known=len(env.current_knowledge['Knowns']),
                mean_ad_conf=mean_ad_conf,
                mean_cs_conf=mean_cs_conf,
                epistemic_avail=env.epistemic_actions_available,
                budget=env.current_budget,
                device=env.device
            )

            done = env.has_episode_ended()
            agent.remember(state, action, reward, next_state, done, global_step)
            agent.replay(global_step)

            if done:
                env.reset()

        # Always step the environment (drift, growth, step counter)
        env.step_environment()
        if not id_args.agency or global_step < intelligence_warmup:
            if env.has_episode_ended():
                env.reset()

        # 8. Train inference module
        # Supervised loss for AD and CS
        one_hot_labels = torch.zeros(query_zda_labels.size(0), num_known, device=env.device)
        # only for samples whose gt_label is in Knowns
        for i, l in enumerate(query_gt_labels):
            if l in known_classes:
                idx = known_classes.index(l)
                one_hot_labels[i, idx] = 1.0

        # CS loss (only on known samples)
        known_mask = one_hot_labels.sum(1) > 0
        if known_mask.any():
            cs_loss = cs_criterion(logits[known_mask], one_hot_labels[known_mask].argmax(1))
        else:
            cs_loss = torch.tensor(0.0, device=env.device)

        # AD loss (all query samples)
        # zda_labels are 0 or 1
        ad_loss = os_criterion(zda_preds_logits, query_zda_labels.float())

        # KR loss
        semantic_kernel = torch.zeros(len(query_gt_labels), len(query_gt_labels), device=env.device)
        for i in range(len(query_gt_labels)):
            for j in range(len(query_gt_labels)):
                if query_gt_labels[i] == query_gt_labels[j]:
                    semantic_kernel[i, j] = 1.0
        kr_loss = kr_criterion(semantic_kernel, predicted_kernel)

        total_inf_loss = cs_loss + ad_loss
        if id_args.clustering_loss_backprop:
            total_inf_loss += kr_loss

        optimizer.zero_grad()
        total_inf_loss.backward()
        optimizer.step()

        # 9. Log metrics and handle saving
        if global_step % report_freq == 0:

            plot_freq = int(id_args.get('plot_step_freq', 100))
            if reporter and global_step % plot_freq == 0:
                # TigerReporter.report(preds, hiddens, labels, predicted_clusters, query_mask, phase, ...)
                # In simple_tiger_server, hiddens and labels are already for the whole batch, query_mask applied later.
                # reporter expects labels as tensor [N, 1]
                labels_tensor = encoded_labels.unsqueeze(1)
                plots = reporter.report(
                    preds=logits,
                    hiddens=hiddens,
                    labels=labels_tensor,
                    predicted_clusters=predicted_clusters,
                    query_mask=query_mask,
                    phase=TRAINING # Sandbox uses TRAINING phase for simplicity
                )
                wb_tracker.wb_run.log(plots, step=global_step)

            # compute metrics for reporting
            with torch.no_grad():
                # AD Acc (Balanced)
                ad_preds = (zda_preds_logits > 0.5).float()
                # simplified balanced accuracy
                tp = ((ad_preds == 1) & (query_zda_labels == 1)).float().sum()
                tn = ((ad_preds == 0) & (query_zda_labels == 0)).float().sum()
                fp = ((ad_preds == 1) & (query_zda_labels == 0)).float().sum()
                fn = ((ad_preds == 0) & (query_zda_labels == 1)).float().sum()
                tpr = tp / (tp + fn + 1e-10)
                tnr = tn / (tn + fp + 1e-10)
                ad_acc = (tpr + tnr) / 2

                # CS Acc
                if known_mask.any():
                    cs_preds = logits[known_mask].argmax(1)
                    cs_acc = (cs_preds == one_hot_labels[known_mask].argmax(1)).float().mean().item()
                else:
                    cs_acc = 1.0

                # KR Metrics
                from sklearn.metrics import adjusted_rand_score
                kr_ari = adjusted_rand_score(np.array(query_gt_labels), predicted_clusters.cpu().numpy())
                kr_nmi = normalized_mutual_info_score(np.array(query_gt_labels), predicted_clusters.cpu().numpy())

            if save_models:
                if cs_acc > best_cs_acc:
                    best_cs_acc = cs_acc
                    torch.save(classifier.state_dict(), os.path.join(models_dir, f"simple_classifier_{id_args.agent}.pt"))
                if ad_acc > best_ad_acc:
                    best_ad_acc = ad_acc
                    torch.save(confidence_decoder.state_dict(), os.path.join(models_dir, f"simple_decoder_{id_args.agent}.pt"))

            if wb_tracker:
                metrics = {
                    'AGENT_budget': env.current_budget,
                    'AGENT_reward': reward,
                    'clustering_reward': reward,
                    'Epistemic Actions taken': epistemic_actions_count / (num_episodes_steps + 1e-10),
                    'epistemic_costs': epistemic_costs_sum / (num_episodes_steps + 1e-10),
                    'rewards_per_accepted_clusters': accepted_rewards_sum / (num_episodes_steps + 1e-10),
                    'rewards_per_blocked_clusters': blocked_rewards_sum / (num_episodes_steps + 1e-10),
                    'Training/Loss': cs_loss.item(),
                    'Training/Acc': cs_acc,
                    'Training/AD Loss': ad_loss.item(),
                    'Training/AD Acc': ad_acc.item(),
                    'Training/KR_LOSS': kr_loss.item(),
                    'Training/KR_ARI': kr_ari,
                    'Training/KR_NMI': kr_nmi,
                    'Mean EVAL AD ACC': ad_acc.item(),
                    'Mean EVAL CS ACC': cs_acc,
                    'Mean EVAL KR PREC': kr_ari,
                    'sandbox/gaussian_std': np.mean(list(env.class_stds.values())),
                    'sandbox/n_clusters_detected': len(cluster_infos)
                }
                for i, c in enumerate(cluster_infos[:max_clusters]):
                    metrics[f'sandbox/cluster_{i}_energy'] = c['uncertainty']
                    metrics[f'sandbox/cluster_{i}_dist_to_known'] = c['dist_to_known']

                wb_tracker.wb_run.log(metrics, step=global_step)

        global_step += 1
        time.sleep(step_freq)

if __name__ == "__main__":
    import sys
    # set dummy logger for SyntheticEnvironment if needed
    uvicorn.run(app, host="0.0.0.0", port=8001)
