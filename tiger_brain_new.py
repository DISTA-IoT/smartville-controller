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

import os
import torch
import torch.optim as optim
import torch.nn as nn
import threading
import random
import time
import copy
import queue
from functools import wraps
from contextlib import contextmanager

from smartController.replay_buffer import RawReplayBuffer, Batch
from smartController.wandb_tracker import WandBTracker
from smartController.tiger_environment_new import NewTigerEnvironment
from smartController.tiger_agents import (
    ValueLearningAgent, DAIP_Agent, DAIA_Agent,
    DAIF_Agent, DAISA_Agent
)
from smartController.attr_dict import AttrDict

# Local imports
from smartController.label_encoder import DynamicLabelEncoder
from smartController.brain_utils import (
    efficient_cm, efficient_os_cm, get_balanced_accuracy,
    get_clusters, get_metrics_tensor,
    TRAINING, INFERENCE, EVALUATION, AGENT, OS_ACC, OS_LOSS, CS_ACC, CS_LOSS,
    KR_ARI, KR_NMI, KR_LOSS, ANOMALY_BALANCE,
    CONFIDENCE_DECODER_CLASS_NAME, KERNEL_REGRESSION_LOSS_CLASS_NAME,
    ONE_STREAM_MULTICLASS_FLOW_CLASSIFIER_CLASS_NAME,
    TWO_STREAM_MULTICLASS_FLOW_CLASSIFIER_CLASS_NAME,
    THREE_STREAM_MULTICLASS_FLOW_CLASSIFIER_CLASS_NAME
)
from smartController.tiger_reporter import TigerReporter

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def thread_safe(method):
    """
    Decorator to ensure thread-safe access to a method using the instance's _lock.
    """
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        with self._lock:
            return method(self, *method_args, **method_kwargs)
    return _impl


def epistemic_thread_safe(method):
    """
    Decorator to ensure thread-safe access to a method using the instance's _epistemic_lock.
    """
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        with self._epistemic_lock:
            return method(self, *method_args, **method_kwargs)
    return _impl


class TigerBrain:
    """
    Main class for the TigerBrain intelligence module.
    Responsible for flow classification, anomaly detection, clustering, and mitigation.
    """

    def __init__(self, kwargs, wb_tracker=None):
        """
        Initializes the TigerBrain module with provided configuration and optional WandB tracker.
        """
        args = AttrDict(kwargs)
        
        # Concurrency and logging
        self._lock = threading.Lock()
        self._epistemic_lock = threading.Lock()
        self.logger_instance = kwargs['logger']

        # Configuration parameters
        self.kwargs = kwargs
        self.intrusion_detection_kwargs = kwargs['intrusion_detection']
        self.eval = args.intrusion_detection.eval
        self.use_packet_feats = args.use_packet_feats
        self.use_node_feats = args.node_features
        self.flow_feat_dim = int(args.intrusion_detection.flow_feat_dim)
        self.packet_feat_dim = int(args.intrusion_detection.packet_feat_dim)
        self.hidden_size = int(args.neural_modules.hidden_size)
        self.multi_class = args.intrusion_detection.multi_class
        self.kernel_regression = args.intrusion_detection.kernel_regression
        self.device = args.device

        # Training and Evaluation parameters
        self.seed = int(args.intrusion_detection.seed)
        random.seed(self.seed)
        self.k_shot = int(args.intrusion_detection.k_shot)
        self.batch_size = int(args.intrusion_detection.batch_size)
        self.report_step_freq = int(args.intrusion_detection.report_step_freq)
        self.plot_step_freq = int(args.intrusion_detection.plot_step_freq)
        self.online_eval_step_freq = int(args.intrusion_detection.online_eval_step_freq)
        self.use_neural_AD = args.intrusion_detection.use_neural_AD
        self.use_neural_KR = args.intrusion_detection.use_neural_KR
        self.use_neural_CS = args.intrusion_detection.use_neural_CS
        self.online_evaluation = args.intrusion_detection.online_evaluation
        self.wrong_inference_penalisation = args.intrusion_detection.wrong_inference_penalisation
        self.bad_classif_cost_factor = float(args.intrusion_detection.bad_classif_cost_factor)
        self.bad_clustering_cost_factor = float(args.intrusion_detection.bad_clustering_cost_factor)
        self.online_eval_rounds = int(args.intrusion_detection.online_evaluation_rounds)
        self.load_pretrained_inference_module = args.intrusion_detection.pretrained_inference
        self.clustering_loss_backprop = args.intrusion_detection.clustering_loss_backprop
        self.attractive_weight = float(args.intrusion_detection.attractive_weight)
        self.repulsive_weight = float(args.intrusion_detection.repulsive_weight)
        self.learning_rate = float(args.intrusion_detection.learning_rate)
        self.replay_buffer_max_capacity = int(args.intrusion_detection.replay_buffer_max_capacity)
        self.pretrained_models_dir = args.intrusion_detection.pretrained_models_dir
        self.update_target_freq = int(args.intrusion_detection.update_target_freq)
        

        # Environment and Networking
        self.container_ips = args.container_ips
        self.ips_containers = args.ips_containers
        self.traffic_dict = args.traffic_dict
        self.episode_count = -1
        self.env = NewTigerEnvironment(args)

        # WandB Tracking
        self.wbt = args.wandb.wb_tracking
        self.wb_tracker = wb_tracker
        self.wb_run = None
        if self.wbt:
            self.wb_run = self.wb_tracker.wb_run
        args.intrusion_detection.wbl = self.wb_run

        # Initialize internal components
        self.encoder = DynamicLabelEncoder()
        self.reporter = TigerReporter(kwargs, self.wb_run, self.encoder, self.logger_instance, self.seed)

        self.init_agents(args)
        self.init_intelligence()

        # Agency and Persistency
        self.agency = args.intrusion_detection.agency
        self.save_models_flag = args.intrusion_detection.save_models

        # Performance profiling
        self.profiling_stats = {}
        self._rewards_lookup = None
        self._g1_codes_tensor = None
        self._g2_codes_tensor = None

    @contextmanager
    def profile(self, name):
        """Elegant context manager for timing code blocks. Appends to lists for mean calculation."""
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()
            
        start = time.perf_counter()
        
        yield  # The wrapped code runs here
        
        if use_cuda:
            torch.cuda.synchronize()
            
        elapsed_ms = (time.perf_counter() - start) * 1000
        key = f"ml_profiling_millis/{name}"
        
        # Initialize the list if it doesn't exist, then append the new time
        if key not in self.profiling_stats:
            self.profiling_stats[key] = []
        self.profiling_stats[key].append(elapsed_ms)

    def shutdown(self):
        """
        Shuts down monitoring threads and WandB tracker.
        """
        if hasattr(self, '_stop_monitoring'):
            self._stop_monitoring.set()
            for t in self._monitoring_threads:
                if t.is_alive():
                    t.join(timeout=2.0)

        if self.wb_tracker:
            self.wb_tracker.shutdown()
            
    def init_intelligence(self):
        """
        Initializes metrics, confusion matrices, and the environment.
        """
        self.eval_queue = queue.Queue()
        self.current_known_classes_count = 0
        self.current_test_known_classes_count = 0
        self.batch_processing_allowed = False
        self.best_cs_accuracy = 0
        self.best_AD_accuracy = 0
        self.best_KR_accuracy = 0
        self.reset_train_cms()
        self.reset_test_cms()
        self.replay_buffers = {}
        self.reset_environment()

    @epistemic_thread_safe
    def reset_environment(self):
        """
        Resets the environment and initializes inference modules.
        """
        self.env.reset()    
        self.init_inference_neural_modules()
        self.episode_count += 1
        
    def init_agents(self, args):
        """
        Initializes the mitigation agent (e.g., DQN, DAI variants).
        """
        self.state_space_dim = self.hidden_size
        if self.use_node_feats:
            self.state_space_dim += self.hidden_size
        if self.use_packet_feats:
            self.state_space_dim += self.hidden_size

        # State space components:
        # 0. centroid of collective anomaly (an all-zeros centroid for known traffic)
        # 1. number of anomalies inferred in the batch
        # 2. mean confidence of anomaly inference.  
        # 3. number of known classes inferred in the batch
        # 4. mean confidence of known class classification
        # 5. available CTI options (boolean flag)
        # 6. current system budget
        self.state_space_dim += 6

        agent_mapping = {
            'DQN': ValueLearningAgent,
            'DDQN': ValueLearningAgent,
            'DuelingDQN': ValueLearningAgent,
            'DuelingDDQN': ValueLearningAgent,
            'DAI_P': DAIP_Agent,
            'DAI_A': DAIA_Agent,
            'DAI_SA': DAISA_Agent,
            'DAI_F': DAIF_Agent
        }
        
        agent_type = args.intrusion_detection.agent
        if agent_type not in agent_mapping:
            raise ValueError(f'Unknown agent type: {agent_type}')

        agent_class = agent_mapping[agent_type]
        args.intrusion_detection.action_size = 3  # block, pass or TCI acquisition
        args.intrusion_detection.state_size = self.state_space_dim
        
        self.mitigation_agent = agent_class(args)

    def add_replay_buffer(self, class_name):
        """
        Adds a new replay buffer for a newly discovered class.
        """
        self.batch_processing_allowed = False
        
        self.replay_buffers[self.current_known_classes_count-1] = RawReplayBuffer(
            capacity=self.replay_buffer_max_capacity,
            seed=self.seed)
        self.logger_instance.info(f'Replay buffer with code: {self.current_known_classes_count-1} for class: {class_name} was added')
    
    @thread_safe
    def add_class_to_knowledge_base(self, new_class):
        """
        Updates the knowledge base with a new class.
        """
        self.current_known_classes_count += 1
        self.add_replay_buffer(new_class)
        self.reset_train_cms()
        self.reset_test_cms()

    def reset_train_cms(self):
        """Resets training confusion matrices."""
        self.training_cs_cm = torch.zeros(
            [max(1, self.current_known_classes_count), max(1, self.current_known_classes_count)],
            device=self.device)
        self.training_os_cm = torch.zeros(
            size=(2, 2),
            device=self.device)
        
    def reset_test_cms(self):
        """Resets evaluation confusion matrices."""
        self.eval_cs_cm = torch.zeros(
            [max(1, self.current_known_classes_count), max(1, self.current_known_classes_count)],
            device=self.device)
        self.eval_os_cm = torch.zeros(
            size=(2, 2),
            device=self.device)

    def load_models_from_source(self):
        """
        Safely load models from source code string provided in kwargs.
        """
        try:
            namespace = {}
            exec(self.kwargs['models'], namespace)
            
            model_classes = {}
            for name, obj in namespace.items():
                if (isinstance(obj, type) and 
                    hasattr(obj, '__bases__') and 
                    any('Module' in base.__name__ for base in obj.__bases__ if hasattr(base, '__name__'))):
                    model_classes[name] = obj
            
            return model_classes
            
        except Exception as e:
            self.logger_instance.error(f"Error loading models from source: {e}")
            return {}
    
    def init_inference_neural_modules(self):
        """
        Initializes neural modules (classifier, confidence decoder, criterion).
        Loads pre-trained weights if specified.
        """
        torch.manual_seed(self.seed)
        model_classes = self.load_models_from_source()

        if CONFIDENCE_DECODER_CLASS_NAME not in model_classes:
                raise RuntimeError(f"A class named {CONFIDENCE_DECODER_CLASS_NAME} was not found in your models.py file")
        
        self.confidence_decoder = model_classes[CONFIDENCE_DECODER_CLASS_NAME](device=self.device)
        
        self.os_criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.cs_criterion = nn.SmoothL1Loss(reduction='mean').to(self.device) if self.kwargs['intrusion_detection'].get('use_huber_cs') else nn.CrossEntropyLoss().to(self.device)
        
        if KERNEL_REGRESSION_LOSS_CLASS_NAME not in model_classes:
            raise RuntimeError(f"A class named {KERNEL_REGRESSION_LOSS_CLASS_NAME} was not found in your models.py file")
        
        try:
            self.kr_criterion = model_classes[KERNEL_REGRESSION_LOSS_CLASS_NAME](
                repulsive_weigth=self.repulsive_weight, 
                attractive_weigth=self.attractive_weight
                ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Error initializing {KERNEL_REGRESSION_LOSS_CLASS_NAME} criterion: {e}")
        
        # Determine the correct classifier class based on features used
        if self.use_packet_feats and self.use_node_feats:
            target_class = THREE_STREAM_MULTICLASS_FLOW_CLASSIFIER_CLASS_NAME
        elif self.use_packet_feats or self.use_node_feats:
            target_class = TWO_STREAM_MULTICLASS_FLOW_CLASSIFIER_CLASS_NAME
        else:
            target_class = ONE_STREAM_MULTICLASS_FLOW_CLASSIFIER_CLASS_NAME

        if target_class not in model_classes:
            raise RuntimeError(f"Required classifier {target_class} not found in models.py")

        try:
            self.classifier = model_classes[target_class](kwargs=self.kwargs['neural_modules'])
        except Exception as e:
            raise RuntimeError(f"Error initializing {target_class} classifier: {e}")

        self.check_pretrained()

        params_for_optimizer = \
            list(self.confidence_decoder.parameters()) + \
            list(self.classifier.parameters())

        self.classifier.to(self.device)
        self.optimizer = optim.Adam(params_for_optimizer, lr=self.learning_rate)

        if self.eval:
            self.classifier.eval()
            self.confidence_decoder.eval()
            self.logger_instance.info(f"Using MODULES in EVAL mode!")                

    def check_pretrained(self):
        """
        Loads pre-trained weights for the classifier and confidence decoder.
        """
        # Build path based on feature configuration
        feats_str = ""
        if self.use_packet_feats: feats_str += "_packet"
        if self.use_node_feats: feats_str += "_node"

        self.classifier_path = f"{self.pretrained_models_dir}multiclass_flow{feats_str}_classifier_pretrained_h{self.hidden_size}"
        self.confidence_decoder_path = f"{self.pretrained_models_dir}flow{feats_str}_confidence_decoder_pretrained_h{self.hidden_size}"

        if self.load_pretrained_inference_module:
            if os.path.exists(self.pretrained_models_dir):
                if os.path.exists(self.classifier_path+'.pt'):
                    self.classifier.load_state_dict(torch.load(self.classifier_path+'.pt', weights_only=True))
                    self.logger_instance.info(f"Pre-trained weights loaded from {self.classifier_path}.pt")
                    
                if self.multi_class and os.path.exists(self.confidence_decoder_path+'.pt'):
                    self.confidence_decoder.load_state_dict(torch.load(self.confidence_decoder_path+'.pt', weights_only=True))
                    self.logger_instance.info(f"Pre-trained weights loaded from {self.confidence_decoder_path}.pt")
            else:
                self.logger_instance.info(f"Pre-trained folder not found at {self.pretrained_models_dir}.")

    def infer(self, classifier, batch, known_classes_count, query_mask):
        """
        Performs forward inference on the given classifier.
        """
        try:
            inputs = [batch.flow_features]
            if self.use_packet_feats:
                inputs.append(batch.packet_features)
            if self.use_node_feats:
                inputs.append(batch.node_features)

            inputs.extend([batch.class_labels, known_classes_count, query_mask])

            logits, hiddens, predicted_kernel = classifier(*inputs)
            return logits, hiddens, predicted_kernel
        except Exception as e:
            self.logger_instance.error(f'Inference error: {e}')
            raise RuntimeError(f'Inference error: {e}')

    def push_to_replay_buffers(self, flow_input_batch, packet_input_batch, node_feat_input_batch, batch_labels):
        """
        Saves input samples into their respective class replay buffers.
        """
        unique_labels = torch.unique(batch_labels)
        buffers = self.replay_buffers

        for label in unique_labels:
            mask = batch_labels == label
            masked_flow = flow_input_batch[mask]
            masked_packet = packet_input_batch[mask] if self.use_packet_feats else None
            masked_node = node_feat_input_batch[mask] if self.use_node_feats else None
            masked_labels = batch_labels[mask]
            
            for sample_idx in range(masked_flow.shape[0]):
                try:
                    buffers[label.item()].push(
                        flow_state=masked_flow[sample_idx].unsqueeze(0),
                        packet_state=(masked_packet[sample_idx].unsqueeze(0) if self.use_packet_feats else None),
                        node_state=(masked_node[sample_idx].unsqueeze(0) if self.use_node_feats else None),
                        label=masked_labels[sample_idx].unsqueeze(0))
                except Exception as e:
                    raise RuntimeError(f'Error while pushing sample {sample_idx} with label {label} to replay buffer {label}: {e}')

        if not self.batch_processing_allowed:
            buff_lengths = [(cl, len(buffers[idx])) for cl, idx in self.encoder.get_mapping().items()]
      
            if len(buff_lengths) > 1:
                has_enough_samples = all(bl > self.batch_size for (_, bl) in buff_lengths)
                has_known_class = any(cn in self.env.current_knowledge['Knowns'] for (cn, _) in buff_lengths)
                self.batch_processing_allowed = has_enough_samples and has_known_class

            self.logger_instance.info(f'Buffer lengths: {buff_lengths}')

    def merge_batches(self, left_batch, right_batch):
        """
        Merges two batches together.
        """
        flow_features = torch.vstack([left_batch.flow_features, right_batch.flow_features])
        packet_features = (torch.vstack([left_batch.packet_features, right_batch.packet_features]) if self.use_packet_feats else None)
        node_features = (torch.vstack([left_batch.node_features, right_batch.node_features]) if self.use_node_feats else None)

        class_labels = torch.cat([left_batch.class_labels.squeeze(1), right_batch.class_labels]).unsqueeze(1)
        zda_labels = torch.cat([left_batch.zda_labels.squeeze(1), right_batch.zda_labels.squeeze(1)]).unsqueeze(1)
        test_zda_labels = torch.cat([left_batch.test_zda_labels.squeeze(1), right_batch.test_zda_labels.squeeze(1)]).unsqueeze(1)

        return Batch(
                flow_features=flow_features,
                packet_features=packet_features,
                node_features=node_features,
                class_labels=class_labels,
                zda_labels=zda_labels,
                test_zda_labels=test_zda_labels)

    def get_zda_labels(self, batch, mode):
        """
        Retrieves ZDA labels based on natural-language labels and current knowledge.
        Optimized with caching for g1/g2 codes.
        """
        class_codes = batch.class_labels.squeeze(-1)
        if mode == TRAINING:
            g1_labels = tuple(sorted(self.env.current_knowledge['G1s']))
            if not hasattr(self, '_g1_cache') or self._g1_cache != g1_labels:
                g1_codes = self.encoder.get_codes_for_labels(self.env.current_knowledge['G1s'])
                self._g1_codes_tensor = torch.tensor(g1_codes, device=class_codes.device, dtype=class_codes.dtype)
                self._g1_cache = g1_labels

            zda_labels = torch.isin(class_codes, self._g1_codes_tensor).unsqueeze(-1).to(torch.float32)
            test_zda_labels = torch.zeros_like(zda_labels)
        elif mode == INFERENCE:
            g2_labels = tuple(sorted(self.env.current_knowledge['G2s']))
            if not hasattr(self, '_g2_cache') or self._g2_cache != g2_labels:
                g2_codes = self.encoder.get_codes_for_labels(self.env.current_knowledge['G2s'])
                self._g2_codes_tensor = torch.tensor(g2_codes, device=class_codes.device, dtype=class_codes.dtype)
                self._g2_cache = g2_labels

            zda_labels = torch.isin(class_codes, self._g2_codes_tensor).unsqueeze(-1).to(torch.float32)
            test_zda_labels = zda_labels

        return zda_labels, test_zda_labels

    def get_rewards_from_encoded_labels(self, encoded_labels):
        """
        Resolves per-sample rewards based on their encoded class labels.
        """
        if self._rewards_lookup is None or len(self._rewards_lookup) != len(self.encoder.get_mapping()):
            self._rewards_lookup = {
                class_idx: self.env.flow_rewards_dict[class_name]
                for class_name, class_idx in self.encoder.get_mapping().items()
            }

        rewards = [self._rewards_lookup[label.item()] for label in encoded_labels]
        return torch.tensor(rewards, dtype=torch.float32)

    def online_anomaly_detection(self, batch, logits, one_hot_labels, query_mask):
        """
        Performs anomaly detection on the online batch.
        """
        if self.use_neural_AD:
            known_class_h_mask = self.get_known_classes_mask(batch, one_hot_labels)
            
            try:
                zda_predictions = self.confidence_decoder(scores=logits[:, known_class_h_mask])
            except Exception as e:
                self.logger_instance.error(f'Confidence decoder error: {e}')
                raise RuntimeError(f'Confidence decoder error: {e}')
        
            predicted_zda_mask = (zda_predictions > 0.5).to(torch.bool).squeeze(-1)
        else:
            zda_predictions = batch.zda_labels[query_mask]
            predicted_zda_mask = zda_predictions.to(torch.bool).squeeze(-1)

        return zda_predictions, predicted_zda_mask
    
    def prepare_online_batch(self, online_batch):
        """
        Enriches the online batch with auxiliary samples for prototypical classification.
        """
        online_batch.zda_labels, online_batch.test_zda_labels = self.get_zda_labels(online_batch, mode=INFERENCE)

        aux_batch = self.sample_from_replay_buffers(samples_per_class=self.batch_size, mode=INFERENCE)
        if aux_batch is None:
            return None
    
        aux_query_mask = self.get_canonical_query_mask(aux_batch.class_labels.shape[0])
        online_query_mask = torch.ones_like(online_batch.class_labels).to(torch.bool)
        merged_query_mask = torch.cat([aux_query_mask, online_query_mask])

        accuracy_mask = torch.cat([torch.zeros_like(aux_query_mask), online_query_mask])
        merged_batch = self.merge_batches(aux_batch, online_batch)

        return merged_batch, merged_query_mask, accuracy_mask

    def perform_cs_inference(self, merged_batch, logits, predicted_online_zda_mask, num_of_online_samples, number_of_predicted_known_samples):
        """
        Performs closed-set classification for traffic predicted as known.
        """
        online_class_labels = merged_batch.class_labels[-num_of_online_samples:][~predicted_online_zda_mask].squeeze(-1)

        if self.use_neural_CS:
            online_class_preds = logits[-num_of_online_samples:][~predicted_online_zda_mask].max(1)[1]
        else:
            online_class_preds = online_class_labels

        known_correct_classification_mask = online_class_labels == online_class_preds

        # Calculate accuracy for reporting
        cs_acc = 1.0
        if online_class_labels.shape[0] > 0:
            cs_acc = (known_correct_classification_mask.sum() / online_class_labels.shape[0]).item()

        interest_logits_slice = logits[-num_of_online_samples:][~predicted_online_zda_mask]
        number_of_known_classes = logits.shape[1]

        if number_of_predicted_known_samples == 0:
            self.cs_classif_confidence = torch.ones(1) * 10
        else:
            non_choosed_mask = torch.ones(number_of_predicted_known_samples, number_of_known_classes)
            non_choosed_mask[torch.arange(number_of_predicted_known_samples), online_class_preds] = 0 
            mean_non_choosed_values = interest_logits_slice[non_choosed_mask.to(torch.bool)].mean()
            mean_choosed_logits = interest_logits_slice.max(1)[0].mean()
            self.cs_classif_confidence = torch.log(mean_choosed_logits / mean_non_choosed_values).unsqueeze(-1)
            
        return known_correct_classification_mask, cs_acc

    def evaluate_closed_set(self, class_labels, class_predictions, mode, accuracy_mask=None):
        """
        Calculates loss and accuracy for closed-set classification.
        Assumes class_labels and class_predictions already correspond to the same samples (e.g. query subset).
        """
        if mode == INFERENCE and not self.use_neural_CS:
            acc = torch.tensor(1.0, device=self.device)
            cs_loss = torch.tensor(0.0, device=self.device)
        else:
            targets = class_labels.squeeze(1)

            if accuracy_mask is not None:
                target_in = targets[accuracy_mask]
                pred_in = class_predictions[accuracy_mask]
            else:
                target_in = targets
                pred_in = class_predictions

            if isinstance(self.cs_criterion, nn.SmoothL1Loss):
                # Huber loss expects one-hot targets for classification if used this way
                oh_targets = torch.zeros_like(pred_in).scatter_(1, target_in.unsqueeze(1), 1)
                cs_loss = self.cs_criterion(pred_in, oh_targets).mean()
            else:
                cs_loss = self.cs_criterion(input=pred_in, target=target_in)

            acc = self.get_accuracy(logits_preds=class_predictions, decimal_labels=class_labels, accuracy_mask=accuracy_mask)

        metrics = {mode+'/'+CS_ACC: acc.item(), mode+'/'+CS_LOSS: cs_loss.item()}
        return cs_loss, acc, metrics

    def evaluate_zda_confidence(self, zda_predictions, predicted_online_zda_mask, num_of_online_samples):
        """
        Calculates confidence for anomaly detection.
        """
        online_anomaly_logits = zda_predictions[-num_of_online_samples:]
        online_non_anomaly_pred_logits = online_anomaly_logits[~predicted_online_zda_mask]
        online_anomaly_pred_logits = online_anomaly_logits[predicted_online_zda_mask]

        conf_normalizer = 0
        if online_non_anomaly_pred_logits.shape[0] > 0:
            self.zda_confidence += (1 - online_non_anomaly_pred_logits).mean()
            conf_normalizer += 1
        if online_anomaly_pred_logits.shape[0] > 0:
            self.zda_confidence += online_anomaly_pred_logits.mean()
            conf_normalizer += 1
        if conf_normalizer > 0:
            self.zda_confidence /= conf_normalizer

    def act_on_known_traffic(self, num_of_anomalies, num_known, correct_mask, hiddens, zda_mask, rewards):
        """
        Assembly state and perform action for known traffic.
        """
        classification_reward = 0
        empty_state_vec = -1 * torch.ones(1, hiddens.shape[1])

        state_vec = self.assembly_state_vector(empty_state_vec, num_of_anomalies, num_known, self.env.current_budget)

        if self.intrusion_detection_kwargs['automatic_cs_acceptance']:
            action_signal = torch.Tensor([0]).long()
        else:
            action_signal = self.act(state_vec)
              
        self.env.steps_done += 1
        self.wb_tracker.step_counter += 1

        known_samples_costs = rewards[~zda_mask]
        correct_classif_rewards = torch.zeros_like(known_samples_costs)
        bad_classif_costs = torch.zeros_like(known_samples_costs)
        no_confidence_penalty = 0

        if action_signal.item() == 0:
            correct_classif_rewards = torch.abs(known_samples_costs * correct_mask)
            if self.wrong_inference_penalisation == 'easy':
                bad_classif_costs = -torch.abs(known_samples_costs * (~correct_mask))
            else:
                bad_classif_costs = -torch.abs(known_samples_costs * (~correct_mask) * self.bad_classif_cost_factor)
            classification_reward += (correct_classif_rewards.sum() + bad_classif_costs.sum()).item()
        else:
            no_confidence_penalty = -float(self.intrusion_detection_kwargs['no_confidence_penalty'])
            classification_reward = no_confidence_penalty

        self.env.current_budget += classification_reward
        
        if not self.intrusion_detection_kwargs['automatic_cs_acceptance']:
            new_state = state_vec.detach().clone()
            new_state[-1] = self.env.current_budget 
            end_signal = torch.tensor([self.env.has_episode_ended(self.wb_tracker.step_counter)], dtype=torch.long)
            self.mitigation_agent.remember(state_vec.detach(), action_signal, torch.Tensor([classification_reward]), new_state, end_signal, self.wb_tracker.step_counter)

        self.env.episode_rewards.append(classification_reward)
        self.env.episode_budgets.append(self.env.current_budget)

        if self.wbt and self.wb_tracker.step_counter % self.report_step_freq == 0:
            self.reporter.log_scalars({
                AGENT+'/'+'generic_reward': self.env.episode_rewards[-1],
                AGENT+'/'+'classification_reward': classification_reward,
                AGENT+'/'+'budget': self.env.current_budget,
                AGENT+'/'+'correct_classification_rewards': correct_classif_rewards.sum().item(),
                AGENT+'/'+'bad_classification_cost': bad_classif_costs.sum().item(),
                AGENT+'/'+'known traffic action': action_signal.item(),
                AGENT+'/'+'no_confidence_penalty': no_confidence_penalty
            }, step=self.wb_tracker.step_counter)

    def collective_anomaly_detection(self, merged_batch, predicted_kernel, one_hot_labels, predicted_online_zda_mask, num_of_online_samples, hiddens):
        """
        Clusters eventual ZDAs using kernel regression or ground truth.
        """
        if self.use_neural_KR:
            _, predicted_decimal_clusters, kr_metrics = self.evaluate_kernel_regression(
                predicted_kernel[-num_of_online_samples:][:,-num_of_online_samples:], 
                one_hot_labels[-num_of_online_samples:],
                INFERENCE)
        else:
            predicted_decimal_clusters = merged_batch.class_labels[-num_of_online_samples:].squeeze(1)
            _, _, kr_metrics = self.evaluate_kernel_regression(
                predicted_kernel[-num_of_online_samples:][:,-num_of_online_samples:],
                one_hot_labels[-num_of_online_samples:],
                INFERENCE)
            
        num_clusters = predicted_decimal_clusters.max() + 1
        anomalous_clusters = predicted_decimal_clusters[predicted_online_zda_mask]
        predicted_clusters_oh = torch.nn.functional.one_hot(anomalous_clusters, num_classes=num_clusters)

        centroids, missing = self.get_centroids(hiddens[-num_of_online_samples:][predicted_online_zda_mask], predicted_clusters_oh.to(torch.float32))

        return predicted_clusters_oh, centroids, missing, kr_metrics
    
    def act_on_unknown_clusters(self, clusters_oh, centroids, missing, num_anom, num_known, zda_mask, rewards):
        """
        Performs mitigation actions (block/pass/CTI) on detected unknown clusters.
        """
        num_identified = centroids[~missing].shape[0]
        rewards_if_acc = (clusters_oh * rewards[zda_mask].unsqueeze(-1)).sum(0)
        cost_if_acc = -torch.relu(-rewards_if_acc)
        benign_rewards = torch.relu(rewards[zda_mask])
        benign_per_cluster = (clusters_oh * benign_rewards.unsqueeze(-1)).sum(0)

        clustering_reward = 0
        epistemic_actions_taken = 0
        epistemic_costs = 0
        rewards_per_accepted_clusters = 0
        rewards_per_blocked_clusters = 0

        for idx, centroid in enumerate(centroids[~missing]):
            accepted_cluster = False
            epistemic_action = False

            state_vec = self.assembly_state_vector(centroid.unsqueeze(0), num_anom, num_known, self.env.current_budget)
            action = self.act(state_vec)
            current_reward = 0

            if action == 0:
                accepted_cluster = True
            elif action == 2:
                epistemic_action = True
                accepted_cluster = not self.intrusion_detection_kwargs['epistemic_is_blocking']

            if accepted_cluster: 
                cost = cost_if_acc[~missing][idx]
                f = self.bad_clustering_cost_factor if self.wrong_inference_penalisation == 'hard' else 1.0
                current_reward += f * cost
                current_reward += benign_per_cluster[~missing][idx].item()
            else:
                cost = self.bad_classif_cost_factor * benign_per_cluster[~missing][idx]
                f = self.bad_clustering_cost_factor if self.wrong_inference_penalisation == 'hard' else 1.0
                current_reward -= f * cost
            
            if epistemic_action:
                updates_dict = self.perform_epistemic_action()
                current_reward -= updates_dict['price_payed']

            self.env.current_budget += current_reward
            next_state = state_vec.detach().clone()
            
            if idx < num_identified - 1:
                next_state[:-6] = centroids[~missing][idx+1]
            else:
                next_state[:-6] = -1 * torch.ones_like(next_state[:-6])
            
            next_state[-2] = self.env.epistemic_actions_available
            next_state[-1] = self.env.current_budget

            self.env.steps_done += 1
            self.wb_tracker.step_counter += 1
            end_signal = torch.tensor([self.env.has_episode_ended(self.wb_tracker.step_counter)], dtype=torch.long)

            self.mitigation_agent.remember(state_vec.detach(), action, current_reward, next_state, end_signal, self.wb_tracker.step_counter)
            self.env.episode_rewards.append(current_reward.item() if hasattr(current_reward, 'item') else current_reward)
            self.env.episode_budgets.append(self.env.current_budget)

            reward_val = current_reward.item() if hasattr(current_reward, 'item') else current_reward

            clustering_reward += reward_val
            epistemic_actions_taken += int(epistemic_action)
            epistemic_costs += reward_val if epistemic_action else 0
            rewards_per_accepted_clusters += reward_val if accepted_cluster else 0
            rewards_per_blocked_clusters += reward_val if not accepted_cluster else 0

        if len(centroids[~missing]) > 0 and \
            self.wbt and self.wb_tracker.step_counter % self.report_step_freq == 0:
                
                clustering_reward /= len(centroids[~missing])
                epistemic_actions_taken /= len(centroids[~missing])
                epistemic_costs /= len(centroids[~missing])
                rewards_per_accepted_clusters /= len(centroids[~missing])
                rewards_per_blocked_clusters /= len(centroids[~missing])

                self.reporter.log_scalars({
                    AGENT+'/'+'generic_reward': clustering_reward,
                    AGENT+'/'+'clustering_reward': clustering_reward,
                    AGENT+'/'+'budget': self.env.current_budget,
                    AGENT+'/'+'Epistemic Actions taken': epistemic_actions_taken,
                    AGENT+'/'+'epistemic_costs': epistemic_costs,
                    AGENT+'/'+'rewards_per_accepted_clusters': rewards_per_accepted_clusters,
                    AGENT+'/'+'rewards_per_blocked_clusters': rewards_per_blocked_clusters,
                }, step=self.wb_tracker.step_counter)

    def online_inference(self, online_batch):
        """
        Executes the online inference loop.
        """
        self.cs_classif_confidence = torch.zeros(1)
        self.zda_confidence = torch.zeros(1)
        
        self.classifier.eval()
        self.confidence_decoder.eval()

        online_batch_tuple = self.prepare_online_batch(online_batch)
        if online_batch_tuple is None: return
        
        merged_batch, merged_query_mask, accuracy_mask = online_batch_tuple
        
        with torch.no_grad():
            with self.profile("onl_inf_forward_pass"):
                logits, hiddens, predicted_kernel = self.infer(self.classifier, merged_batch, self.current_known_classes_count, query_mask=merged_query_mask)
            
            one_hot_labels = self.get_oh_labels(merged_batch, self.current_known_classes_count)

            with self.profile("onl_inf_AD"):
                zda_predictions, predicted_zda_mask = self.online_anomaly_detection(merged_batch, logits, one_hot_labels, merged_query_mask)
        
        num_online = online_batch.zda_labels.shape[0]
        # Always evaluate to update online stats (CMs) and get metrics
        # evaluate_anomaly_detection expects zda_labels and zda_predictions for the query subset
        _, _, ad_metrics = self.evaluate_anomaly_detection(merged_batch.zda_labels[merged_query_mask], zda_predictions, accuracy_mask[merged_query_mask], INFERENCE)
        # evaluate_closed_set expects pre-subsetted labels and predictions
        _, _, cs_metrics = self.evaluate_closed_set(merged_batch.class_labels[merged_query_mask], logits, INFERENCE, accuracy_mask=accuracy_mask[merged_query_mask])

        # accuracy_mask and merged_query_mask alignment:
        # logits shape is [N_query, K]. accuracy_mask[merged_query_mask] is [N_query].
        if not self.use_neural_CS:
            perfect_preds = one_hot_labels[merged_query_mask][accuracy_mask[merged_query_mask]]
            self.eval_cs_cm += efficient_cm(preds=perfect_preds, targets_onehot=perfect_preds)
        else:
            self.eval_cs_cm += efficient_cm(preds=logits[accuracy_mask[merged_query_mask]].detach(), targets_onehot=one_hot_labels[merged_query_mask][accuracy_mask[merged_query_mask]])

        pred_online_zda_mask = predicted_zda_mask[-num_online:]
        num_known = (~pred_online_zda_mask).sum()
        num_anom = pred_online_zda_mask.sum()
        rewards = self.get_rewards_from_encoded_labels(merged_batch.class_labels[-num_online:].squeeze(-1))

        self.evaluate_zda_confidence(zda_predictions, pred_online_zda_mask, num_online)
        correct_mask, cs_acc = self.perform_cs_inference(merged_batch, logits, pred_online_zda_mask, num_online, num_known)

        kr_metrics = {}
        if num_known > 0:
            if self.agency:
                self.act_on_known_traffic(num_anom, num_known, correct_mask, hiddens, pred_online_zda_mask, rewards)
            
            if num_anom > 0:
                with self.profile("onl_inf_CAD"):
                    clusters_oh, centroids, missing, kr_metrics = self.collective_anomaly_detection(merged_batch, predicted_kernel, one_hot_labels, pred_online_zda_mask, num_online, hiddens)
                if self.agency:
                    self.act_on_unknown_clusters(clusters_oh, centroids, missing, num_anom, num_known, pred_online_zda_mask, rewards)

        if self.agency:
            with self.profile("onl_inf_ER"):
                self.mitigation_agent.replay(self.wb_tracker.step_counter)

        if self.agency:
            self.logger_instance.info(f'Online {INFERENCE} current budget: {self.env.current_budget} \n')
        else:
            self.logger_instance.info(f'Online {INFERENCE} CS ACC: {cs_acc} \n')
        
        if self.wbt and self.wb_tracker.step_counter % self.report_step_freq == 0:
            all_metrics = {
                'online_inference/real_num_of_anomalies': online_batch.zda_labels.sum().item(),
                'online_inference/num_predicted_knowns': num_known.item(),
                'online_inference/num_predicted_unknowns': num_anom.item(),
                'online_inference/known_classif_confidente': self.cs_classif_confidence.item(),
                'online_inference/zda_classif_confidence': self.zda_confidence.item(),
            }
            all_metrics.update(ad_metrics)
            all_metrics.update(cs_metrics)
            all_metrics.update(kr_metrics)

            self.reporter.log_scalars(all_metrics, step=self.wb_tracker.step_counter)
            
        self.classifier.train()
        self.confidence_decoder.train()

        if self.agency and self.env.has_episode_ended(self.wb_tracker.step_counter): 
            if self.wbt:
                self.reporter.log_scalars({
                    'episode_count': self.episode_count,
                    'mean_episode_reward': torch.Tensor(self.env.episode_rewards).mean(),
                    'sum_episode_rewards': torch.Tensor(self.env.episode_rewards).sum(),
                    'mean_episode_budget': torch.Tensor(self.env.episode_budgets).mean(),
                    'epistemic_actions_per_episode': self.env.epistemic_actions,
                    'steps_per_episode': self.env.steps_done
                }, step=self.wb_tracker.step_counter)
            self.reset_environment()

    def get_centroids(self, hidden_vectors, onehot_labels):
        """Calculates centroids for each class."""
        cluster_agg = onehot_labels.T @ hidden_vectors
        samples_per_cluster = onehot_labels.sum(0)
        centroids = torch.zeros_like(cluster_agg, device=self.device)
        missing_clusters = samples_per_cluster == 0
        existent_centroids = cluster_agg[~missing_clusters] / samples_per_cluster[~missing_clusters].unsqueeze(-1)
        centroids[~missing_clusters] = existent_centroids
        return centroids, missing_clusters

    def assembly_state_vector(self, centroid, num_anom, num_known, curr_budget):
        """Assembles the state vector for the agent."""
        return torch.cat([
            centroid.squeeze(0),
            torch.tensor([
                float(num_anom), self.zda_confidence.item(),
                float(num_known), self.cs_classif_confidence.item(),
                float(self.env.epistemic_actions_available), float(curr_budget)
            ], device=centroid.device, dtype=centroid.dtype)
        ])
    
    def act(self, state_vec):
        """Gets action from the mitigation agent."""
        action = self.mitigation_agent.act(state_vec)
        return torch.Tensor([action]).long()

    def process_input(self, flows, node_feats: dict = None):
        """Main entry point for processing new network flows."""
        if len(flows) > 0:
            with self.profile("process_input_total"):
                with self.profile("input_assembly"):
                    batch = self.assembly_input_tensor(flows, node_feats)

                with self._lock:
                    self.push_to_replay_buffers(batch.flow_features, batch.packet_features, batch.node_features, batch_labels=batch.class_labels)

                    if self.batch_processing_allowed:
                        with self.profile("online_inference_total"):
                            self.online_inference(batch)
                
                    # we check again if batch_processing allowed because 
                    # knowledge can change during online inference.
                    if self.batch_processing_allowed:
                        with self.profile("train_inf_module_single_batch"):
                            self.train_inf_module_single_batch()

                    if not self.agency:
                        # If there's no agency, the step increments here, 
                        # otherwise it increments with each action
                        self.wb_tracker.step_counter += 1
        
        if self.agency and self.wb_tracker.step_counter % self.update_target_freq == 0:
            self.mitigation_agent.update_target_model()

        if self.agency and self.wb_tracker.step_counter % self.update_target_freq == 0:
            self.mitigation_agent.update_target_model()

    def _sample_from_frozen_buffers(self, frozen_buffers, frozen_int_to_label, frozen_knowledge, samples_per_class):
        """Helper for async evaluation thread to sample from buffers safely."""
        init = True
        balanced_flow_batch = None
        balanced_packet_batch = None
        balanced_node_feat_batch = None
        balanced_labels = None
        balanced_zda_labels = None
        balanced_test_zda_labels = None

        for class_idx, replay_buff in frozen_buffers.items():
            class_nl_label = frozen_int_to_label.get(class_idx)
            if class_nl_label is None: continue

            test_zda_batch_labels = zda_batch_labels = torch.zeros(samples_per_class, 1)
            if class_nl_label in frozen_knowledge.get('G2s', set()):
                test_zda_batch_labels = zda_batch_labels = torch.ones(samples_per_class, 1)

            try:
                flow_batch, packet_batch, node_feat_batch, batch_labels = replay_buff.sample(samples_per_class)
            except:
                continue

            if init:
                balanced_flow_batch = flow_batch
                balanced_labels = batch_labels
                balanced_zda_labels = zda_batch_labels
                balanced_test_zda_labels = test_zda_batch_labels
                balanced_packet_batch = packet_batch
                balanced_node_feat_batch = node_feat_batch
                init = False
            else:
                balanced_flow_batch = torch.vstack([balanced_flow_batch, flow_batch])
                balanced_labels = torch.vstack([balanced_labels, batch_labels])
                balanced_zda_labels = torch.vstack([balanced_zda_labels, zda_batch_labels])
                balanced_test_zda_labels = torch.vstack([balanced_test_zda_labels, test_zda_batch_labels])
                if packet_batch is not None: balanced_packet_batch = torch.vstack([balanced_packet_batch, packet_batch])
                if node_feat_batch is not None: balanced_node_feat_batch = torch.vstack([balanced_node_feat_batch, node_feat_batch])

        if init: return None
        return Batch(flow_features=balanced_flow_batch, packet_features=balanced_packet_batch, node_features=balanced_node_feat_batch, class_labels=balanced_labels, zda_labels=balanced_zda_labels, test_zda_labels=balanced_test_zda_labels)

    def sample_from_replay_buffers(self, samples_per_class, mode):
        """Samples a balanced batch from all active replay buffers."""
        all_flow, all_packet, all_node, all_labels, all_zda, all_test_zda = [], [], [], [], [], []
        classes_decimal = torch.tensor(list(self.replay_buffers.keys()), device=self.device).to(torch.long)
        nl_labels = self.encoder.inverse_transform(classes_decimal)
        
        for replay_buff, nl_label in zip(self.replay_buffers.values(), nl_labels):
            test_zda_labels = zda_labels = torch.zeros(samples_per_class, 1)

            if mode == TRAINING:
                if nl_label in self.env.current_knowledge['G2s']: continue
                if nl_label in self.env.current_knowledge['G1s']: zda_labels = torch.ones(samples_per_class, 1)
            
            if mode == INFERENCE:
                if nl_label in self.env.current_knowledge['G2s']:
                    test_zda_labels = zda_labels = torch.ones(samples_per_class, 1)
            
            try:
                f, p, n, l = replay_buff.sample(samples_per_class)
            except:
                continue

            all_flow.append(f)
            all_labels.append(l)
            all_zda.append(zda_labels)
            all_test_zda.append(test_zda_labels)
            if p is not None: all_packet.append(p)
            if n is not None: all_node.append(n)

        if not all_flow: return None
        return Batch(flow_features=torch.cat(all_flow, dim=0), packet_features=(torch.cat(all_packet, dim=0) if all_packet else None), node_features=(torch.cat(all_node, dim=0) if all_node else None), class_labels=torch.cat(all_labels, dim=0), zda_labels=torch.cat(all_zda, dim=0), test_zda_labels=torch.cat(all_test_zda, dim=0))

    def get_canonical_query_mask(self, whole_batch_size):
        """Creates a query mask based on the K-shot parameter."""
        N = whole_batch_size // self.batch_size
        M = self.batch_size
        query_mask = torch.ones(size=(N, M), device=self.device).to(torch.bool)
        query_mask[:, :self.k_shot] = False
        return query_mask.view(-1)

    def get_oh_labels(self, batch, class_shape):
        """One-hot encodes class labels."""
        targets = batch.class_labels.to(torch.int64)
        targets_onehot = torch.zeros(size=(batch.class_labels.shape[0], class_shape), device=targets.device)
        targets_onehot.scatter_(1, targets.view(-1, 1), 1)
        return targets_onehot
    
    def evaluate_anomaly_detection(self, zda_labels, zda_predictions, accuracy_mask, mode):
        """Calculates loss and accuracy for ZDA detection."""
        if mode == INFERENCE and not self.use_neural_AD:
            os_loss = torch.tensor(0.0, device=self.device)
            cummulative_os_acc = torch.tensor(1.0, device=self.device)
            zda_balance = zda_labels[accuracy_mask].to(torch.float32).mean().item()

            # Update confusion matrix with perfect predictions
            onehot_zda_labels = torch.zeros(size=(zda_labels.shape[0], 2), device=self.device).long()
            onehot_zda_labels.scatter_(1, zda_labels.long().view(-1, 1), 1)
            batch_os_cm = efficient_os_cm(preds=zda_labels[accuracy_mask].long().squeeze(-1), targets_onehot=onehot_zda_labels[accuracy_mask].long())
            self.eval_os_cm += batch_os_cm
        else:
            os_loss = self.os_criterion(input=zda_predictions[accuracy_mask], target=zda_labels[accuracy_mask])
            onehot_zda_labels = torch.zeros(size=(zda_labels.shape[0], 2), device=self.device).long()
            onehot_zda_labels.scatter_(1, zda_labels.long().view(-1, 1), 1)

            batch_os_cm = efficient_os_cm(preds=(zda_predictions[accuracy_mask].detach() > 0.5).long(), targets_onehot=onehot_zda_labels[accuracy_mask].long())

            cummulative_os_cm = (self.training_os_cm if mode == TRAINING else self.eval_os_cm)
            cummulative_os_cm += batch_os_cm
            zda_balance = zda_labels[accuracy_mask].to(torch.float32).mean().item()
            cummulative_os_acc = get_balanced_accuracy(cummulative_os_cm, negative_weight=0.5)

        metrics = {mode+'/'+OS_ACC: cummulative_os_acc.item(), mode+'/'+OS_LOSS: os_loss.item(), mode+'/'+ANOMALY_BALANCE: zda_balance}
        return os_loss, cummulative_os_acc, metrics
    
    def evaluate_kernel_regression(self, predicted_kernel, one_hot_labels, mode):
        """Evaluates clustering performance via kernel regression."""
        if not self.kernel_regression:
            return 0, None, {}

        if mode == INFERENCE and not self.use_neural_KR:
            kernel_loss = torch.tensor(0.0, device=self.device)
            kr_ari = 1.0
            kr_nmi = 1.0
            decimal_predicted = one_hot_labels.max(1)[1]
        else:
            semantic_kernel = one_hot_labels @ one_hot_labels.T
            kernel_loss = self.kr_criterion(baseline_kernel=semantic_kernel, predicted_kernel=predicted_kernel)

            decimal_semantic = one_hot_labels.max(1)[1].detach().cpu().numpy()
            decimal_predicted = get_clusters(predicted_kernel.detach())
            np_dec_pred = decimal_predicted.cpu().numpy()

            kr_ari = adjusted_rand_score(decimal_semantic, np_dec_pred)
            kr_nmi = normalized_mutual_info_score(decimal_semantic, np_dec_pred)

        metrics = {mode+'/'+KR_ARI: kr_ari, mode+'/'+KR_NMI: kr_nmi, mode+'/'+KR_LOSS: kernel_loss.item()}
        return kernel_loss, decimal_predicted, metrics

    def get_known_classes_mask(self, batch, one_hot_labels):
        """Identifies classes that are NOT ZDAs in the current batch."""
        known_oh_labels = one_hot_labels[~batch.zda_labels.squeeze(1).bool()]
        return known_oh_labels.sum(0) > 0

    def train_inf_module_single_batch(self):
        """Performs a training step of the inference module sampling from buffers."""
        training_batch = self.sample_from_replay_buffers(samples_per_class=self.batch_size, mode=TRAINING)
        if training_batch is None: return
        
        training_batch.zda_labels, training_batch.test_zda_labels = self.get_zda_labels(training_batch, mode=TRAINING)
        query_mask = self.get_canonical_query_mask(training_batch.class_labels.shape[0])

        
        logits, hiddens, pred_kernel = self.infer(self.classifier, training_batch, self.current_known_classes_count, query_mask=query_mask)
        
        one_hot_labels = self.get_oh_labels(training_batch, logits.shape[1])
        known_h_mask = self.get_known_classes_mask(training_batch, one_hot_labels)
        loss = 0

        ad_metrics = {}
        if torch.any(known_h_mask):
            zda_preds = self.confidence_decoder(scores=logits[:, known_h_mask])
            if self.multi_class:
                zda_loss, _, ad_metrics = self.evaluate_anomaly_detection(training_batch.zda_labels[query_mask], zda_preds, torch.ones(query_mask.sum()).to(torch.bool), TRAINING)
                loss += zda_loss

        kr_loss, pred_clusters, kr_metrics = self.evaluate_kernel_regression(pred_kernel, one_hot_labels, TRAINING)
        if self.clustering_loss_backprop: loss += kr_loss
        
        classif_loss, cs_acc, cs_metrics = self.evaluate_closed_set(training_batch.class_labels[query_mask], logits, TRAINING)
        loss += classif_loss

        self.training_cs_cm += efficient_cm(preds=logits.detach(), targets_onehot=one_hot_labels[query_mask])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        

        if self.wb_tracker.step_counter % self.report_step_freq == 0:
            all_metrics = {}
            all_metrics.update(ad_metrics)
            all_metrics.update(kr_metrics)
            all_metrics.update(cs_metrics)
            self.reset_train_cms()
            self.reporter.log_scalars(all_metrics, step=self.wb_tracker.step_counter)

        if self.wb_tracker.step_counter % self.plot_step_freq == 0:
            plots = self.reporter.report(logits[:,known_h_mask], hiddens.detach(), training_batch.class_labels, pred_clusters, query_mask, TRAINING, training_cs_cm=self.training_cs_cm, training_os_cm=self.training_os_cm)
            if self.wbt: self.wb_run.log(plots, step=self.wb_tracker.step_counter)
        
        if self.online_evaluation and self.wb_tracker.step_counter % self.online_eval_step_freq == 0:
            self.start_async_evaluation()

            while not self.eval_queue.empty():
                async_results = self.eval_queue.get()
                if self.wbt: self.reporter.log_scalars(async_results, step=self.wb_tracker.step_counter)
                # Reset evaluation confusion matrices after reporting
                self.reset_test_cms()

                if self.save_models_flag:
                    self.check_progress_and_save(
                        async_results[f'{EVALUATION}/Mean EVAL CS ACC'], 
                        async_results[f'{EVALUATION}/Mean EVAL AD ACC'], 
                        async_results[f'{EVALUATION}/Mean EVAL KR PREC'])

    @epistemic_thread_safe 
    def perform_epistemic_action(self, current_action=0):      
        """Acquires a CTI label, updating the knowledge base and replay buffers."""
        updates = self.env.perform_epistemic_action(current_action)
        new_label = updates['updated_label']
        
        if new_label is not None:
            if self.encoder.update_label(new_label=new_label, logger=self.logger_instance):
                self.current_known_classes_count += 1
                self.add_replay_buffer(new_label)
                self.reset_train_cms()
                self.reset_test_cms()
        return updates

    def start_async_evaluation(self):
        """Starts a background thread for model evaluation."""
        if hasattr(self, '_eval_thread') and self._eval_thread.is_alive(): return
        self._eval_thread = threading.Thread(target=self._async_evaluate_models, args=(self.current_known_classes_count,))
        self._eval_thread.start()

    def _async_evaluate_models(self, known_count):
        """Background worker for lock-free model evaluation."""
        with self._epistemic_lock:
            frozen_buffers = dict(self.replay_buffers)
            frozen_int_to_label = dict(self.encoder._int_to_label)
            frozen_knowledge = {k: set(v) for k, v in self.env.current_knowledge.items()}

        classifier_clone = copy.deepcopy(self.classifier).eval()
        decoder_clone = copy.deepcopy(self.confidence_decoder).eval() if self.multi_class else None
        
        m_ad, m_cs, m_kr = 0.0, 0.0, 0.0
        l_cs_cm = torch.zeros([known_count, known_count], device=self.device)
        l_os_cm = torch.zeros(size=(2, 2), device=self.device)
        l_logits, l_hiddens, l_labels, l_pred, l_q_mask, l_k_mask = None, None, None, None, None, None

        with torch.no_grad():
            for _ in range(self.online_eval_rounds):
                eval_batch = self._sample_from_frozen_buffers(frozen_buffers, frozen_int_to_label, frozen_knowledge, self.batch_size)
                if eval_batch is None: continue
                
                q_mask = self.get_canonical_query_mask(eval_batch.class_labels.shape[0])
                logits, hiddens, pred_kernel = self.infer(classifier_clone, eval_batch, known_count, query_mask=q_mask)
                oh = self.get_oh_labels(eval_batch, logits.shape[1])
                k_mask = self.get_known_classes_mask(eval_batch, oh)

                ad_acc = 0.0
                if self.multi_class and decoder_clone:
                    if self.use_neural_AD:
                        zda_p = decoder_clone(scores=logits[:, k_mask])
                        zda_l = eval_batch.zda_labels[q_mask]
                        oh_zda = torch.zeros(size=(zda_l.shape[0], 2), device=self.device).long().scatter(1, zda_l.long().view(-1, 1), 1)
                        b_os_cm = efficient_os_cm(preds=(zda_p > 0.5).long(), targets_onehot=oh_zda)
                        l_os_cm += b_os_cm
                        ad_acc = get_balanced_accuracy(b_os_cm, negative_weight=zda_l.to(torch.float32).mean().item()).item()
                    else:
                        ad_acc = 1.0
                        zda_l = eval_batch.zda_labels[q_mask]
                        oh_zda = torch.zeros(size=(zda_l.shape[0], 2), device=self.device).long().scatter(1, zda_l.long().view(-1, 1), 1)
                        b_os_cm = efficient_os_cm(preds=zda_l.long().squeeze(-1), targets_onehot=oh_zda)
                        l_os_cm += b_os_cm

                kr_p, pred_cl = 0.0, None
                if self.kernel_regression:
                    if self.use_neural_KR:
                        pred_cl = get_clusters(pred_kernel)
                        kr_p = adjusted_rand_score(oh.max(1)[1].cpu().numpy(), pred_cl.cpu().numpy())
                    else:
                        pred_cl = oh.max(1)[1]
                        kr_p = 1.0

                l_cs_cm += efficient_cm(preds=logits, targets_onehot=oh[q_mask])
                if self.use_neural_CS:
                    match = logits.max(1)[1] == eval_batch.class_labels.max(1)[0][q_mask]
                    cs_acc = (match.sum() / match.shape[0]).item()
                else:
                    cs_acc = 1.0

                m_ad += ad_acc / self.online_eval_rounds
                m_cs += cs_acc / self.online_eval_rounds
                m_kr += kr_p / self.online_eval_rounds
                l_logits, l_hiddens, l_labels, l_pred, l_q_mask, l_k_mask = logits, hiddens, eval_batch.class_labels, pred_cl, q_mask, k_mask

        plots = self.reporter.report(l_logits[:, l_k_mask], l_hiddens, l_labels, l_pred, l_q_mask, EVALUATION, custom_cs_cm=l_cs_cm, custom_os_cm=l_os_cm)
        self.eval_queue.put({
            f'{EVALUATION}/Mean EVAL AD ACC': m_ad, f'{EVALUATION}/Mean EVAL CS ACC': m_cs, f'{EVALUATION}/Mean EVAL KR PREC': m_kr, **plots
        })

    def get_profiling_stats_dict(self):
        """Computes and clears the profiling statistics."""
        metrics = {k: sum(v)/len(v) for k, v in self.profiling_stats.items() if len(v) > 0}
        self.profiling_stats.clear()
        return metrics

    def check_progress_and_save(self, curr_cs, curr_ad, curr_kr):
        """Checks if current performance is better than previous best and saves models."""
        if curr_cs > self.best_cs_accuracy:
            self.best_cs_accuracy = curr_cs
            self.save_model(self.classifier, self.classifier_path + 'single.pt', "flow classifier")
        if curr_ad > self.best_AD_accuracy:
            self.best_AD_accuracy = curr_ad
            self.save_model(self.confidence_decoder, self.confidence_decoder_path + 'single.pt', "confidence decoder")
        if curr_kr > self.best_KR_accuracy:
            self.best_KR_accuracy = curr_kr
            self.save_model(self.classifier, self.classifier_path + 'coupled.pt', "flow classifier (coupled)")
            if self.multi_class: self.save_model(self.confidence_decoder, self.confidence_decoder_path + 'coupled.pt', "confidence decoder (coupled)")

    def save_model(self, model, path, name):
        """Saves a model's state dictionary to a file."""
        torch.save(model.state_dict(), path)
        self.logger_instance.info(f'\033[95mNew {name} model version saved to {path}\033[0m')

    def get_accuracy(self, logits_preds, decimal_labels, accuracy_mask=None):
        """Calculates accuracy for a set of predictions."""
        preds = logits_preds.max(1)[1]
        targets = decimal_labels.max(1)[0]
        if accuracy_mask is not None:
            preds = preds[accuracy_mask]
            targets = targets[accuracy_mask]

        if targets.shape[0] == 0: return torch.tensor(1.0, device=self.device)
        match = preds == targets
        return match.sum() / match.shape[0]

    def get_labels(self, flows):
        """Encodes string labels from flows into integers."""
        labels = [f.element_class for f in flows]
        for cl in self.encoder.fit(labels): self.add_class_to_knowledge_base(cl)
        return self.encoder.transform(labels).to(torch.long)
    
    def assembly_input_tensor(self, flows, node_feats):
        """Assemblies a batch from current flow observations."""
        f_batch = torch.stack([f.get_flow_features() for f in flows])
        p_batch = torch.stack([f.get_packet_features() for f in flows]) if self.use_packet_feats else None
        n_batch = torch.stack([get_metrics_tensor(node_feats, f.dest_ip, self.kwargs['health']) for f in flows]) if self.use_node_feats else None
        return Batch(flow_features=f_batch, packet_features=p_batch, node_features=n_batch, class_labels=self.get_labels(flows))
