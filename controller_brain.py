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
from smartController.neural_modules import  MultiClassFlowClassifier, ThreeStreamMulticlassFlowClassifier, \
        TwoStreamMulticlassFlowClassifier, KernelRegressionLoss, ConfidenceDecoder
from smartController.replay_buffer import ReplayBuffer, Batch
import os
import torch
import torch.optim as optim
import torch.nn as nn
from smartController.wandb_tracker import WandBTracker
from pox.core import core  # only for logging.
import seaborn as sns
import matplotlib.pyplot as plt
import threading
from wandb import Image as wandbImage
import itertools
from sklearn.decomposition import PCA
import random
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
import requests 

# List of colors
colors = [
    'red', 'blue', 'green', 'purple', 'orange', 'pink', 'cyan',  'brown', 'yellow',
    'olive', 'lime', 'teal', 'maroon', 'navy', 'fuchsia', 'aqua', 'silver', 'sienna', 'gold',
    'indigo', 'violet', 'turquoise', 'tomato', 'orchid', 'slategray', 'peru', 'magenta', 'limegreen',
    'royalblue', 'coral', 'darkorange', 'darkviolet', 'darkslateblue', 'dodgerblue', 'firebrick',
    'lightseagreen', 'mediumorchid', 'orangered', 'powderblue', 'seagreen', 'springgreen', 'tan', 'wheat',
    'burlywood', 'chartreuse', 'crimson', 'darkgoldenrod', 'darkolivegreen', 'darkseagreen', 'indianred',
    'lavender', 'lightcoral', 'lightpink', 'lightsalmon', 'limegreen', 'mediumseagreen', 'mediumpurple',
    'midnightblue', 'palegreen', 'rosybrown', 'saddlebrown', 'salmon', 'slateblue', 'steelblue',
]

"""
######## PORT STATS: ###################
00: 'collisions'
01: 'port_no'
02: 'rx_bytes'
03: 'rx_crc_err'
04: 'rx_dropped'
05: 'rx_errors'
06: 'rx_frame_err'
07: 'rx_over_err'
08: 'rx_packets'
09: 'tx_bytes'
10: 'tx_dropped'
11: 'tx_errors'
12: 'tx_packets'
#########################################


#############FLOW FEATURES###############
'byte_count', 
'duration_nsec' / 10e9,
'duration_sec',
'packet_count'
#########################################
"""

RAM = 'RAM'
CPU = 'CPU'
IN_TRAFFIC = 'IN_TRAFFIC'
OUT_TRAFFIC = 'OUT_TRAFFIC'
DELAY = 'DELAY'



PRETRAINED_MODELS_DIR = '/pox/pox/smartController/models/'

REPLAY_BUFFER_MAX_CAPACITY=1000

LEARNING_RATE=1e-3

REPULSIVE_WEIGHT = 1

ATTRACTIVE_WEIGHT = 1

KERNEL_REGRESSOR_HEADS = 2

EVALUATION_ROUNDS = 50

CLUSTERING_LOSS_BACKPROP = True

# Constants for wandb monitoring:
INFERENCE = 'Inference'
TRAINING = 'Training'
CS_ACC = 'Acc'
CS_LOSS = 'Loss'
OS_ACC = 'AD Acc'
OS_LOSS = 'AD Loss'
KR_LOSS = 'KR_LOSS'
KR_ARI = 'KR_ARI'
KR_NMI = 'KR_NMI'
STEP_LABEL = 'step'
ANOMALY_BALANCE = 'ANOMALY_BALANCE'
CLOSED_SET = 'CS'
ANOMALY_DETECTION = 'AD'

# Create a lock object
lock = threading.Lock()


def efficient_cm(preds, targets_onehot):

    predictions_decimal = preds.argmax(dim=1).to(torch.int64)
    predictions_onehot = torch.zeros_like(
        preds,
        device=preds.device)
    predictions_onehot.scatter_(1, predictions_decimal.view(-1, 1), 1)

    return targets_onehot.T @ predictions_onehot


def efficient_os_cm(preds, targets_onehot):

    predictions_onehot = torch.zeros(
        [preds.size(0), 2],
        device=preds.device)
    predictions_onehot.scatter_(1, preds.view(-1, 1), 1)

    return targets_onehot.T @ predictions_onehot.long()


def get_balanced_accuracy(os_cm, negative_weight):
        
    N = os_cm[0][0] + os_cm[0][1]
    TN = os_cm[0][0]
    TNR = TN / (N + 1e-10)
        

    P = os_cm[1][1] + os_cm[1][0]
    TP = os_cm[1][1]
    TPR = TP / (P + 1e-10)
    
    return (negative_weight * TNR) + ((1-negative_weight) * TPR)


def get_clusters(predicted_kernel):
        
        discrete_predicted_kernel = (predicted_kernel > 0.5).long()
        
        assigned_mask = torch.zeros_like(discrete_predicted_kernel.diag())
        clusters = torch.zeros_like(discrete_predicted_kernel.diag())
        curr_cluster = 1

        for idx in range(discrete_predicted_kernel.shape[0]):
            if assigned_mask[idx] > 0:
                continue
            new_cluster_mask = discrete_predicted_kernel[idx]
            new_cluster_mask = torch.relu(new_cluster_mask - assigned_mask)
            assigned_mask += new_cluster_mask
            clusters += new_cluster_mask*curr_cluster
            if new_cluster_mask.sum() > 0:
                curr_cluster += 1

        return clusters -1 
    

class DynamicLabelEncoder:

    def __init__(self):
        self.label_to_int = {}
        self.int_to_label = {}
        self.current_code = 0


    def fit(self, labels):
        """
        returns the number of new classes found!
        """
        new_labels = set(labels) - set(self.label_to_int.keys())

        for label in new_labels:
            self.label_to_int[label] = self.current_code
            self.int_to_label[self.current_code] = label
            self.current_code += 1

        return new_labels


    def transform(self, labels):
        encoded_labels = [self.label_to_int[label] for label in labels]
        return torch.tensor(encoded_labels)


    def inverse_transform(self, encoded_labels):
        decoded_labels = [self.int_to_label[code.item()] for code in encoded_labels]
        return decoded_labels


    def get_mapping(self):
        return self.label_to_int


    def get_labels(self):
        return list(self.label_to_int.keys())


class ControllerBrain():

    def __init__(self,
                 eval,
                 use_packet_feats,
                 use_node_feats,
                 flow_feat_dim,
                 packet_feat_dim,
                 h_dim,
                 dropout,
                 multi_class,
                 init_k_shot,
                 replay_buffer_batch_size,
                 kernel_regression,
                 host_ip_addr,
                 device='cpu',
                 seed=777,
                 debug=False,
                 wb_track=False,
                 wb_project_name='',
                 wb_run_name='',
                 report_step_freq=50,
                 **kwargs):
        
        self.eval = eval
        self.use_packet_feats = use_packet_feats
        self.use_node_feats = use_node_feats
        self.flow_feat_dim = flow_feat_dim
        self.packet_feat_dim = packet_feat_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.multi_class = multi_class
        self.AI_DEBUG = debug
        self.best_cs_accuracy = 0
        self.best_AD_accuracy = 0
        self.best_KR_accuracy = 0
        self.backprop_counter = 0
        self.wbt = wb_track
        self.wbl = None
        self.kernel_regression = kernel_regression
        self.logger_instance = core.getLogger()
        self.device=device
        self.seed = seed
        random.seed(seed)
        self.current_known_classes_count = 0
        self.current_training_known_classes_count = 0
        self.current_test_known_classes_count = 0
        self.reset_train_cms()
        self.reset_test_cms()
        self.k_shot = init_k_shot
        self.replay_buff_batch_size = replay_buffer_batch_size
        self.encoder = DynamicLabelEncoder()
        self.replay_buffers = {}
        self.test_replay_buffers = {}
        self.init_neural_modules(LEARNING_RATE, seed)
        self.report_step_freq = report_step_freq
        self.container_manager_ep = f'http://{host_ip_addr}:7777/'
        if self.wbt:

            kwargs['PRETRAINED_MODEL_PATH'] = PRETRAINED_MODELS_DIR
            kwargs['REPLAY_BUFFER_MAX_CAPACITY'] = REPLAY_BUFFER_MAX_CAPACITY
            kwargs['LEARNING_RATE'] = LEARNING_RATE
            kwargs['REPORT_STEP_FREQUENCY'] = self.report_step_freq
            kwargs['KERNEL_REGRESSOR_HEADS'] = KERNEL_REGRESSOR_HEADS
            kwargs['REPULSIVE_WEIGHT']  = REPULSIVE_WEIGHT
            kwargs['ATTRACTIVE_WEIGHT']  = ATTRACTIVE_WEIGHT
            # Clustering loss is actually the opposite of a regularization, because it slightly may bias the manifold
            # toward training class distributions. We keep the "regularization" name in wandb reporting for retrocompatibility.
            kwargs['REGULARIZATION']  = CLUSTERING_LOSS_BACKPROP

            self.wbl = WandBTracker(
                wanb_project_name=wb_project_name,
                run_name=wb_run_name,
                config_dict=kwargs).wb_logger        
        self.restart_traffic()


    def restart_traffic(self):
        print(f'stopping eventual traffic flows...')
        response = requests.get(self.container_manager_ep+'stop_traffic')
        time.sleep(0.5)
        print(f'launching traffic...')
        response = requests.get(self.container_manager_ep+'launch_traffic')
        if response.status_code == 200:
            data = response.text
            print(data)
        else:
            print(f"Error: Received status code {response.status_code}")


    def add_replay_buffer(self, class_name):
        self.inference_allowed = False
        self.experience_learning_allowed = False
        self.eval_allowed = False 
        
        if not 'G2' in class_name:
            self.replay_buffers[self.current_known_classes_count-1] = ReplayBuffer(
                capacity=REPLAY_BUFFER_MAX_CAPACITY,
                batch_size=self.replay_buff_batch_size,
                seed=self.seed)
            self.logger_instance.info(f'Added a replay buffer with code {self.current_known_classes_count-1}' +\
                                        ' in the training replay buffers')
        if not 'G1' in class_name:
            self.test_replay_buffers[self.current_known_classes_count-1] = ReplayBuffer(
                        capacity=REPLAY_BUFFER_MAX_CAPACITY,
                        batch_size=self.replay_buff_batch_size,
                        seed=self.seed)
            self.logger_instance.info(f'Added a replay buffer with code {self.current_known_classes_count-1}' +\
                                      ' in the test replay buffers')


    def add_class_to_knowledge_base(self, new_class):
        if self.AI_DEBUG:
            self.logger_instance.info(f'New class found: {new_class}')
        self.current_known_classes_count += 1
        if not 'G2' in new_class:
            self.current_training_known_classes_count += 1 
        if not 'G1' in new_class:
            self.current_test_known_classes_count += 1
        self.add_replay_buffer(new_class)
        self.reset_train_cms()
        self.reset_test_cms()


    def reset_train_cms(self):
        self.training_cs_cm = torch.zeros(
            [self.current_known_classes_count, self.current_known_classes_count],
            device=self.device)
        self.training_os_cm = torch.zeros(
            size=(2, 2),
            device=self.device)
        
    
    def reset_test_cms(self):
        self.eval_cs_cm = torch.zeros(
            [self.current_known_classes_count, self.current_known_classes_count],
            device=self.device)
        self.eval_os_cm = torch.zeros(
            size=(2, 2),
            device=self.device)
        

    def init_neural_modules(self, lr, seed):
        torch.manual_seed(seed)
        self.confidence_decoder = ConfidenceDecoder(device=self.device)
        self.os_criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.cs_criterion = nn.CrossEntropyLoss().to(self.device)
        self.kr_criterion = KernelRegressionLoss(repulsive_weigth=REPULSIVE_WEIGHT, 
            attractive_weigth=ATTRACTIVE_WEIGHT).to(self.device)
        
        if self.use_packet_feats:
            
            if self.use_node_feats:

                self.classifier = ThreeStreamMulticlassFlowClassifier(
                    flow_input_size=self.flow_feat_dim, 
                    second_stream_input_size=self.packet_feat_dim,
                    third_stream_input_size=5,
                    hidden_size=self.h_dim,
                    kr_heads=KERNEL_REGRESSOR_HEADS,
                    dropout_prob=self.dropout,
                    device=self.device)
            
            else: 

                self.classifier = TwoStreamMulticlassFlowClassifier(
                    flow_input_size=self.flow_feat_dim, 
                    second_stream_input_size=self.packet_feat_dim,
                    hidden_size=self.h_dim,
                    kr_heads=KERNEL_REGRESSOR_HEADS,
                    dropout_prob=self.dropout,
                    device=self.device)

        else:
            
            if self.use_node_feats:
                
                self.classifier = TwoStreamMulticlassFlowClassifier(
                    flow_input_size=self.flow_feat_dim, 
                    second_stream_input_size=5,
                    hidden_size=self.h_dim,
                    kr_heads=KERNEL_REGRESSOR_HEADS,
                    dropout_prob=self.dropout,
                    device=self.device)
            else:

                self.classifier = MultiClassFlowClassifier(
                    input_size=self.flow_feat_dim, 
                    hidden_size=self.h_dim,
                    dropout_prob=self.dropout,
                    kr_heads=KERNEL_REGRESSOR_HEADS,
                    device=self.device)
            


        self.check_pretrained()

        params_for_optimizer = \
            list(self.confidence_decoder.parameters()) + \
            list(self.classifier.parameters())

        self.classifier.to(self.device)
        self.optimizer = optim.Adam(
            params_for_optimizer, 
            lr=lr)

        if self.eval:
            self.classifier.eval()
            self.confidence_decoder.eval()
            self.logger_instance.info(f"Using MODULES in EVAL mode!")                


    def check_pretrained(self):

        if self.use_packet_feats:
            if self.use_node_feats:
                self.classifier_path = PRETRAINED_MODELS_DIR+'multiclass_flow_packet_node_classifier_pretrained'
                self.confidence_decoder_path = PRETRAINED_MODELS_DIR+'flow_packet_node_confidence_decoder_pretrained'
            else:
                self.classifier_path = PRETRAINED_MODELS_DIR+'multiclass_flow_packet_classifier_pretrained'
                self.confidence_decoder_path = PRETRAINED_MODELS_DIR+'flow_packet_confidence_decoder_pretrained'
        else:
            if self.use_node_feats:
                self.classifier_path = PRETRAINED_MODELS_DIR+'multiclass_flow_node_classifier_pretrained'
                self.confidence_decoder_path = PRETRAINED_MODELS_DIR+'flow_node_confidence_decoder_pretrained'
            else:    
                self.classifier_path = PRETRAINED_MODELS_DIR+'multiclass_flow_classifier_pretrained'
                self.confidence_decoder_path = PRETRAINED_MODELS_DIR+'flow_confidence_decoder_pretrained'

        # Check if the file exists
        if os.path.exists(PRETRAINED_MODELS_DIR):

            if os.path.exists(self.classifier_path+'.pt'):
                # Load the pre-trained weights
                self.classifier.load_state_dict(torch.load(self.classifier_path+'.pt'))
                self.logger_instance.info(f"Pre-trained weights loaded successfully from {self.classifier_path}.pt")
            else:
                self.logger_instance.info(f"Pre-trained weights not found at {self.classifier_path}.pt")
                
            if self.multi_class:
                if os.path.exists(self.confidence_decoder_path+'.pt'):
                    self.confidence_decoder.load_state_dict(torch.load(self.confidence_decoder_path+'.pt'))
                    self.logger_instance.info(f"Pre-trained weights loaded successfully from {self.confidence_decoder_path}.pt")
                else:
                    self.logger_instance.info(f"Pre-trained weights not found at {self.confidence_decoder_path}.pt")             
             

        elif self.AI_DEBUG:
            self.logger_instance.info(f"Pre-trained folder not found at {PRETRAINED_MODELS_DIR}.")


    def infer(
            self,
            batch,
            query_mask):
        """
        Forward inference pass on neural modules.
        Notice that we can feed our prototypical learners with all kind of labels, (including zdas)
        as far as we do not back-propagate error gradients from the corresponding outputs. 
        We'll handle this epistemic correctness during learning, not during forward pass. 
            Returns:
                logits: logit multiclass classification predictions (only for the QUERY samples!) 
                hiddens: 
                predicted_kernel:
        """
        if self.use_packet_feats:
            if self.use_node_feats:
                logits, hiddens, predicted_kernel = self.classifier(
                    batch.flow_features, 
                    batch.packet_features, 
                    batch.node_features,
                    batch.class_labels, 
                    self.current_known_classes_count,
                    query_mask)
            else:
                logits, hiddens, predicted_kernel = self.classifier(
                    batch.flow_features, 
                    batch.packet_features, 
                    batch.class_labels, 
                    self.current_known_classes_count,
                    query_mask)
        else:
            if self.use_node_feats:
                logits, hiddens, predicted_kernel = self.classifier(
                    batch.flow_features, 
                    batch.node_features, 
                    batch.class_labels, 
                    self.current_known_classes_count,
                    query_mask)
            else:
                logits, hiddens, predicted_kernel = self.classifier(
                    batch.flow_features, 
                    batch.class_labels, 
                    self.current_known_classes_count,
                    query_mask)


        return logits, hiddens, predicted_kernel


    def push_to_replay_buffers(
            self,
            flow_input_batch, 
            packet_input_batch,
            node_feat_input_batch,
            batch_labels,
            zda_batch_labels,
            test_zda_batch_labels,
            mode):
        """
        Don't know why, but you can have more than one sample
        per class in inference time. 
        (More than one flowstats object for a single Flow!)
        So we need to take care of carefully populating our buffers...
        Otherwise we will have bad surprises when sampling from them!!!
        (i.e. sampling more elements than those requested!)
        """
        buffers = (self.replay_buffers if mode == TRAINING else self.test_replay_buffers)

        unique_labels = torch.unique(batch_labels)

        for label in unique_labels:

            mask = batch_labels == label
            
            for sample_idx in range(flow_input_batch[mask].shape[0]):
                buffers[label.item()].push(
                    flow_state=flow_input_batch[mask][sample_idx].unsqueeze(0), 
                    packet_state=(packet_input_batch[mask][sample_idx].unsqueeze(0) if self.use_packet_feats else None),
                    node_state=(node_feat_input_batch[mask][sample_idx].unsqueeze(0) if self.use_node_feats else None),
                    label=batch_labels[mask][sample_idx].unsqueeze(0),
                    zda_label=zda_batch_labels[mask][sample_idx].unsqueeze(0),
                    test_zda_label=test_zda_batch_labels[mask][sample_idx].unsqueeze(0))

        if not self.experience_learning_allowed or not self.inference_allowed or not self.eval_allowed:

            buff_lengths = []
            for class_label, class_idx in self.encoder.get_mapping().items():

                if class_idx in buffers.keys():
                    curr_buff_len = len(buffers[class_idx]) 
                    buff_lengths.append((class_label, curr_buff_len))

            if self.AI_DEBUG: self.logger_instance.info(f'{mode} Buffer lengths: {buff_lengths}')

            if mode == TRAINING:
                self.experience_learning_allowed = torch.all(
                    torch.Tensor([buff_len  > self.replay_buff_batch_size for (_, buff_len) in buff_lengths]))        

            if mode == INFERENCE:
                self.inference_allowed = self.eval_allowed = torch.all(
                    torch.Tensor([buff_len  > self.replay_buff_batch_size for (_, buff_len) in buff_lengths]))



    def get_pushing_mask(self, zda_labels, test_zda_labels, mode):
        """
        The pushing mask is a vertical mask (i.e. a binary mask taht assings 1 or 0 to each 
        sample in the batch). The pushing mask tells us what samples are going to be saved in the 
        replay buffer. In the case of a training replay buffer, we save every input sample except those 
        that appertain to test zda labels. In the case of a test replay buffer, we omit the samples 
        that come from training zdas classes, so we take both known classes and test zdas.
        """
        if mode==TRAINING:
            return ~test_zda_labels.bool()
        else:
            known_mask = ~zda_labels.bool()
            return torch.logical_or(test_zda_labels.bool(), known_mask)


    def merge_batches(self, left_batch, rigth_batch):
        """
        Self-explainable... handles squeezing for you....
        """
        flow_features = torch.vstack([left_batch.flow_features, rigth_batch.flow_features])
        packet_features = (torch.vstack([left_batch.packet_features, rigth_batch.packet_features]) if self.use_packet_feats else None)
        node_features = (torch.vstack([left_batch.node_features, rigth_batch.node_features]) if self.use_node_feats else None)

        class_labels = torch.cat([left_batch.class_labels.squeeze(1), rigth_batch.class_labels]).unsqueeze(1)
        zda_labels = torch.cat([left_batch.zda_labels.squeeze(1), rigth_batch.zda_labels]).unsqueeze(1)
        test_zda_labels = torch.cat([left_batch.test_zda_labels.squeeze(1), rigth_batch.test_zda_labels]).unsqueeze(1)

        return Batch(
                flow_features=flow_features,
                packet_features=packet_features,
                node_features=node_features,
                class_labels=class_labels,
                zda_labels=zda_labels,
                test_zda_labels=test_zda_labels)


    def online_inference(
            self, 
            online_batch):
        
        flows_to_block = []

        self.classifier.eval()
        self.confidence_decoder.eval()

        support_batch = self.sample_from_replay_buffers(
                                samples_per_class=self.replay_buff_batch_size,
                                mode=INFERENCE)
        
        support_query_mask = self.get_canonical_query_mask(INFERENCE)
        online_query_mask = torch.ones_like(online_batch.class_labels).to(torch.bool)

        merged_query_mask = torch.cat([
                                support_query_mask, 
                                online_query_mask
                                ])
        
        accuracy_mask = torch.cat([
                                torch.zeros_like(support_query_mask), 
                                online_query_mask
                                ])

        merged_batch = self.merge_batches(support_batch, online_batch)

        logits, hidden_vectors, predicted_kernel = self.infer(
            merged_batch,
            query_mask=merged_query_mask)

        one_hot_labels = self.get_oh_labels(merged_batch, logits)
        # known class horizonal mask:
        known_class_mask = self.get_known_classes_mask(merged_batch, one_hot_labels)

        _, predicted_clusters, kr_precision = self.kernel_regression_step(
            predicted_kernel, 
            one_hot_labels,
            INFERENCE)
        
        _, ad_acc = self.zda_classification_step(
            zda_labels=merged_batch.zda_labels[merged_query_mask], 
            preds=logits[:, known_class_mask], 
            mode=INFERENCE)
        
        _, cs_acc = self.class_classification_step(merged_batch.class_labels, logits, INFERENCE, merged_query_mask)

        if self.AI_DEBUG: 
            self.logger_instance.info(f'Online {INFERENCE} AD accuracy: {ad_acc.item()} '+\
                                    f'Online {INFERENCE}  CS accuracy: {cs_acc.item()}')
            self.logger_instance.info(f'Online {INFERENCE} KR accuracy: {kr_precision}')
        
        self.classifier.train()
        self.confidence_decoder.train()

        return flows_to_block


    def process_input(self, flows, node_feats: dict = None):
        """
        """
        flows_to_block = []

        if len(flows) > 0:
            
            batch = self.assembly_input_tensor(flows, node_feats)
            
            mode=(TRAINING if random.random() > 0.3 else INFERENCE)

            to_push_mask = self.get_pushing_mask(
                batch.zda_labels, 
                batch.test_zda_labels, 
                mode)

            self.push_to_replay_buffers(
                batch.flow_features[to_push_mask], 
                (batch.packet_features[to_push_mask] if self.use_packet_feats else None),
                (batch.node_features[to_push_mask] if self.use_node_feats else None),  
                batch_labels=batch.class_labels[to_push_mask],
                zda_batch_labels=batch.zda_labels[to_push_mask],
                test_zda_batch_labels=batch.test_zda_labels[to_push_mask],
                mode=mode)
            
            if self.inference_allowed:
                flows_to_block = self.online_inference(batch)

            if self.experience_learning_allowed:
                self.experience_learning()
                self.backprop_counter += 1

        return flows_to_block        


    def sample_from_replay_buffers(self, samples_per_class, mode):
        balanced_packet_batch = None
        balanced_node_feat_batch = None

        init = True
        if mode == TRAINING:
            buffers_dict = self.replay_buffers
        elif mode == INFERENCE:
            buffers_dict = self.test_replay_buffers

        for replay_buff in buffers_dict.values():
            flow_batch, \
                packet_batch, \
                    node_feat_batch, \
                        batch_labels, \
                            zda_batch_labels, \
                                test_zda_batch_labels = replay_buff.sample(samples_per_class)
            
            if init:
                balanced_flow_batch = flow_batch
                balanced_labels = batch_labels
                balanced_zda_labels = zda_batch_labels
                balanced_test_zda_labels = test_zda_batch_labels
                if packet_batch is not None:
                    balanced_packet_batch = packet_batch
                if node_feat_batch is not None:
                    balanced_node_feat_batch = node_feat_batch

            else: 
                balanced_flow_batch = torch.vstack(
                    [balanced_flow_batch, flow_batch])
                balanced_labels = torch.vstack(
                    [balanced_labels, batch_labels])
                balanced_zda_labels = torch.vstack(
                    [balanced_zda_labels, zda_batch_labels])
                balanced_test_zda_labels = torch.vstack(
                    [balanced_test_zda_labels, test_zda_batch_labels])
                if packet_batch is not None:
                    balanced_packet_batch = torch.vstack(
                        [balanced_packet_batch, packet_batch])
                if node_feat_batch is not None:
                    balanced_node_feat_batch = torch.vstack(
                       [balanced_node_feat_batch, node_feat_batch])

            init = False

        return Batch(
            flow_features=balanced_flow_batch, 
            packet_features=balanced_packet_batch, 
            node_features=balanced_node_feat_batch,
            class_labels=balanced_labels,
            zda_labels=balanced_zda_labels,
            test_zda_labels=balanced_test_zda_labels)


    def get_canonical_query_mask(self, phase):
        """
        The query mask differentiates support from query samples. 
        Support samples are used for centroid computation.
        Query samples are used for  prototypical learning, i.e., they are assigned to each centroid based on their simmilarity in the manifold.

        This method returns a vertical mask, i.e. a one-dimentional binary mask that assigns 1 or 0 to each sample in the batch, indicating
        if it is a query sample or not (in which case it will be a support sample).

        This method assumes that the samples in the batch are concatenated in continuous slices, i.e. all the 
        samples corresponding to class A are in the first M positions, where M correspondons to the number of support + query samples for each class,
        In the positions M+1 to 2M positions, we'll have samples from class B, and so on... 
        
        So we need to mask M-K samples of each class, where K is the number of support samples for each class, 
        and this methods masks the first M-K samples of each class. 
        
        The unique difference between training and evaluation cases is that the number of known classes might be different, so 
        we take that number into account for creating the quesy mask... 
        (the mask will have dimensions N times M where N is the number of classes in the knowledge base)
        """
        if phase == TRAINING:
            class_count = self.current_training_known_classes_count
        elif phase == INFERENCE:
            class_count = self.current_test_known_classes_count

        query_mask = torch.zeros(
            size=(class_count, self.replay_buff_batch_size),
            device=self.device).to(torch.bool)
        
        # This is a trick for labelling withouth messing around with indexes.
        # We started from a bi-dimensional binary mask, all zeros, 
        # we fill with ones the last self.replay_buff_batch_size - self.K_shot positions (that correspond to the query samples)
        #  of every row (that corresponds to each class)
        query_mask[:, self.replay_buff_batch_size - self.k_shot:] = True
        # and then we flatten the di-dimensional mask to get a one-dimensional one that works for us! 
        query_mask = query_mask.view(-1)
        return query_mask


    def update_k_shot(self):
        """
        As an AI to get the k_shot
        """

        # action = self.dealer(self.current_accs, self.current_costs)
        action = 0

        self.k_shot = self.k_shot + action

        if self.k_shot < self.min_k_shot:
            self.k_shot = self.min_k_shot
        if self.k_shot > self.max_k_shot:
            self.k_shot = self.max_k_shot


    def get_oh_labels(self, batch, logits):
        """
        Create a one-hot encoding of the targets.
        """
        curr_shape=(
                batch.class_labels.shape[0],
                logits.shape[1])
        
        targets=batch.class_labels
        targets = targets.to(torch.int64)
        
        targets_onehot = torch.zeros(
            size=curr_shape,
            device=targets.device)
        targets_onehot.scatter_(1, targets.view(-1, 1), 1)

        return targets_onehot
    

    def zda_classification_step(
            self,
            zda_labels,  
            preds, 
            mode):
        """
        Params:
            zda_labels: 
                vertical mask indicating if the sample is a Zda or not.
                (1 bit for each query sample. not including support samples here) 
            preds: 
                these logit vectors have logits that indicate guessed assignation to known classes ONLY.
                note also that we have one logits vector for each query sample. (not including support samples here)
            mode:
                can be TRAINING or INFERENCE, helps to differentiate the number of classes in play...
        """

        # Binary outcomes guessing if the sample is a Zda or a sample from a known class. 
        zda_predictions = self.confidence_decoder(scores=preds)

        os_loss = self.os_criterion(
            input=zda_predictions,
            target=zda_labels)
        
        # one-hot binary labels have two positions. 
        # [1,0] -> Zda
        # [0,1] -> known stuff  
        # we make this to rapidly compute the confusion matrix in pytorch...   
        onehot_zda_labels = torch.zeros(size=(zda_labels.shape[0],2)).long()
        onehot_zda_labels.scatter_(1, zda_labels.long().view(-1, 1), 1)

        batch_os_cm = efficient_os_cm(
            preds=(zda_predictions.detach() > 0.5).long(),
            targets_onehot=onehot_zda_labels.long()
            )
        
        cummulative_os_cm = (self.training_os_cm if mode == TRAINING else self.eval_os_cm)
        cummulative_os_cm += batch_os_cm
        zda_balance = zda_labels.to(torch.float16).mean().item()
        batch_os_acc = get_balanced_accuracy(batch_os_cm, negative_weight=zda_balance)
        cummulative_os_acc = get_balanced_accuracy(cummulative_os_cm, negative_weight=0.5)

        
        if self.wbt:
            self.wbl.log({mode+'_'+OS_ACC: cummulative_os_acc.item()}, step=self.backprop_counter)
            self.wbl.log({mode+'_'+OS_LOSS: os_loss.item()}, step=self.backprop_counter)
            self.wbl.log({mode+'_'+ANOMALY_BALANCE: zda_balance}, step=self.backprop_counter)

        """
        if self.AI_DEBUG: 
            # self.logger_instance.info(f'{mode} Groundtruth Batch ZDA balance is {zda_balance:.2f}')
            # self.logger_instance.info(f'{mode} Predicted Batch ZDA balance is {zda_predictions.to(torch.float32).mean():.2f}')
            # self.logger_instance.info(f'{mode} Batch ZDA detection accuracy: {batch_os_acc:.2f}')
            # self.logger_instance.info(f'{mode} Episode ZDA detection accuracy: {cummulative_os_acc:.2f}')
        """

        return os_loss, cummulative_os_acc
    

    def kernel_regression_step(self, predicted_kernel, one_hot_labels, mode):

        if self.kernel_regression:
            
            semantic_kernel = one_hot_labels @ one_hot_labels.T

            kernel_loss = self.kr_criterion(
                baseline_kernel=semantic_kernel,
                predicted_kernel=predicted_kernel
            )

            decimal_sematic_kernel = one_hot_labels.max(1)[1].detach().numpy()
            decimal_predicted_kernel = get_clusters(predicted_kernel.detach())
            np_dec_pred_kernel = decimal_predicted_kernel.numpy()

            # Compute clustering metrics
            kr_ari = adjusted_rand_score(
                decimal_sematic_kernel, 
                np_dec_pred_kernel)
            kr_nmi = normalized_mutual_info_score(
                decimal_sematic_kernel,
                np_dec_pred_kernel)

            if self.wbt:
                self.wbl.log({mode+'_'+KR_ARI: kr_ari}, step=self.backprop_counter)
                self.wbl.log({mode+'_'+KR_NMI: kr_nmi}, step=self.backprop_counter)
                self.wbl.log({mode+'_'+KR_LOSS: kernel_loss.item()}, step=self.backprop_counter)
            """
            if self.AI_DEBUG: 
                self.logger_instance.info(f'{mode} kernel regression ARI: {kr_ari:.2f} NMI:{kr_nmi:.2f}')
                # self.logger_instance.info(f'{mode} kernel regression loss: {kernel_loss.item():.2f}')
            """
            return kernel_loss, decimal_predicted_kernel, kr_ari


    def get_known_classes_mask(self, batch, one_hot_labels):
        """
        get known class horizonal mask: An horizontal mask is a one-dimensional tensor with as many items
        as the number of NOT ZDA classes. For each class, it is telling us if it is a zda or not.  
        """
        known_oh_labels = one_hot_labels[~batch.zda_labels.squeeze(1).bool()]
        return known_oh_labels.sum(0)>0


    def experience_learning(self):
        """
        This method performs learning. (It is the only one who does so). 
        Other methods may use the neural networks, but just for inference. 
        """

        training_batch = self.sample_from_replay_buffers(
                samples_per_class=self.replay_buff_batch_size,
                mode=TRAINING)
        
        query_mask = self.get_canonical_query_mask(TRAINING)

        logits, hidden_vectors, predicted_kernel = self.infer(
            batch=training_batch,
            query_mask=query_mask)
        
        one_hot_labels = self.get_oh_labels(training_batch, logits)
        known_class_mask = self.get_known_classes_mask(training_batch, one_hot_labels)
        
        loss = 0
 
        kr_loss, predicted_clusters, _ = self.kernel_regression_step(
            predicted_kernel, 
            one_hot_labels, 
            TRAINING)
        
        if CLUSTERING_LOSS_BACKPROP: loss += kr_loss
        
        if self.multi_class:
            # we perform zda detection only when we make inferences about DIFFERENT types of attacks.
            # if instead we are on a binary attack/non attack classification setting, 
            # we do not care if the detected attacks are known or unknown.. 
            zda_detection_loss, _ = self.zda_classification_step(
                zda_labels=training_batch.zda_labels[query_mask], 
                preds=logits[:, known_class_mask], # take only the known-class logit scores  
                mode=TRAINING)
            loss += zda_detection_loss

        classification_loss, cs_acc = self.class_classification_step(
            training_batch.class_labels, 
            logits, 
            TRAINING, 
            query_mask)
        
        self.training_cs_cm += efficient_cm(
        preds=logits.detach(),
        targets_onehot=one_hot_labels[query_mask])      
        
        loss += classification_loss
            
        # Only during training we learn from feedback errors.  
        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        # update weights
        self.optimizer.step()


        if self.AI_DEBUG: 
            # self.logger_instance.info(f'{TRAINING} batch groundthruth class labels mean: {training_batch.class_labels.to(torch.float16).mean().item():.2f}')
            # self.logger_instance.info(f'{TRAINING} batch prediction class labels mean: {logits.max(1)[1].to(torch.float32).mean():.2f}')
            self.logger_instance.info(f'{TRAINING} batch multiclass classif accuracy: {cs_acc:.2f}')


        if self.backprop_counter % self.report_step_freq == 0:
            self.report(
                preds=logits[:,known_class_mask],  
                hiddens=hidden_vectors.detach(), 
                labels=training_batch.class_labels,
                predicted_clusters=predicted_clusters, 
                query_mask=query_mask,
                phase=TRAINING)

            if self.eval_allowed:
                self.evaluate_models()


    def evaluate_models(self):

        self.classifier.eval()
        self.confidence_decoder.eval()
        
        mean_eval_ad_acc = 0
        mean_eval_cs_acc = 0
        mean_eval_kr_ari = 0

        for _ in range(EVALUATION_ROUNDS):
                
            eval_batch = self.sample_from_replay_buffers(
                                    samples_per_class=self.replay_buff_batch_size,
                                    mode=INFERENCE)
            
            query_mask = self.get_canonical_query_mask(INFERENCE)

            assert query_mask.shape[0] == eval_batch.class_labels.shape[0]

            logits, hidden_vectors, predicted_kernel = self.infer(
                eval_batch,
                query_mask=query_mask)

            one_hot_labels = self.get_oh_labels(eval_batch, logits)
            known_class_h_mask = self.get_known_classes_mask(eval_batch, one_hot_labels)

            _, predicted_clusters, kr_precision = self.kernel_regression_step(
            predicted_kernel, 
            one_hot_labels,
            INFERENCE)

            _, ad_acc = self.zda_classification_step(
                zda_labels=eval_batch.zda_labels[query_mask], 
                preds=logits[:, known_class_h_mask], 
                mode=INFERENCE)

            self.eval_cs_cm += efficient_cm(
            preds=logits.detach(),
            targets_onehot=one_hot_labels[query_mask])
            
            _, cs_acc = self.class_classification_step(eval_batch.class_labels, logits, INFERENCE, query_mask)

            mean_eval_ad_acc += (ad_acc / EVALUATION_ROUNDS)
            mean_eval_cs_acc += (cs_acc / EVALUATION_ROUNDS)
            mean_eval_kr_ari += (kr_precision / EVALUATION_ROUNDS)

        if self.AI_DEBUG: 
            self.logger_instance.info(f'EVAL mean eval AD accuracy: {mean_eval_ad_acc.item():.2f} '+\
                                    f'EVAL mean eval CS accuracy: {mean_eval_cs_acc.item():.2f}')
            self.logger_instance.info(f'EVAL mean eval KR accuracy: {mean_eval_kr_ari:.2f}')
        if self.wbt:
            self.wbl.log({'Mean EVAL AD ACC': mean_eval_ad_acc.item()}, step=self.backprop_counter)
            self.wbl.log({'Mean EVAL CS ACC': mean_eval_cs_acc.item()}, step=self.backprop_counter)
            self.wbl.log({'Mean EVAL KR PREC': mean_eval_kr_ari}, step=self.backprop_counter)

        if not self.eval:
            self.check_kr_progress(curr_kr_acc=mean_eval_kr_ari)
            self.check_cs_progress(curr_cs_acc=mean_eval_cs_acc.item())
            self.check_AD_progress(curr_ad_acc=mean_eval_ad_acc.item())

        self.report(
                preds=logits[:,known_class_h_mask], 
                hiddens=hidden_vectors.detach(), 
                labels=eval_batch.class_labels,
                predicted_clusters=predicted_clusters, 
                query_mask=query_mask,
                phase=INFERENCE)

        self.classifier.train()
        self.confidence_decoder.train()


    def report(self, preds, hiddens, labels, predicted_clusters, query_mask, phase):

        if phase == TRAINING:
            cs_cm_to_plot = self.training_cs_cm
            os_cm_to_plot = self.training_os_cm
        elif phase == INFERENCE:
            cs_cm_to_plot = self.eval_cs_cm
            os_cm_to_plot = self.eval_os_cm

        if self.wbt:
            self.plot_confusion_matrix(
                mod=CLOSED_SET,
                cm=cs_cm_to_plot,
                phase=phase,
                norm=False,
                classes=self.encoder.get_labels())
            self.plot_confusion_matrix(
                mod=ANOMALY_DETECTION,
                cm=os_cm_to_plot,
                phase=phase,
                norm=False,
                classes=['Known', 'ZdA'])
            self.plot_hidden_space(hiddens=hiddens, labels=labels, predicted_labels=predicted_clusters, phase=phase)
            self.plot_scores_vectors(score_vectors=preds, labels=labels[query_mask], phase=phase)

        if self.AI_DEBUG:
            self.logger_instance.info(f'{phase} CS Conf matrix: \n {cs_cm_to_plot}')
            self.logger_instance.info(f'{phase} AD Conf matrix: \n {os_cm_to_plot}')
        
        if phase == TRAINING:
            self.reset_train_cms()
        elif phase == INFERENCE:
            self.reset_test_cms()


    def check_cs_progress(self, curr_cs_acc):
        self.best_cs_accuracy = curr_cs_acc
        self.save_cs_model()

    
    def check_AD_progress(self, curr_ad_acc):
        self.best_AD_accuracy = curr_ad_acc
        self.save_ad_model()

    
    def check_kr_progress(self, curr_kr_acc):
        self.best_KR_accuracy = curr_kr_acc
        self.save_models() 


    def save_cs_model(self, postfix='single'):
        torch.save(
            self.classifier.state_dict(), 
            self.classifier_path+postfix+'.pt')
        if self.AI_DEBUG: 
            self.logger_instance.info(f'New {postfix} flow classifier model version saved to {self.classifier_path}{postfix}.pt')


    def save_ad_model(self, postfix='single'):
        torch.save(
            self.confidence_decoder.state_dict(), 
            self.confidence_decoder_path+postfix+'.pt')
        if self.AI_DEBUG: 
            self.logger_instance.info(f'New {postfix} confidence decoder model version saved to {self.confidence_decoder_path}{postfix}.pt')


    def save_models(self):
        self.save_cs_model(postfix='coupled')
        if self.multi_class:
            self.save_ad_model(postfix='coupled')


    def class_classification_step(
            self, 
            class_labels, 
            class_predictions, 
            mode, 
            query_mask):
        """
        Class classification:
        It obviously computes the error  signal taking into account only query samples,
        Note:  
            This learning signal includes the qeury samples of train zdas and their corresponding class labels.
            This is epistemically legal, in the sense that train zdas are fake zdas.
            During testing instead, everythong is legal because we are not learning anymore, just evaluating.
        """
        cs_loss = self.cs_criterion(
            input=class_predictions,
            target=class_labels[query_mask].squeeze(1))

        # compute accuracy (inclue zda class labels for computing accuracy)
        acc = self.get_accuracy(
            logits_preds=class_predictions,
            decimal_labels=class_labels,
            query_mask=query_mask)

        # report progress
        if self.wbt:
            self.wbl.log({mode+'_'+CS_ACC: acc.item()}, step=self.backprop_counter)
            self.wbl.log({mode+'_'+CS_LOSS: cs_loss.item()}, step=self.backprop_counter)

        return cs_loss, acc
    

    def get_accuracy(self, logits_preds, decimal_labels, query_mask):
        """
        labels must not be one hot!
        """
        match_mask = logits_preds.max(1)[1] == decimal_labels.max(1)[0][query_mask]
        return match_mask.sum() / match_mask.shape[0]


    def get_labels(self, flows):

        string_labels = [flow.element_class for flow in flows]
        new_classes = self.encoder.fit(string_labels)
        for new_class in new_classes:
            self.add_class_to_knowledge_base(new_class)

        encoded_labels = self.encoder.transform(string_labels)
        zda_labels = torch.Tensor([flow.zda for flow in flows])
        test_zda_labels = torch.Tensor([flow.test_zda for flow in flows])

        return encoded_labels.to(torch.long), zda_labels, test_zda_labels
    

    def assembly_input_tensor(
            self,
            flows,
            node_feats):
        """
        Assemblies a batch from current observations. 
        A (flow) batch is composed of a set of flows. 
        Each Flow has a bidimensional feature tensor. 
        (self.MAX_FLOW_TIMESTEPS x 4 features)

        Returns a Batch object containing the corresponding features and labels.
        """

        flow_input_batch = flows[0].get_feat_tensor().unsqueeze(0)
        packet_input_batch = None
        node_feat_input_batch = None

        if self.use_packet_feats:
            packet_input_batch = flows[0].packets_tensor.buffer.unsqueeze(0)
        if self.use_node_feats:
            flows[0].node_feats = -1 * torch.ones(
                    size=(10,5),
                    device=self.device)
            
            if flows[0].dest_ip in node_feats.keys():
                flows[0].node_feats[:len(node_feats[flows[0].dest_ip][CPU]),:]  = torch.hstack([
                        torch.Tensor(node_feats[flows[0].dest_ip][CPU]).unsqueeze(1),
                        torch.Tensor(node_feats[flows[0].dest_ip][RAM]).unsqueeze(1),
                        torch.Tensor(node_feats[flows[0].dest_ip][IN_TRAFFIC]).unsqueeze(1),
                        torch.Tensor(node_feats[flows[0].dest_ip][OUT_TRAFFIC]).unsqueeze(1),
                        torch.Tensor(node_feats[flows[0].dest_ip][DELAY]).unsqueeze(1)])
            
            node_feat_input_batch = flows[0].node_feats.unsqueeze(0)

        for flow in flows[1:]:
            flow_input_batch = torch.cat( 
                [flow_input_batch,
                 flow.get_feat_tensor().unsqueeze(0)],
                 dim=0)
            if self.use_packet_feats:
                packet_input_batch = torch.cat( 
                    [packet_input_batch,
                    flow.packets_tensor.buffer.unsqueeze(0)],
                    dim=0)
            if self.use_node_feats:
                flow.node_feats = -1 * torch.ones(
                        size=(10,5),
                        device=self.device)
                if flow.dest_ip in node_feats.keys():
                    flow.node_feats[:len(node_feats[flow.dest_ip][CPU]),:] = torch.hstack([
                        torch.Tensor(node_feats[flow.dest_ip][CPU]).unsqueeze(1),
                        torch.Tensor(node_feats[flow.dest_ip][RAM]).unsqueeze(1),
                        torch.Tensor(node_feats[flow.dest_ip][IN_TRAFFIC]).unsqueeze(1),
                        torch.Tensor(node_feats[flow.dest_ip][OUT_TRAFFIC]).unsqueeze(1),
                        torch.Tensor(node_feats[flow.dest_ip][DELAY]).unsqueeze(1)]) 
                       
                node_feat_input_batch = torch.cat(
                            [node_feat_input_batch,
                            flow.node_feats.unsqueeze(0)],
                            dim=0)
                        

        batch_labels, \
                    zda_labels, \
                        test_zda_labels = self.get_labels(flows)
        
        return Batch(
            flow_features=flow_input_batch, 
            packet_features=packet_input_batch, 
            node_features=node_feat_input_batch,
            class_labels=batch_labels,
            zda_labels=zda_labels,
            test_zda_labels=test_zda_labels)
         
    

    def plot_confusion_matrix(
            self,
            mod,
            cm,
            phase,
            norm=True,
            dims=(10,10),
            classes=None):

        if norm:
            # Rapresented classes:
            rep_classes = cm.sum(1) > 0
            # Normalize
            denom = cm.sum(1).reshape(-1, 1)
            denom[~rep_classes] = 1
            cm = cm / denom
            fmt_str = ".2f"
        else:
            fmt_str = ".0f"

        # Plot heatmap using seaborn
        sns.set_theme()
        plt.figure(figsize=dims)
        ax = sns.heatmap(
            cm,
            annot=True,
            cmap='Blues',
            fmt=fmt_str,
            xticklabels=classes, 
            yticklabels=classes)

        # Rotate x-axis and y-axis labels vertically
        ax.set_xticklabels(classes, rotation=90)
        ax.set_yticklabels(classes, rotation=0)

        # Add x and y axis labels
        plt.xlabel("Predicted")
        plt.ylabel("Baseline")
        plt.title(f'{phase} Confusion Matrix')
        
        if self.wbl is not None:
            self.wbl.log({f'{phase} {mod} Confusion Matrix': wandbImage(plt)}, step=self.backprop_counter)

        plt.cla()
        plt.close()
    
    
    def plot_hidden_space(
        self,
        hiddens,
        labels, 
        predicted_labels,
        phase):

        color_iterator = itertools.cycle(colors)
        # If dimensionality is > 2, reduce using PCA
        if hiddens.shape[1]>2:
            pca = PCA(n_components=2)
            hiddens = pca.fit_transform(hiddens)

        plt.figure(figsize=(16, 6))

        # Real labels
        plt.subplot(1, 2, 1)
        # List of attacks:
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            data = hiddens[labels.squeeze(1) == label]
            p_label = self.encoder.inverse_transform(label.unsqueeze(0))[0]
            color_for_scatter = next(color_iterator)
            plt.scatter(
                data[:, 0],
                data[:, 1],
                label=p_label,
                c=color_for_scatter,
                alpha=0.5,
                s=200)
        plt.title(f'{phase} Ground-truth clusters')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # Predicted labels
        plt.subplot(1, 2, 2)
        unique_labels = torch.unique(predicted_labels)
        for label in unique_labels:
            data = hiddens[predicted_labels == label]
            color_for_scatter = next(color_iterator)
            plt.scatter(
                data[:, 0],
                data[:, 1],
                label=label.item(),
                c=color_for_scatter,
                alpha=0.5,
                s=200)
        plt.title(f'{phase} Predicted clusters')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        if self.wbl is not None:
            self.wbl.log({f"{phase} Latent Space Representations": wandbImage(plt)}, step=self.backprop_counter)

        plt.cla()
        plt.close()


    def plot_scores_vectors(
        self,
        score_vectors,
        labels,
        phase):

        # Create an iterator that cycles through the colors
        color_iterator = itertools.cycle(colors)
        
        pca = PCA(n_components=2)
        score_vectors = pca.fit_transform(score_vectors.detach())

        plt.figure(figsize=(10, 6))

        # Two plots:
        plt.subplot(1, 1, 1)
        
        # List of attacks:
        unique_labels = torch.unique(labels)

        # Print points for each attack
        for label in unique_labels:

            data = score_vectors[labels.squeeze(1) == label]
            p_label = self.encoder.inverse_transform(label.unsqueeze(0))[0]

            color_for_scatter = next(color_iterator)

            plt.scatter(
                data[:, 0],
                data[:, 1],
                label=p_label,
                c=color_for_scatter,
                alpha=0.5,
                s=200)
                
        plt.title(f'{phase} PCA reduction of association scores')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


        plt.tight_layout()
        
        if self.wbl is not None:
            self.wbl.log({f"{phase} PCA of ass. scores": wandbImage(plt)}, step=self.backprop_counter)

        plt.cla()
        plt.close()