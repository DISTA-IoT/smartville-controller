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
from smartController.replay_buffer import RawReplayBuffer, Batch
import os
import torch
import torch.optim as optim
import torch.nn as nn
from smartController.wandb_tracker import WandBTracker
import seaborn as sns
import matplotlib.pyplot as plt
import threading
from wandb import Image as wandbImage
import itertools
from sklearn.decomposition import PCA
import random
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from smartController.tiger_environment_new import NewTigerEnvironment
from smartController.tiger_agents import ValueLearningAgent, DAIAgent
from functools import wraps

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
AGENT = 'AGENT'

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


def thread_safe(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        with self._lock:
            return method(self, *method_args, **method_kwargs)
    return _impl


def epistemic_thread_safe(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        with self._epistemic_lock:
            return method(self, *method_args, **method_kwargs)
    return _impl


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
    """
    Convert a predicted adjacency matrix (predicted_kernel) into discrete clusters by assigning
    each node to a specific cluster based on the binary adjacency matrix.
    
    The function takes a predicted kernel, which is essentially a regression or a probabilistic 
    prediction of an adjacency matrix, and performs the following steps:
    
    1. Binarizes the predicted kernel by applying a threshold of 0.5, converting it into 
       a discrete adjacency matrix.
    2. Iterates over each node and checks whether it has already been assigned to a cluster.
    3. For unassigned nodes, creates a new cluster by assigning connected nodes to the same cluster.
    4. Returns a tensor with cluster labels for each node.
    
    Args:
    predicted_kernel (torch.Tensor): A 2D tensor representing a predicted or probabilistic adjacency 
                                     matrix of size (N, N), where N is the number of nodes.
    
    Returns:
    torch.Tensor: A 1D tensor of cluster labels, where each unique label represents a different cluster. 
                  The label values range from 0 to (num_clusters - 1).
    """
    
    # All fellas are in its own cluster, so we should start by adding that condition:
    # predicted_kernel = predicted_kernel + torch.eye(predicted_kernel.shape[0])

    # Binarize the predicted kernel to create a discrete adjacency matrix (0 or 1)
    discrete_predicted_kernel = (predicted_kernel > 0.5).long()
    
    # Initialize a mask to keep track of which nodes have already been assigned to clusters
    assigned_mask = torch.zeros_like(discrete_predicted_kernel.diag())
    
    # Initialize a tensor to store cluster assignments for each node
    clusters = torch.zeros_like(discrete_predicted_kernel.diag())
    
    # Cluster index starts at 1 (0 is reserved for unassigned nodes)
    curr_cluster = 1

    # Iterate over each node in the adjacency matrix
    for idx in range(discrete_predicted_kernel.shape[0]):
        # Skip nodes that have already been assigned to a cluster
        if assigned_mask[idx] > 0:
            continue

        # Create a mask for the current node's connections (its cluster)
        new_cluster_mask = discrete_predicted_kernel[idx]
        
        # Remove nodes that have already been assigned to other clusters
        new_cluster_mask = torch.relu(new_cluster_mask - assigned_mask)
        
        # Mark the nodes in the current cluster as assigned
        assigned_mask += new_cluster_mask
    
        # Assign the current cluster index to all nodes in the new cluster
        clusters += new_cluster_mask*curr_cluster

        # If any node was assigned to the new cluster, increment the cluster index
        if new_cluster_mask.sum() > 0:
            curr_cluster += 1

    # Subtract 1 from cluster labels to make cluster labels start from 0
    return clusters -1 
    

class DynamicLabelEncoder:
    """
    Thread-safe dynamic label encoder.
    
    Warning:
        All access to this class's data should be done through its methods.
        Direct dictionary access is not thread-safe.
    """
    
    def __init__(self):
        self._label_to_int = {}
        self._int_to_label = {}
        self._current_code = 0


    def fit(self, labels):
        """
        returns the number of new classes found!
        """

        # get the new labels found in the batch  
        # batch_labels  - changed labels - current labels 
        new_labels = set(labels) - set(self._label_to_int.keys())

        for label in new_labels:
            self.add_class(label)

        return new_labels


    def add_class(self, label):

        if label in self._label_to_int:
            return
        
        self._label_to_int[label] = self._current_code
        self._int_to_label[self._current_code] = label
        self._current_code += 1


    def transform(self, labels):
        
        encoded_labels = []

        for label in labels:
            encoded_labels.append(self._label_to_int[label])

        return torch.tensor(encoded_labels)


    def inverse_transform(self, encoded_labels):
        decoded_labels = [self._int_to_label[code.item()] for code in encoded_labels]
        return decoded_labels


    def get_mapping(self):
        return self._label_to_int


    def get_labels(self):
        return list(self._label_to_int.keys())
    

    def update_label(self, new_label, logger):

        # the caller needs to know if he should add a replay buffer 
        add_replay_buffer_signal = False

        if not new_label in self._label_to_int:

            logger.info(f'Proactively added {new_label}')
            self.add_class(new_label)
            add_replay_buffer_signal = True

        return add_replay_buffer_signal



class TigerBrain():

    def __init__(self,
                 eval,
                 flow_feat_dim,
                 packet_feat_dim,
                 dropout,
                 multi_class,
                 init_k_shot,
                 replay_buffer_batch_size,
                 kernel_regression,
                 device='cpu',
                 seed=777,
                 debug=False,
                 wb_track=False,
                 wb_project_name='',
                 wb_run_name='',
                 report_step_freq=50,
                 kwargs={}):
        self._lock = threading.Lock()
        self._epistemic_lock = threading.Lock()
        self.eval = eval
        self.intrusion_detection_kwargs = kwargs
        self.use_packet_feats = kwargs['use_packet_feats'] 
        self.use_node_feats = kwargs['node_features'] 
        self.flow_feat_dim = flow_feat_dim
        self.packet_feat_dim = packet_feat_dim
        self.h_dim = kwargs['h_dim']
        self.dropout = dropout
        self.multi_class = multi_class
        self.AI_DEBUG = debug
        self.step_counter = 0
        self.wbt = wb_track
        self.wbl = None
        self.kernel_regression = kernel_regression
        self.logger_instance = kwargs['logger']
        self.device=device
        self.seed = seed
        random.seed(seed)
        self.k_shot = init_k_shot
        self.replay_buff_batch_size = replay_buffer_batch_size
        self.report_step_freq = report_step_freq 
        self.use_neural_AD = kwargs['use_neural_AD']
        self.use_neural_KR = kwargs['use_neural_KR']
        self.online_evaluation = kwargs['online_evaluation']
        self.bad_classif_cost_factor =  int(kwargs['bad_classif_cost_factor'])
        self.online_eval_rounds = kwargs['online_evaluation_rounds']
        self.load_pretrained_inference_module = kwargs['pretrained_inference']
        self.clustering_loss_backprop = kwargs['clustering_loss_backprop']
        self.kernel_regressor_heads = kwargs['kernel_regressor_heads']
        self.attractive_weight = kwargs['attractive_weight']
        self.repulsive_weight = kwargs['repulsive_weight']
        self.learning_rate= float(kwargs['learning_rate'])
        self.replay_buffer_max_capacity= kwargs['replay_buffer_max_capacity']
        self.pretrained_models_dir = kwargs['pretrained_models_dir']
        self.container_ips = kwargs['container_ips']
        self.ips_containers = kwargs['ips_containers']
        self.traffic_dict = kwargs['traffic_dict']
        self.episode_count = -1
        self.env = NewTigerEnvironment(kwargs)
        self.init_agents(kwargs)
        self.init_intelligence()
        self.epistemic_agency = kwargs['epistemic_agency']
        self.save_models_flag = kwargs['save_models']

        if self.wbt:
            self.wbl = WandBTracker(
                wanb_project_name=wb_project_name,
                run_name=wb_run_name,
                config_dict=kwargs).wb_logger        

    
    def init_intelligence(self):
        self.current_known_classes_count = 0
        self.current_test_known_classes_count = 0
        self.batch_processing_allowed = False
        self.best_cs_accuracy = 0
        self.best_AD_accuracy = 0
        self.best_KR_accuracy = 0
        self.reset_train_cms()
        self.reset_test_cms()
        self.encoder = DynamicLabelEncoder()
        self.replay_buffers = {}
        self.reset_environment()


    @epistemic_thread_safe
    def reset_environment(self):
        self.env.reset()    
        self.init_inference_neural_modules(self.learning_rate, self.seed, self.intrusion_detection_kwargs)
        self.episode_count += 1
        

    def init_agents(self, kwargs):
        
        self.state_space_dim = kwargs['h_dim']
        if self.use_node_feats:
            self.state_space_dim += kwargs['h_dim']
        if self.use_packet_feats:
            self.state_space_dim += kwargs['h_dim'] 

        # The state space will be composed of          
        # 0. centroid of collective anomaly (an all-zeros centroid for known traffic)
        # 1. number of anomalies inferred in the batch
        # 2. mean confidence of anomaly inference.  
        # 3. number of known classes inferred in the batch
        # 4. mean confidence of known class classification
        # 5. current system budget 
        self.state_space_dim += 5

        agent_class = ValueLearningAgent if kwargs['agent'] in ['DQN', 'DDQN'] else DAIAgent

        kwargs['action_size'] = 3  # block, pass or TCI acquisition
        kwargs['state_size'] = self.state_space_dim # the "current_budget" scalar is part of the state space  
        
        self.mitigation_agent = agent_class(
            kwargs=kwargs)

    
    def add_replay_buffer(self, class_name):

        # take care of waiting to have a minimum quantity of samples for each new class before
        # doing prototypical learning. 
        self.batch_processing_allowed = False
        
        self.replay_buffers[self.current_known_classes_count-1] = RawReplayBuffer(
            capacity=self.replay_buffer_max_capacity,
            batch_size=self.replay_buff_batch_size,
            seed=self.seed)
        self.logger_instance.info(f'Replay buffer with code: {self.current_known_classes_count-1} for class: {class_name} was added')
    

    @thread_safe
    def add_class_to_knowledge_base(self, new_class):

        self.current_known_classes_count += 1
        
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
        

    def init_inference_neural_modules(self, lr, seed, kwargs):
        torch.manual_seed(seed)
        self.confidence_decoder = ConfidenceDecoder(device=self.device)
        self.os_criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.cs_criterion = nn.CrossEntropyLoss().to(self.device)
        self.kr_criterion = KernelRegressionLoss(repulsive_weigth=self.repulsive_weight, 
            attractive_weigth=self.attractive_weight).to(self.device)
        
        if self.use_packet_feats:
            
            if self.use_node_feats:

                self.classifier = ThreeStreamMulticlassFlowClassifier(
                    flow_input_size=self.flow_feat_dim, 
                    second_stream_input_size=self.packet_feat_dim,
                    third_stream_input_size=5,
                    hidden_size=self.h_dim,
                    kr_heads=self.kernel_regressor_heads,
                    dropout_prob=self.dropout,
                    device=self.device,
                    kwargs=kwargs)
            
            else: 

                self.classifier = TwoStreamMulticlassFlowClassifier(
                    flow_input_size=self.flow_feat_dim, 
                    second_stream_input_size=self.packet_feat_dim,
                    hidden_size=self.h_dim,
                    kr_heads=self.kernel_regressor_heads,
                    dropout_prob=self.dropout,
                    device=self.device,
                    kwargs=kwargs)

        else:
            
            if self.use_node_feats:
                
                self.classifier = TwoStreamMulticlassFlowClassifier(
                    flow_input_size=self.flow_feat_dim, 
                    second_stream_input_size=5,
                    hidden_size=self.h_dim,
                    kr_heads=self.kernel_regressor_heads,
                    dropout_prob=self.dropout,
                    device=self.device,
                    kwargs=kwargs)
            else:

                self.classifier = MultiClassFlowClassifier(
                    input_size=self.flow_feat_dim, 
                    hidden_size=self.h_dim,
                    dropout_prob=self.dropout,
                    kr_heads=self.kernel_regressor_heads,
                    device=self.device,
                    kwargs=kwargs)
            

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
                self.classifier_path = self.pretrained_models_dir+f'multiclass_flow_packet_node_classifier_pretrained_h{self.h_dim}'
                self.confidence_decoder_path = self.pretrained_models_dir+f'flow_packet_node_confidence_decoder_pretrained_h{self.h_dim}'
            else:
                self.classifier_path = self.pretrained_models_dir+f'multiclass_flow_packet_classifier_pretrained_h{self.h_dim}'
                self.confidence_decoder_path = self.pretrained_models_dir+f'flow_packet_confidence_decoder_pretrained_h{self.h_dim}'
        else:
            if self.use_node_feats:
                self.classifier_path = self.pretrained_models_dir+f'multiclass_flow_node_classifier_pretrained_h{self.h_dim}'
                self.confidence_decoder_path = self.pretrained_models_dir+f'flow_node_confidence_decoder_pretrained_h{self.h_dim}'
            else:    
                self.classifier_path = self.pretrained_models_dir+f'multiclass_flow_classifier_pretrained_h{self.h_dim}'
                self.confidence_decoder_path = self.pretrained_models_dir+f'flow_confidence_decoder_pretrained_h{self.h_dim}'

        if self.load_pretrained_inference_module:
            # Check if the file exists
            if os.path.exists(self.pretrained_models_dir):

                if os.path.exists(self.classifier_path+'.pt'):
                    # Load the pre-trained weights
                    self.classifier.load_state_dict(torch.load(self.classifier_path+'.pt', weights_only=True))
                    self.logger_instance.info(f"Pre-trained weights loaded successfully from {self.classifier_path}.pt")
                else:
                    self.logger_instance.info(f"Pre-trained weights not found at {self.classifier_path}.pt")
                    
                if self.multi_class:
                    if os.path.exists(self.confidence_decoder_path+'.pt'):
                        self.confidence_decoder.load_state_dict(torch.load(self.confidence_decoder_path+'.pt', weights_only=True))
                        self.logger_instance.info(f"Pre-trained weights loaded successfully from {self.confidence_decoder_path}.pt")
                    else:
                        self.logger_instance.info(f"Pre-trained weights not found at {self.confidence_decoder_path}.pt")             
                

            elif self.AI_DEBUG:
                self.logger_instance.info(f"Pre-trained folder not found at {self.pretrained_models_dir}.")


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
            mode):
        """
        Don't know why, but you can have more than one sample
        per class in inference time. 
        (More than one flowstats object for a single Flow!)
        So we need to take care of carefully populating our buffers...
        Otherwise we will have bad surprises when sampling from them!!!
        (i.e. sampling more elements than those requested!)
        """
        unique_labels = torch.unique(batch_labels)

        buffers = self.replay_buffers

        for label in unique_labels:

            mask = batch_labels == label
            
            for sample_idx in range(flow_input_batch[mask].shape[0]):
                try:
                    buffers[label.item()].push(
                        flow_state=flow_input_batch[mask][sample_idx].unsqueeze(0), 
                        packet_state=(packet_input_batch[mask][sample_idx].unsqueeze(0) if self.use_packet_feats else None),
                        node_state=(node_feat_input_batch[mask][sample_idx].unsqueeze(0) if self.use_node_feats else None),
                        label=batch_labels[mask][sample_idx].unsqueeze(0))
                except:
                    print('something went wrong')
                    assert 1 == 0

        if not self.batch_processing_allowed:

            buff_lengths = []
            for class_label, class_idx in self.encoder.get_mapping().items():
            
                curr_buff_len = len(buffers[class_idx]) 
                buff_lengths.append((class_label, curr_buff_len))
      
            self.batch_processing_allowed = torch.all(
                    torch.Tensor([buff_len  > self.replay_buff_batch_size for (_, buff_len) in buff_lengths]))        

            if self.AI_DEBUG: self.logger_instance.info(f'Buffer lengths: {buff_lengths}')


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
        zda_labels = torch.cat([left_batch.zda_labels.squeeze(1), rigth_batch.zda_labels.squeeze(1)]).unsqueeze(1)
        test_zda_labels = torch.cat([left_batch.test_zda_labels.squeeze(1), rigth_batch.test_zda_labels.squeeze(1)]).unsqueeze(1)

        return Batch(
                flow_features=flow_features,
                packet_features=packet_features,
                node_features=node_features,
                class_labels=class_labels,
                zda_labels=zda_labels,
                test_zda_labels=test_zda_labels)


    def preds_to_mask(self, zda_predictions):
        """
        Confidence stuff comes into play here...
        """ 
        # it turns out it wont work well if not boolean type!
        mask = (zda_predictions > 0.5).to(torch.bool)
        return mask   


    def get_zda_labels(self, batch, mode):
        """
        Get the zda labels based on natural-language labels
        - If we are in TRAINING mode, then we are doing experience learning and we should not have any G2 class.
        - If we are in INFERENCE mode, then we are doing online inference or evaluation and we should not label G1s as anomalies 
        because they are not.
        @TODO optimise  
        """

        nl_labels = self.encoder.inverse_transform(batch.class_labels)
        
        if mode == TRAINING:
            test_zda_labels = torch.zeros((len(nl_labels),1))
            zda_labels = torch.Tensor([nl_label in self.env.current_knowledge['G1s'] for nl_label in nl_labels]).unsqueeze(-1)
        elif mode == INFERENCE:
            test_zda_labels = zda_labels = torch.Tensor([nl_label in self.env.current_knowledge['G2s'] for nl_label in nl_labels]).unsqueeze(-1)

        return zda_labels, test_zda_labels


    def online_anomaly_detection(self, batch, logits, one_hot_labels, query_mask, accuracy_mask):

        if self.use_neural_AD:

            # known class horizonal mask:
            known_class_h_mask = self.get_known_classes_mask(batch, one_hot_labels)
            
            # separate between candidate known traffic and unknown traffic.
            zda_predictions = self.confidence_decoder(scores=logits[:, known_class_h_mask])

            # using the inference module to classify anomalies. 
            _, ad_acc = self.zda_classification_step(
                zda_labels=batch.zda_labels[query_mask], 
                zda_predictions=zda_predictions,
                accuracy_mask=accuracy_mask[query_mask],
                mode=INFERENCE)
            predicted_zda_mask = self.preds_to_mask(zda_predictions).squeeze(-1)
        else:
            # assuming a perfect anomaly detector (just for eval purposes, not learning from this experience tuple)
            zda_predictions = batch.zda_labels[query_mask].squeeze(-1) 
            predicted_zda_mask = zda_predictions.to(torch.bool)
            ad_acc = torch.ones(1)

        return ad_acc, zda_predictions, predicted_zda_mask
    

    def online_inference(
            self, 
            online_batch):
        """
        Does online inference with the given online batch.
        It uses some support samples from the replay buffers to aid in prototypical learning
        After the known-unknown class inferences and the clustering of unknowns, the agent has 
        performs three different actions for each cluster:
        0. let it pass          (practic action)
        1. block it             (practic action)
        2. acquire a TCI label  (epistemic action)
        """
        
        # do not run gradients on the inference modules!
        self.classifier.eval()
        self.confidence_decoder.eval()

    
        # get zda labels for the online batch
        online_batch.zda_labels, online_batch.test_zda_labels = self.get_zda_labels(online_batch, mode=INFERENCE)

        # sample from the replay buffers
        aux_batch = self.sample_from_replay_buffers(
                                samples_per_class=self.replay_buff_batch_size,
                                mode=INFERENCE)
        
        # query masks
        aux_query_mask = self.get_canonical_query_mask(aux_batch.class_labels.shape[0])
        online_query_mask = torch.ones_like(online_batch.class_labels).to(torch.bool)
        merged_query_mask = torch.cat([aux_query_mask, online_query_mask])
        
        # we report and compute rewards only over online samples
        accuracy_mask = torch.cat([torch.zeros_like(aux_query_mask), online_query_mask])

        merged_batch = self.merge_batches(aux_batch, online_batch)

        # inference
        logits, hidden_vectors, predicted_kernel = self.infer(
            merged_batch,
            query_mask=merged_query_mask)           
        
        one_hot_labels = self.get_oh_labels(merged_batch, logits.shape[1])

        # 
        # (Individual) Anomaly detection:   
        # 
        ad_acc, zda_predictions, predicted_zda_mask = self.online_anomaly_detection(
            batch=merged_batch,
            logits=logits,
            one_hot_labels=one_hot_labels,
            query_mask=merged_query_mask,
            accuracy_mask=accuracy_mask
        )
        
        
        # We needed the aux samples to perform inference in the prototypical way, but
        # actually, we only care about online anomalies:
        # so we compute how many samples we had in the online batch: 
        num_of_online_samples = online_batch.zda_labels.shape[0] 
        # and we know those were in the last positions of the merged batch: 
        predicted_online_zda_mask = predicted_zda_mask[-num_of_online_samples:]  
          
        #  
        # CLASSIFICATION OF KNOWN TRAFFIC: 
        # rewards need to be given if classification is good, (as if we were accepting/blocking stuff...)
        # notice we select only traffic classified as known using the inverse of the anomaly-classified mask
        # i.e., the ~predicted_online_zda_mask mask
        #
        online_class_labels = merged_batch.class_labels[-num_of_online_samples:][~predicted_online_zda_mask].squeeze(-1)
        online_class_preds = logits[-num_of_online_samples:][~predicted_online_zda_mask].max(1)[1]
        known_correct_classification_mask = online_class_labels == online_class_preds
 
        # classif. accuracy (only for reporting purposes) 
        cs_acc = known_correct_classification_mask.sum() / known_correct_classification_mask.shape[0] 
        if self.wbt: self.wbl.log({INFERENCE+'_'+CS_ACC: cs_acc.item()}, step=self.step_counter)

        # how many online samples are we classifying as known? 
        number_of_predicted_known_samples = (~predicted_online_zda_mask).sum()

        # how many of them are we classifying as anomalies instead? 
        num_of_predicted_anomalies = predicted_online_zda_mask.sum()

        #
        # Computing confidence on known-class classification:
        # take the polarisation-degree of your inferences. 
        #  
        # analysing only the logits of samples predicted as known: 
        interest_logits_slice = logits[-num_of_online_samples:][~predicted_online_zda_mask]
        number_of_known_classes = logits.shape[1]


        # 
        # CONFIDENCE MEASUREMENT:
        # 
        # Conf. of multiclass classification: 
        # we compute the ratio of the average max logits with those of the other logits:  
        non_choosed_mask = torch.ones(number_of_predicted_known_samples, number_of_known_classes)
        non_choosed_mask[torch.arange(number_of_predicted_known_samples), online_class_preds] = 0 
        mean_non_choosed_values = interest_logits_slice[non_choosed_mask.to(torch.bool)].mean()
        mean_choosed_logits = interest_logits_slice.max(1)[0].mean()
        known_classification_confidence = torch.log(mean_choosed_logits / mean_non_choosed_values)

        # Conf. of anomaly detection:
        # We say you're an anomaly if the logit is greater or equals than 0.5
        # but how well polarized were these logits?
        # we take the mean closeness to zero of non-anomaly logits and the mean closeness to 1 of anomaly logits
        online_anomaly_logits =  zda_predictions[-num_of_online_samples:]
        online_non_anomaly_pred_logits = online_anomaly_logits[~predicted_online_zda_mask]
        online_anomaly_pred_logits = online_anomaly_logits[predicted_online_zda_mask]
        
        zda_confidence = 0
        conf_normalizer = 0
        if online_non_anomaly_pred_logits.shape[0] > 0: 
            # we do think there are some known samples:
            zda_confidence += (1 - online_non_anomaly_pred_logits).mean()
            conf_normalizer += 1
        if  online_anomaly_pred_logits.shape[0] > 0:
            # we also think there are anomalies:
            zda_confidence += online_anomaly_pred_logits.mean()
            conf_normalizer += 1
        zda_confidence /= conf_normalizer

                
        # the reward obtained in this batch 
        batch_reward = torch.zeros(1)

        # natural language labels
        nl_labels = self.encoder.inverse_transform(
            merged_batch.class_labels[-num_of_online_samples:])    

        # sample-specific rewards:
        sample_rewards = torch.Tensor([self.env.flow_rewards_dict[label] for label in nl_labels])

        #
        # Acting over the known traffic:
        # One state vector is assembled which has a zeros centroid. 
        # Actions do not change anything in the system, apart of the rewards: 
        # If you accept it, you'll get the reward based on the classification accuracy. 
        # For the other actions, no reward is given.

        action_signal, state_vecs = self.act( 
            torch.zeros(1, hidden_vectors.shape[1]),
            num_of_predicted_anomalies,
            zda_confidence,
            number_of_predicted_known_samples,
            known_classification_confidence,
            self.env.current_budget
            )
        
        # Computing the rewads for known traffic: 
        known_samples_costs = sample_rewards[~predicted_online_zda_mask]

        classification_reward = 0
        
        correct_classif_rewards = torch.zeros_like(known_samples_costs)
        bad_classif_costs = torch.zeros_like(known_samples_costs)

        if action_signal.item() == 0:
            # Each well classified sample is rewarded positively:        
            correct_classif_rewards = torch.abs(known_samples_costs * known_correct_classification_mask)
            # Incorrectly classified stuff has a cost: 
            if self.intrusion_detection_kwargs['bad_classif_penalisation'] == 'easy':
                bad_classif_costs = -torch.abs(known_samples_costs * (~known_correct_classification_mask))
            else:
                bad_classif_costs = -torch.abs(known_samples_costs * (~known_correct_classification_mask) * self.bad_classif_cost_factor)
            # total classification reward:  
            classification_reward = (correct_classif_rewards.sum() + bad_classif_costs.sum()).item()

        # update the current budget 
        self.env.current_budget += classification_reward
        
        # the new state is a copy of the old one: 
        new_state = state_vecs[0].detach().clone()
        # but changing the current budget: 
        new_state[-1] = self.env.current_budget 

        # an episode ends if the budget ends... 
        end_signal = self.env.has_episode_ended(self.step_counter)

        # store the experience tuple:          
        self.mitigation_agent.remember(
            state_vecs[0].detach(),
            action_signal,
            classification_reward,
            new_state,
            end_signal
        )
        
        # add the good classification reward to the batch reward (only for reporting purposes)
        batch_reward += classification_reward
        
        #  
        # COLLECTIVE anomaly detection (i.e., clustering eventual zdas) 
        # 

        kr_precision = torch.ones(1)

        purchased_mask = torch.zeros(1)

        cluster_action_signals = torch.zeros(1)

        bought_labels = []

        rewards_per_accepted_clusters = torch.zeros(1)
        benign_blocking_cost_per_cluster = torch.zeros(1)
        epistemic_costs = torch.zeros(1)

        if num_of_predicted_anomalies > 0:
            # Anomaly clustering is going to be done only if there are predicted anomalies.
            print('num of predicted anomalies: ', num_of_predicted_anomalies)
            if self.use_neural_KR:
                # use inference modules for clustering...
                _, predicted_decimal_clusters, kr_precision = self.kernel_regression_step(
                    predicted_kernel[-num_of_online_samples:][:,-num_of_online_samples:], 
                    one_hot_labels[-num_of_online_samples:],
                    INFERENCE)
            else:
                # or assume instead a perfect clusterer
                predicted_decimal_clusters = merged_batch.class_labels[-num_of_online_samples:].squeeze(1)
                
            # how many clusters have we identified? (useful for one-hot encoding)
            num_of_predicted_clusters = predicted_decimal_clusters.max() + 1

            # take only the clusters of predicted zdas...
            anomalous_predicted_decimal_clusters = predicted_decimal_clusters[predicted_online_zda_mask] 
                
            # one-hot encode the predicted clusters (don't worry about exact class assignments, we just need to group stuff...)
            predicted_clusters_oh = torch.nn.functional.one_hot(
                anomalous_predicted_decimal_clusters,
                num_classes=num_of_predicted_clusters
            )

            # get latent centroids
            centroids, missing_clusters = self.get_centroids(
                hidden_vectors[-num_of_online_samples:][predicted_online_zda_mask], 
                predicted_clusters_oh.to(torch.float32))

            # decide if blocking or accepting each unknown... 
            cluster_action_signals, state_vecs = self.act( 
                centroids[~missing_clusters],
                num_of_predicted_anomalies,
                zda_confidence,
                number_of_predicted_known_samples,
                known_classification_confidence,
                self.env.current_budget)
            
            rewards_per_cluster = torch.zeros_like(cluster_action_signals).float()
            epistemic_costs = torch.zeros_like(cluster_action_signals).float()
            # get the potential rewards per cluster 
            # This LOC takes into account every sample, and computes the reward for ACCEPTING each cluster as is.
            # Notice the reward takes into account intersections with good and bad samples
            rewards_per_accepted_clusters = (predicted_clusters_oh * sample_rewards[predicted_online_zda_mask].unsqueeze(-1)).sum(0)

            # get the cluster_passing_mask:
            cluster_passing_mask = cluster_action_signals == 0

            # get the cluster-specific passing rewards
            rewards_per_accepted_clusters = rewards_per_accepted_clusters[~missing_clusters] *  cluster_passing_mask

            rewards_per_cluster += rewards_per_accepted_clusters

            # benign traffic mask:
            benign_rewards = torch.relu(sample_rewards[predicted_online_zda_mask]) 

            # potential benign rewards in each cluster  (this is gonna be useful to penalize blocking good stuff)
            benign_rewards_per_cluster = (predicted_clusters_oh * benign_rewards.unsqueeze(-1)).sum(0)

            # blocked benign traffic implies to pay a cost:
            benign_blocking_cost_per_cluster = self.bad_classif_cost_factor * benign_rewards_per_cluster[~missing_clusters] * (1 - cluster_passing_mask.to(torch.long)) 

            # subtract the price of neglecting benign traffic
            rewards_per_cluster -= benign_blocking_cost_per_cluster

            # get the epistemic action mask:
            purchased_mask = cluster_action_signals == 2 

            # perform epistemic action 
            for epistemic_action_index in purchased_mask.nonzero():
            
                # get information about the change in the curriculum
                updates_dict = self.perform_epistemic_action()
                # you'll pay the price of aqcuiring a TCI label:
                epistemic_costs[epistemic_action_index] -= updates_dict['price_payed']
                rewards_per_cluster[epistemic_action_index] -= updates_dict['price_payed']
                bought_labels.append(updates_dict['updated_label'])
            
            # update the batch reward with pure clustering rewards 
            batch_reward += rewards_per_cluster.sum().item()   

            # update the current budget
            self.env.current_budget += rewards_per_cluster.sum().item()

            # ask again if the budget is over: 
            end_signal = self.env.has_episode_ended(self.step_counter)

            # broadcast the end signal for attaching to centroids: 
            broadcasted_end_signal = torch.Tensor([end_signal] * cluster_action_signals.shape[0])

            # the state is going to be assemled using the new budget.
            broadcasted_new_budget = torch.Tensor([self.env.current_budget] * cluster_action_signals.shape[0] ) 
            
            # we can approximate the new state with the previous one, but changing the budget. 
            next_states = torch.hstack(
                [state_vecs[:, :-1], 
                    broadcasted_new_budget.unsqueeze(-1)]
                    )

            # collect experience tuples for training: 
            for experience_tuple in zip(
                    state_vecs, 
                    cluster_action_signals,
                    rewards_per_cluster,
                    next_states,
                    broadcasted_end_signal):
                self.mitigation_agent.remember(*experience_tuple)
        

        # train!
        self.mitigation_agent.replay()     

        if self.AI_DEBUG: 
            self.logger_instance.info(f'\nOnline {INFERENCE} AD accuracy: {ad_acc.item()} \n'+\
                                        f'Online {INFERENCE} CS accuracy: {cs_acc.item()} \n'+\
                                        f'Online {INFERENCE} batch reward: {batch_reward.item()} \n'+\
                                        f'Online {INFERENCE} current budget: {self.env.current_budget} \n'+\
                                        f'Online {INFERENCE} KR accuracy: {kr_precision}')
        
        if self.wbt:
            self.wbl.log({AGENT+'_'+'reward': batch_reward.item()}, step=self.step_counter)
            self.wbl.log({AGENT+'_'+'budget': self.env.current_budget}, step=self.step_counter)
            self.wbl.log({'Epistemic Actions taken': (1 if purchased_mask.sum() > 0 else 0)}, step=self.step_counter)
            self.wbl.log({'known traffic action': action_signal.item(),
                          'cluster actions': cluster_action_signals.tolist(),
                          'classification_reward': classification_reward,
                          'real_num_of_anomalies': online_batch.zda_labels.sum().item(),
                          'correct_classification_rewards': correct_classif_rewards.sum().item(),
                          'bad_classification_cost': bad_classif_costs.sum().item(),
                          'rewards_per_accepted_clusters':rewards_per_accepted_clusters.sum().item(),
                          'rewards_per_blocked_clusters':-benign_blocking_cost_per_cluster.sum().item(),
                          'num_predicted_knowns': number_of_predicted_known_samples.item(),
                          'num_predicted_unknowns': num_of_predicted_anomalies.item(),
                          'known_classif_confidente': known_classification_confidence.item(),
                          'zda_classif_confidence': zda_confidence.item(),
                          'epistemic_costs': epistemic_costs.sum().item(),
                          }, 
                          step=self.step_counter)
        # re-activate gradient tracking on inference modules: 
        self.classifier.train()
        self.confidence_decoder.train()

        # eventually reset the environment. 
        if end_signal: 
            self.reset_environment()
            if self.wbt:
                self.wbl.log({'episode_count': self.episode_count}, step=self.step_counter)
            if self.episode_count % 5 == 0:
                self.mitigation_agent.train_actor()


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
            This learning signal includes the query samples of train zdas and their corresponding class labels.
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
            self.wbl.log({mode+'_'+CS_ACC: acc.item()}, step=self.step_counter)
            self.wbl.log({mode+'_'+CS_LOSS: cs_loss.item()}, step=self.step_counter)

        return cs_loss, acc


    def get_centroids(
            self,
            hidden_vectors,
            onehot_labels):
        """
        Compute the centroids (cluster centers) for a set of hidden representation vectors, 
        based on their one-hot encoded class assignments. This method handles cases where 
        some clusters may not have any samples in the batch (missing clusters).
        
        Args:
        hidden_vectors (torch.Tensor): A 2D tensor of shape (N, D), where N is the number 
                                    of samples and D is the dimensionality of the hidden representations.
        onehot_labels (torch.Tensor): A 2D tensor of shape (N, K), where N is the number 
                                    of samples and K is the number of classes. Each row is 
                                    a one-hot encoded label indicating the class assignment 
                                    for each sample.
        
        Returns:
        tuple: A tuple containing:
            - centroids (torch.Tensor): A 2D tensor of shape (K, D) representing the centroids 
                                        for each class. If a class has no samples in the batch, 
                                        its centroid will remain as zeros.
            - missing_clusters (torch.Tensor): A 1D boolean tensor of length K, where True 
                                            indicates that a class is missing (i.e., has 
                                            no samples in the batch), and False means that 
                                            the class has at least one sample.
        """
        
        # Perform matrix multiplication to aggregate hidden vectors by class (onehot_labels.T @ hidden_vectors)
        # This step sums the hidden_vectors for all samples belonging to each class.
        cluster_agg = onehot_labels.T @ hidden_vectors

        # Compute the number of samples for each class by summing over the one-hot encoded labels.
        samples_per_cluster = onehot_labels.sum(0)
        
        # Initialize centroids as a zero tensor of the same shape as the aggregated hidden vectors.
        centroids = torch.zeros_like(cluster_agg, device=self.device)
        
        # Identify missing clusters, i.e., classes with zero samples.
        missing_clusters = samples_per_cluster == 0

        # For the clusters that have samples, compute the centroid by dividing the summed hidden vectors
        # by the number of samples in each cluster.
        existent_centroids = cluster_agg[~missing_clusters] / samples_per_cluster[~missing_clusters].unsqueeze(-1)
        
        # Assign the computed centroids to their corresponding positions in the centroids tensor.
        centroids[~missing_clusters] = existent_centroids

        return centroids, missing_clusters

   
    def act(
            self, 
            centroids, 
            num_of_anomalies,
            zda_confidence,
            number_of_known_samples_in_batch,
            known_classification_confidence,
            curr_budget):
        
        # the state vectors are gonna be composed of the hidden centroids + broadcasted scalars.
         
        broadcasted_num_of_anomalies = torch.Tensor([num_of_anomalies] * centroids.shape[0])
        broadcasted_zda_confidence = torch.Tensor([zda_confidence] * centroids.shape[0])
        bc_num_of_known_samples = torch.Tensor([number_of_known_samples_in_batch] * centroids.shape[0])
        bc_classif_conf = torch.Tensor([known_classification_confidence] * centroids.shape[0])
        broadcasted_budget = torch.Tensor([curr_budget] * centroids.shape[0])

        # get state vectors
        state_vecs = torch.hstack(
           [centroids, 
            broadcasted_num_of_anomalies.unsqueeze(-1),
            broadcasted_zda_confidence.unsqueeze(-1),
            bc_num_of_known_samples.unsqueeze(-1),
            bc_classif_conf.unsqueeze(-1),
            broadcasted_budget.unsqueeze(-1)]
            )

        action_per_cluster = [] 
        for state_vec in state_vecs:
            action_per_cluster.append(self.mitigation_agent.act(state_vec))

        return torch.Tensor(action_per_cluster).to(torch.long), state_vecs.detach()


    def process_input(self, flows, node_feats: dict = None):
        """
        """
        self.step_counter += 1

        if len(flows) > 0:
            
            batch = self.assembly_input_tensor(flows, node_feats)
            mode=(TRAINING if random.random() > 0.4 else INFERENCE)
  
            self.push_to_replay_buffers(
                batch.flow_features, 
                (batch.packet_features if self.use_packet_feats else None),
                (batch.node_features if self.use_node_feats else None),  
                batch_labels=batch.class_labels,
                mode=mode)

            # this fella could be toogling because of a new class arriving... 
            with self._lock:
                if self.batch_processing_allowed and self.epistemic_agency:
                    self.online_inference(batch)
            with self._lock:
                if self.batch_processing_allowed:
                    self.experience_learning()



    def sample_from_replay_buffers(self, samples_per_class, mode):
        balanced_packet_batch = None
        balanced_node_feat_batch = None

        init = True
  
        classes_decimal_tensor = torch.Tensor(list(self.replay_buffers.keys())).to(torch.long)
        nl_labels = self.encoder.inverse_transform(classes_decimal_tensor)
        
        for replay_buff, class_nl_label  in zip(self.replay_buffers.values(), nl_labels):

            test_zda_batch_labels = zda_batch_labels = torch.zeros(samples_per_class, 1)

            if mode== TRAINING:
                # we cannot use test-time anomalies for backproping gradients on our inference module, (by definition) 
                if class_nl_label in self.env.current_knowledge['G2s']:
                    continue
                # we use fake-anomalies, (i.e. clusters we know but that we can label as anomalies to teach the inference
                # module to learn the cluster separation distribution and be able to detect OOD test-time anomalies or G2 classes) 
                if class_nl_label in self.env.current_knowledge['G1s']:
                    zda_batch_labels = torch.ones(samples_per_class, 1)
            
            # in inference, we sample from the replay buffers to build an auxiliary batch for classification.
            # the auxiliary batch is combined with the online batch, which can potentially contain any record, 
            # (also train-time or fake-anomalies, i.e. G1 classes) for this reason, the aux batch contains also 
            # every class, but we do not treat G1 as anomalies anymore, as we are not interested on making inferences
            # on their benign/malicious nature, as we already known what they are, so the unique anomalies here are the eventual G2s
            # that are still-to-buy as CTI 
            if mode == INFERENCE:
                if class_nl_label in self.env.current_knowledge['G2s']:
                    test_zda_batch_labels = zda_batch_labels = torch.ones(samples_per_class, 1)
            
            flow_batch, \
                packet_batch, \
                    node_feat_batch, \
                        batch_labels = replay_buff.sample(samples_per_class)
            
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


    def get_canonical_query_mask(self, whole_batch_size):
        """
        The query mask differentiates support from query samples. 
        Support samples are used for centroid computation.
        Query samples are used for  prototypical learning, i.e., they are assigned to each centroid based on their simmilarity in the manifold.

        This method returns a vertical mask, i.e. a one-dimentional binary mask that assigns 1 (True) or 0 (False) to each sample in the batch, indicating
        if it is a query sample (1) or not (0) (in which case it will be a support sample).

        This method assumes that the samples in the batch are concatenated in continuous slices, i.e. all the 
        samples corresponding to class A are in the first M positions, where M correspondons to the number of support + query samples for each class,
        In the positions M+1 to 2M positions, we'll have samples from class B, and so on... 
        
        We mask-out (i.e. put zeros on) K samples of each class, where K is the number of support samples for each class, AKA the K-shot parameter. 
        and this methods masks the first K samples of each class. 
        

        (the mask will have dimensions N times M, where N is the number of classes in the knowledge base)
        """

        N = whole_batch_size // self.replay_buff_batch_size
        M = self.replay_buff_batch_size

        query_mask = torch.ones(
            size=(N, M),
            device=self.device).to(torch.bool)
        
        # This is a trick for labelling withouth messing around with indexes.
        # We started from a bi-dimensional binary mask, all zeros, 
        # we mask out the first self.K_shot positions (that correspond to the query samples)
        #  of every row (that corresponds to each class)
        query_mask[:, :self.k_shot] = False
        # and then we flatten the bi-dimensional mask to get a one-dimensional one that works for us! 
        query_mask = query_mask.view(-1)
        return query_mask


    def get_oh_labels(self, batch, class_shape):
        """
        Create a one-hot encoding of the targets.
        """
        curr_shape=(
                batch.class_labels.shape[0],
                class_shape)
        
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
            zda_predictions,
            accuracy_mask,
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

        os_loss = self.os_criterion(
            input=zda_predictions[accuracy_mask],
            target=zda_labels[accuracy_mask])
        
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
            self.wbl.log({mode+'_'+OS_ACC: cummulative_os_acc.item()}, step=self.step_counter)
            self.wbl.log({mode+'_'+OS_LOSS: os_loss.item()}, step=self.step_counter)
            self.wbl.log({mode+'_'+ANOMALY_BALANCE: zda_balance}, step=self.step_counter)

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
                self.wbl.log({mode+'_'+KR_ARI: kr_ari}, step=self.step_counter)
                self.wbl.log({mode+'_'+KR_NMI: kr_nmi}, step=self.step_counter)
                self.wbl.log({mode+'_'+KR_LOSS: kernel_loss.item()}, step=self.step_counter)
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
        
        # get zda labels for the online batch
        training_batch.zda_labels, training_batch.test_zda_labels = self.get_zda_labels(training_batch, mode=TRAINING)
        
        query_mask = self.get_canonical_query_mask(training_batch.class_labels.shape[0])

        logits, hidden_vectors, predicted_kernel = self.infer(
            batch=training_batch,
            query_mask=query_mask)
        
        one_hot_labels = self.get_oh_labels(training_batch, logits.shape[1])
        # known class horizonal mask:
        known_class_h_mask = self.get_known_classes_mask(training_batch, one_hot_labels)
        
        loss = 0

        # separate between candidate known traffic and unknown traffic.
        zda_predictions = self.confidence_decoder(scores=logits[:, known_class_h_mask])
        
        if self.multi_class:
            # we perform zda detection only when we make inferences about DIFFERENT types of attacks.
            # if instead we are on a binary attack/non attack classification setting, 
            # we do not care if the detected attacks are known or unknown.. 
            zda_detection_loss, _ = self.zda_classification_step(
                zda_labels=training_batch.zda_labels[query_mask], 
                zda_predictions=zda_predictions,
                accuracy_mask=torch.ones(query_mask.sum()).to(torch.bool),
                mode=TRAINING)
            loss += zda_detection_loss

        # clusterise everything you have labels about. 
        kr_loss, predicted_clusters, _ = self.kernel_regression_step(
            predicted_kernel, 
            one_hot_labels, 
            TRAINING)
        
        if self.clustering_loss_backprop: loss += kr_loss
        
        # This helps to converge 
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


        if self.step_counter % self.report_step_freq == 0:
            self.report(
                preds=logits[:,known_class_h_mask],  
                hiddens=hidden_vectors.detach(), 
                labels=training_batch.class_labels,
                predicted_clusters=predicted_clusters, 
                query_mask=query_mask,
                phase=TRAINING)

            # Update the target value network in the mitigation agent! 
            self.mitigation_agent.update_target_model()
            
            if self.online_evaluation:
                self.evaluate_models()
            

    @epistemic_thread_safe 
    def perform_epistemic_action(self, current_action=0):      
        
        updates_dict = self.env.perform_epistemic_action(current_action)

        new_label = updates_dict['updated_label']
        
        if new_label is not None:
            
            self.logger_instance.info(f'Bought label {new_label}')
            
            add_replay_buff = self.encoder.update_label(
                new_label=new_label,
                logger=self.logger_instance
            )

            if add_replay_buff:
                if self.AI_DEBUG:
                    self.logger_instance.info(f'label {new_label} bought proactively!')
                # we cant invoke this function here because it would be a deadlock 
                # self.add_class_to_knowledge_base(updates_dict['new_label'])
                # fo we repeat the code: 
                
                self.current_known_classes_count += 1
                
                self.add_replay_buffer(new_label)
                self.reset_train_cms()
                self.reset_test_cms()

        return updates_dict
    

    def evaluate_models(self):

        self.classifier.eval()
        self.confidence_decoder.eval()
        
        mean_eval_ad_acc = 0
        mean_eval_cs_acc = 0
        mean_eval_kr_ari = 0

        for _ in range(self.online_eval_rounds):
                
            eval_batch = self.sample_from_replay_buffers(
                                    samples_per_class=self.replay_buff_batch_size,
                                    mode=INFERENCE)
            
            query_mask = self.get_canonical_query_mask(eval_batch.class_labels.shape[0])

            assert query_mask.shape[0] == eval_batch.class_labels.shape[0]

            logits, hidden_vectors, predicted_kernel = self.infer(
                eval_batch,
                query_mask=query_mask)

            one_hot_labels = self.get_oh_labels(eval_batch, logits.shape[1])
            # known class horizonal mask:
            known_class_h_mask = self.get_known_classes_mask(eval_batch, one_hot_labels)

            # separate between candidate known traffic and unknown traffic.
            zda_predictions = self.confidence_decoder(scores=logits[:, known_class_h_mask])

            _, ad_acc = self.zda_classification_step(
                zda_labels=eval_batch.zda_labels[query_mask], 
                zda_predictions=zda_predictions,
                accuracy_mask=torch.ones(query_mask.sum()).to(torch.bool),
                mode=INFERENCE)
            
            _, predicted_clusters, kr_precision = self.kernel_regression_step(
                predicted_kernel, 
                one_hot_labels,
                INFERENCE)         

            self.eval_cs_cm += efficient_cm(
                preds=logits.detach(),
                targets_onehot=one_hot_labels[query_mask])
            
            _, cs_acc = self.class_classification_step(eval_batch.class_labels, logits, INFERENCE, query_mask)

            mean_eval_ad_acc += (ad_acc / self.online_eval_rounds)
            mean_eval_cs_acc += (cs_acc / self.online_eval_rounds)
            mean_eval_kr_ari += (kr_precision / self.online_eval_rounds)

        if self.AI_DEBUG: 
            self.logger_instance.info(f'\n EVAL mean eval AD accuracy: {mean_eval_ad_acc.item():.2f} \n'+\
                                        f'EVAL mean eval CS accuracy: {mean_eval_cs_acc.item():.2f} \n' +\
                                        f'EVAL mean eval KR accuracy: {mean_eval_kr_ari:.2f}')
        if self.wbt:
            self.wbl.log({'Mean EVAL AD ACC': mean_eval_ad_acc.item()}, step=self.step_counter)
            self.wbl.log({'Mean EVAL CS ACC': mean_eval_cs_acc.item()}, step=self.step_counter)
            self.wbl.log({'Mean EVAL KR PREC': mean_eval_kr_ari}, step=self.step_counter)

        if self.save_models_flag:
            self.check_progress(
                curr_cs_acc=mean_eval_cs_acc.item(),
                curr_ad_acc=mean_eval_ad_acc.item(),
                curr_kr_acc=mean_eval_kr_ari)

        """
        if not self.eval:
            self.check_kr_progress(curr_kr_acc=mean_eval_kr_ari)
            self.check_cs_progress(curr_cs_acc=mean_eval_cs_acc.item())
            self.check_AD_progress(curr_ad_acc=mean_eval_ad_acc.item())
        """

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

        """
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
        """

        if self.AI_DEBUG:
            self.logger_instance.info(f'{phase} CS Conf matrix: \n {cs_cm_to_plot}')
            self.logger_instance.info(f'{phase} AD Conf matrix: \n {os_cm_to_plot}')
        
        if phase == TRAINING:
            self.reset_train_cms()
        elif phase == INFERENCE:
            self.reset_test_cms()



    def check_progress(self, curr_cs_acc, curr_ad_acc, curr_kr_acc):
        self.check_cs_progress(curr_cs_acc)
        self.check_AD_progress(curr_ad_acc)
        self.check_kr_progress(curr_kr_acc)


    def check_cs_progress(self, curr_cs_acc):
        if curr_cs_acc > self.best_cs_accuracy:
            self.best_cs_accuracy = curr_cs_acc
            self.save_cs_model()

    
    def check_AD_progress(self, curr_ad_acc):
        if curr_ad_acc > self.best_AD_accuracy:
            self.best_AD_accuracy = curr_ad_acc
            self.save_ad_model()

    
    def check_kr_progress(self, curr_kr_acc):
        if curr_kr_acc > self.best_KR_accuracy:
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

        return encoded_labels.to(torch.long)
    

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
                        

        batch_labels = self.get_labels(flows)
        
        return Batch(
            flow_features=flow_input_batch, 
            packet_features=packet_input_batch, 
            node_features=node_feat_input_batch,
            class_labels=batch_labels)
         
    

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
            self.wbl.log({f'{phase} {mod} Confusion Matrix': wandbImage(plt)}, step=self.step_counter)

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
            self.wbl.log({f"{phase} Latent Space Representations": wandbImage(plt)}, step=self.step_counter)

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
            self.wbl.log({f"{phase} PCA of ass. scores": wandbImage(plt)}, step=self.step_counter)

        plt.cla()
        plt.close()