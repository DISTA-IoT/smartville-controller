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

import torch

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

# Domain constants
RAM = 'RAM'
CPU = 'CPU'
INBOUND = 'INBOUND'
OUTBOUND = 'OUTBOUND'
RTT = 'RTT'
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

# Model class names
CONFIDENCE_DECODER_CLASS_NAME = 'ConfidenceDecoder'
KERNEL_REGRESSION_LOSS_CLASS_NAME = 'KernelRegressionLoss'
ONE_STREAM_MULTICLASS_FLOW_CLASSIFIER_CLASS_NAME = 'OneStreamMulticlassFlowClassifier'
TWO_STREAM_MULTICLASS_FLOW_CLASSIFIER_CLASS_NAME = 'TwoStreamMulticlassFlowClassifier'
THREE_STREAM_MULTICLASS_FLOW_CLASSIFIER_CLASS_NAME = 'ThreeStreamMulticlassFlowClassifier'

def efficient_cm(preds, targets_onehot):
    """
    Compute a multi-class confusion matrix efficiently using matrix multiplication.
    """
    predictions_decimal = preds.argmax(dim=1).to(torch.int64)
    predictions_onehot = torch.zeros_like(
        preds,
        device=preds.device)
    predictions_onehot.scatter_(1, predictions_decimal.view(-1, 1), 1)

    return targets_onehot.T @ predictions_onehot


def efficient_os_cm(preds, targets_onehot):
    """
    Compute an anomaly detection (binary) confusion matrix efficiently.
    """
    predictions_onehot = torch.zeros(
        [preds.size(0), 2],
        device=preds.device)
    predictions_onehot.scatter_(1, preds.view(-1, 1), 1)

    return targets_onehot.T @ predictions_onehot.long()


def get_balanced_accuracy(os_cm, negative_weight):
    """
    Calculate balanced accuracy from a binary confusion matrix and the weight of the negative class.
    """
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
    predicted_kernel = predicted_kernel + torch.eye(predicted_kernel.shape[0], device=predicted_kernel.device)

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
        new_cluster_mask = torch.relu(new_cluster_mask.float() - assigned_mask.float()).long()

        # Mark the nodes in the current cluster as assigned
        assigned_mask += new_cluster_mask

        # Assign the current cluster index to all nodes in the new cluster
        clusters += new_cluster_mask*curr_cluster

        # If any node was assigned to the new cluster, increment the cluster index
        if new_cluster_mask.sum() > 0:
            curr_cluster += 1

    # Subtract 1 from cluster labels to make cluster labels start from 0
    return clusters - 1


def get_metrics_tensor(metrics_dict, ip, health_args):
    """
    Extract a tensor of metrics for a given IP address based on health arguments.
    """
    metrics_to_monitor = health_args['probe_metrics']
    if metrics_dict is not None and ip in metrics_dict:
        return torch.tensor([metrics_dict[ip][metric] for metric in metrics_to_monitor], dtype=torch.float32).T
    else:
        time_window_len = health_args['node_features_time_window']
        return torch.full((time_window_len, len(metrics_to_monitor)), -1.0, dtype=torch.float32)
