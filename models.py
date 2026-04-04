# This file is part of the "Smartville" project.
# Copyright (c) 2024 University of Insubria
# Licensed under the Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# For the full text of the license, visit:
# https://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ---------------------------------------------------------
# UTILS & DEPENDENCIES
# ---------------------------------------------------------

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for feature encoding.
    """
    def __init__(self, input_size, output_size, dropout=0.2):
        super(MLP, self).__init__()
        hidden_size = max(output_size, (output_size + input_size) // 2)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class RecurrentModel(nn.Module):
    """
    GRU-based recurrent model for sequence encoding.
    """
    def __init__(self, input_size, hidden_size, dropout, recurrent_layers, device='cpu'):
        super(RecurrentModel, self).__init__()
        self.device = device
        self.hidden_size = int(hidden_size)
        # dropout is only applied between layers, so it requires > 1 layer
        self.gru = nn.GRU(
            input_size,
            self.hidden_size,
            dropout=dropout if int(recurrent_layers) > 1 else 0,
            num_layers=int(recurrent_layers),
            batch_first=True
        )

    def forward(self, x):
        # out: [Batch, Seq, Hidden]
        out, _ = self.gru(x)
        # Return the last output of the sequence: [Batch, Hidden]
        return F.relu(out[:, -1, :])

class MulticlassPrototypicalClassifier(nn.Module):
    """
    Prototypical classifier that computes scores based on distance to class centroids.
    """
    def __init__(self, device='cpu'):
        super(MulticlassPrototypicalClassifier, self).__init__()
        self.device = device

    def get_oh_labels(self, decimal_labels, n_way):
        labels_onehot = torch.zeros([decimal_labels.size(0), n_way], device=self.device)
        labels_onehot.scatter_(1, decimal_labels, 1)
        return labels_onehot

    def get_centroids(self, hidden_vectors, onehot_labels):
        cluster_agg = onehot_labels.T @ hidden_vectors
        samples_per_cluster = onehot_labels.sum(0)

        centroids = torch.zeros_like(cluster_agg, device=self.device)
        missing_clusters = samples_per_cluster == 0

        safe_samples = samples_per_cluster[~missing_clusters].unsqueeze(-1)
        centroids[~missing_clusters] = cluster_agg[~missing_clusters] / safe_samples

        return centroids, missing_clusters

    def forward(self, hidden_vectors, labels, known_attacks_count, query_mask):
        # Create one-hot labels for the support set
        oh_labels = self.get_oh_labels(decimal_labels=labels.long(), n_way=known_attacks_count)

        # Calculate centroids from support set (not in query mask)
        centroids, _ = self.get_centroids(hidden_vectors[~query_mask], oh_labels[~query_mask])

        # Compute Euclidean distances: [N_query, N_centroids]
        dists = torch.cdist(hidden_vectors[query_mask], centroids)

        # Inverse distance scores for classification
        scores = 1.0 / (dists + 1e-10)
        return scores

# ---------------------------------------------------------
# KERNEL REGRESSORS (OPTIMIZED)
# ---------------------------------------------------------

class SimmilarityNet(nn.Module):
    """
    Small MLP to compute similarity between a pair of vectors.
    """
    def __init__(self, hidden_size):
        super(SimmilarityNet, self).__init__()
        self.act = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x1, x2):
        # Absolute difference is a symmetric feature suitable for similarity
        input_to_symm = torch.abs(x1 - x2)
        x = self.fc1(input_to_symm)
        x = self.act(x)
        x = self.fc2(x)
        return x

class DistKernelRegressor(nn.Module):
    """
    Optimized Distance-based Kernel Regressor.
    Computes only the upper triangle of the similarity matrix to save ~50% CPU.
    """
    def __init__(self, kwargs):
        super(DistKernelRegressor, self).__init__()
        self.device = kwargs.get('device', 'cpu')
        in_features = int(kwargs.get('in_features', 128))

        sim_net = SimmilarityNet(hidden_size=in_features)

        # Apply torch.compile if available and not disabled via env var
        if os.environ.get('TORCH_COMPILE', '1') == '1' and hasattr(torch, 'compile'):
            try:
                # Optimized for CPU inference
                self.similarity_network = torch.compile(sim_net)
            except Exception:
                self.similarity_network = sim_net
        else:
            self.similarity_network = sim_net

    def forward(self, hiddens):
        n = hiddens.shape[0]
        if n <= 1:
            return hiddens, torch.ones((n, n), device=hiddens.device)

        # 1. Get indices for the upper triangle (i < j)
        indices = torch.triu_indices(n, n, offset=1, device=hiddens.device)

        # 2. Extract unique pairs
        h1 = hiddens[indices[0]]
        h2 = hiddens[indices[1]]

        # 3. Compute similarity for all unique pairs in one batch
        upper_tri_energies = self.similarity_network(h1, h2)
        upper_tri_sim = torch.sigmoid(upper_tri_energies).squeeze(-1)

        # 4. Fill symmetric matrix
        # Self-similarity (diagonal) is 1.0
        kernel = torch.eye(n, device=hiddens.device)
        kernel[indices[0], indices[1]] = upper_tri_sim
        kernel[indices[1], indices[0]] = upper_tri_sim

        return hiddens, kernel

class DotProdKernelRegressor(nn.Module):
    """
    Improved Dot-Product Kernel Regressor with projections and scaling for convergence.
    """
    def __init__(self, kwargs):
        super(DotProdKernelRegressor, self).__init__()
        self.device = kwargs.get('device', 'cpu')
        hidden_dim = int(kwargs.get('in_features', 128))

        # Learnable projections to improve representational capacity and convergence
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.scale = hidden_dim ** 0.5

        # Learnable temperature for sigmoid scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, hiddens):
        h = self.ln(hiddens)
        q = self.query(h)
        k = self.key(h)

        # Scaled dot product similarity: [N, N]
        scores = (q @ k.T) / self.scale

        # Sigmoid to normalize kernel values to [0, 1]
        kernel = torch.sigmoid(scores * self.temperature)

        return hiddens, kernel

# ---------------------------------------------------------
# CLASSIFIERS
# ---------------------------------------------------------

class OneStreamMulticlassFlowClassifier(nn.Module):
    def __init__(self, kwargs):
        super(OneStreamMulticlassFlowClassifier, self).__init__()
        self.device = kwargs.get('device', 'cpu')

        flow_input_size = int(kwargs.get('flow_feat_dim', 30))
        hidden_size = int(kwargs.get('hidden_size', 128))
        dropout_prob = float(kwargs.get('dropout_prob', 0.2))
        recurrent_layers = int(kwargs.get('recurrent_layers', 1))
        kr_type = kwargs.get('kr_type', 'dist')

        self.normalizer = nn.BatchNorm1d(flow_input_size)

        rnn_input_dim = flow_input_size
        self.use_encoder = kwargs.get('use_encoder', False)
        if self.use_encoder:
            rnn_input_dim = hidden_size
            self.encoder = MLP(flow_input_size, hidden_size, dropout_prob)

        self.rnn = RecurrentModel(rnn_input_dim, hidden_size, dropout_prob, recurrent_layers, device=self.device)

        kr_kwargs = {'device': self.device, 'in_features': hidden_size}
        self.kernel_regressor = DistKernelRegressor(kr_kwargs) if kr_type == 'dist' else DotProdKernelRegressor(kr_kwargs)
        self.classifier = MulticlassPrototypicalClassifier(device=self.device)

    def forward(self, x, labels, curr_known_attack_count, query_mask):
        x = self.normalizer(x.transpose(1, 2)).transpose(1, 2)
        if self.use_encoder:
            x = self.encoder(x)
        hiddens = self.rnn(x)
        hiddens, predicted_kernel = self.kernel_regressor(hiddens)
        logits = self.classifier(hiddens, labels, curr_known_attack_count, query_mask)
        return logits, hiddens, predicted_kernel

class TwoStreamMulticlassFlowClassifier(nn.Module):
    def __init__(self, kwargs):
        super(TwoStreamMulticlassFlowClassifier, self).__init__()
        self.device = kwargs.get('device', 'cpu')

        flow_input_size = int(kwargs.get('flow_feat_dim', 30))
        # Second stream is either packet or node features
        packet_dim = int(kwargs.get('packet_feat_dim', 0))
        node_dim = int(kwargs.get('node_feat_dim', 10))
        second_dim = packet_dim if packet_dim > 0 else node_dim

        hidden_size = int(kwargs.get('hidden_size', 128))
        dropout_prob = float(kwargs.get('dropout_prob', 0.2))
        recurrent_layers = int(kwargs.get('recurrent_layers', 1))
        kr_type = kwargs.get('kr_type', 'dist')

        self.flow_normalizer = nn.BatchNorm1d(flow_input_size)
        self.second_normalizer = nn.BatchNorm1d(second_dim)

        f_rnn_in, s_rnn_in = flow_input_size, second_dim
        self.use_encoder = kwargs.get('use_encoder', False)
        if self.use_encoder:
            f_rnn_in = s_rnn_in = hidden_size
            self.flow_encoder = MLP(flow_input_size, hidden_size, dropout_prob)
            self.second_encoder = MLP(second_dim, hidden_size, dropout_prob)

        self.flow_rnn = RecurrentModel(f_rnn_in, hidden_size, dropout_prob, recurrent_layers, device=self.device)
        self.second_rnn = RecurrentModel(s_rnn_in, hidden_size, dropout_prob, recurrent_layers, device=self.device)

        kr_kwargs = {'device': self.device, 'in_features': hidden_size * 2}
        self.kernel_regressor = DistKernelRegressor(kr_kwargs) if kr_type == 'dist' else DotProdKernelRegressor(kr_kwargs)
        self.classifier = MulticlassPrototypicalClassifier(device=self.device)

    def forward(self, flows, second_stream, labels, curr_known_attack_count, query_mask):
        flows = self.flow_normalizer(flows.transpose(1, 2)).transpose(1, 2)
        second_stream = self.second_normalizer(second_stream.transpose(1, 2)).transpose(1, 2)

        if self.use_encoder:
            flows = self.flow_encoder(flows)
            second_stream = self.second_encoder(second_stream)

        flows = self.flow_rnn(flows)
        second_stream = self.second_rnn(second_stream)

        hiddens = torch.cat([flows, second_stream], dim=1)
        hiddens, predicted_kernel = self.kernel_regressor(hiddens)
        logits = self.classifier(hiddens, labels, curr_known_attack_count, query_mask)
        return logits, hiddens, predicted_kernel

class ThreeStreamMulticlassFlowClassifier(nn.Module):
    def __init__(self, kwargs):
        super(ThreeStreamMulticlassFlowClassifier, self).__init__()
        self.device = kwargs.get('device', 'cpu')

        f_dim = int(kwargs.get('flow_feat_dim', 30))
        p_dim = int(kwargs.get('packet_feat_dim', 30))
        n_dim = int(kwargs.get('node_feat_dim', 10))

        hidden_size = int(kwargs.get('hidden_size', 128))
        dropout_prob = float(kwargs.get('dropout_prob', 0.2))
        recurrent_layers = int(kwargs.get('recurrent_layers', 1))
        kr_type = kwargs.get('kr_type', 'dist')

        self.flow_normalizer = nn.BatchNorm1d(f_dim)
        self.packet_normalizer = nn.BatchNorm1d(p_dim)
        self.node_normalizer = nn.BatchNorm1d(n_dim)

        f_rnn_in, p_rnn_in, n_rnn_in = f_dim, p_dim, n_dim
        self.use_encoder = kwargs.get('use_encoder', False)
        if self.use_encoder:
            f_rnn_in = p_rnn_in = n_rnn_in = hidden_size
            self.flow_encoder = MLP(f_dim, hidden_size, dropout_prob)
            self.packet_encoder = MLP(p_dim, hidden_size, dropout_prob)
            self.node_encoder = MLP(n_dim, hidden_size, dropout_prob)

        self.flow_rnn = RecurrentModel(f_rnn_in, hidden_size, dropout_prob, recurrent_layers, device=self.device)
        self.packet_rnn = RecurrentModel(p_rnn_in, hidden_size, dropout_prob, recurrent_layers, device=self.device)
        self.node_rnn = RecurrentModel(n_rnn_in, hidden_size, dropout_prob, recurrent_layers, device=self.device)

        kr_kwargs = {'device': self.device, 'in_features': hidden_size * 3}
        self.kernel_regressor = DistKernelRegressor(kr_kwargs) if kr_type == 'dist' else DotProdKernelRegressor(kr_kwargs)
        self.classifier = MulticlassPrototypicalClassifier(device=self.device)

    def forward(self, flows, packets, nodes, labels, curr_known_attack_count, query_mask):
        flows = self.flow_normalizer(flows.transpose(1, 2)).transpose(1, 2)
        packets = self.packet_normalizer(packets.transpose(1, 2)).transpose(1, 2)
        nodes = self.node_normalizer(nodes.transpose(1, 2)).transpose(1, 2)

        if self.use_encoder:
            flows = self.flow_encoder(flows)
            packets = self.packet_encoder(packets)
            nodes = self.node_encoder(nodes)

        flows = self.flow_rnn(flows)
        packets = self.packet_rnn(packets)
        nodes = self.node_rnn(nodes)

        hiddens = torch.cat([flows, packets, nodes], dim=1)
        hiddens, predicted_kernel = self.kernel_regressor(hiddens)
        logits = self.classifier(hiddens, labels, curr_known_attack_count, query_mask)
        return logits, hiddens, predicted_kernel

# Alias for backward compatibility
MultiClassFlowClassifier = OneStreamMulticlassFlowClassifier

# ---------------------------------------------------------
# ANOMALY DETECTION & LOSS
# ---------------------------------------------------------

class ConfidenceDecoder(nn.Module):
    def __init__(self, device='cpu'):
        super(ConfidenceDecoder, self).__init__()
        self.device = device

    def forward(self, scores):
        # Compute anomaly confidence based on prototype distances
        # If max score (inverse distance) is small, confidence in it being an anomaly is high
        conf = (1.0 - scores).min(dim=1, keepdim=True)[0]
        return torch.sigmoid(conf)

class KernelRegressionLoss(nn.Module):
    def __init__(self, repulsive_weigth: int = 1, attractive_weigth: int = 1, device: str = "cpu"):
        super(KernelRegressionLoss, self).__init__()
        self.r_w = repulsive_weigth
        self.a_w = attractive_weigth
        self.device = device

    def forward(self, baseline_kernel, predicted_kernel):
        """
        Optimized Kernel Regression Loss using weighted Binary Cross Entropy.
        Stable and efficient on CPU.
        """
        pred = predicted_kernel.clamp(min=1e-7, max=1-1e-7)
        # Weight mask: attractive for positive pairs, repulsive for negative pairs
        weights = baseline_kernel * self.a_w + (1 - baseline_kernel) * self.r_w
        return F.binary_cross_entropy(pred, baseline_kernel, weight=weights, reduction='mean')
