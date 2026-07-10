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
import torch.nn as nn
import torch.nn.functional as F


def _as_bool(value):
    """Coerce a kwargs value (which may arrive as a real bool or as a config
    string like "True"/"1") into a Python bool."""
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


# Length of the FIXED SCALAR part of the proprioceptive tail -- the block of
# per-tick scalars that always follows the exteroceptive (centroid/relational)
# block, in order: anomaly count, ZDA confidence, known count, classification
# confidence, acquired-CTI fraction, CTI-available flag, budget.
#
# The full proprioceptive tail is this scalar block PLUS the per-class acquired-
# CTI map (one 0/1 slot per class in env.all_class_labels; see
# TigerBrain.assembly_state_vector / NewTigerEnvironment.acquired_cti_vector),
# so its width depends on the run's class count and is NOT a compile-time
# constant. That true width is computed once in TigerBrain.init_agents and
# threaded to every net through kwargs['proprio_state_size']; the nets split the
# state at THAT boundary (exteroceptive = x[:, :-proprio_state_size],
# proprioceptive = x[:, -proprio_state_size:]). Use proprio_tail_size(kwargs)
# below to read it, falling back to this base when a net is built without the
# key (e.g. a legacy checkpoint with no acquired map). Keep the scalar layout
# here in sync with the channels assembled in assembly_state_vector.
PROPRIOCEPTIVE_STATE_SIZE = 7


def proprio_tail_size(kwargs):
    """Full width of the proprioceptive tail the nets must split off: the fixed
    scalar block (PROPRIOCEPTIVE_STATE_SIZE) plus the per-class acquired-CTI map.
    Read from kwargs['proprio_state_size'] (set in TigerBrain.init_agents);
    falls back to the scalar-only base when absent."""
    return int(kwargs.get('proprio_state_size', PROPRIOCEPTIVE_STATE_SIZE))


def _make_proprio_norm(kwargs):
    """Normaliser for the proprioceptive tail: always an Identity pass-through.

    The proprioceptive tail is normalised per-feature upstream, at assembly time
    (TigerBrain.assembly_state_vector) -- counts log1p'd, budget squashed onto
    (-1, 1), and every already-bounded channel (the confidences, the acquired-CTI
    fraction and its per-class map, the CTI flag) left exactly as-is. Pooling it
    through a LayerNorm here would re-entangle those channels: one diverging
    budget would inflate the per-sample variance and blank the others (including
    the structural zeros that mark the known-vs-unknown regime), and the sparse
    0/1 acquired-CTI map would drag the pooled mean/variance. So the net leaves
    the tail untouched and LayerNorm is confined to the exteroceptive centroid
    (ExteroceptiveNorm), the only sub-block on a raw, drifting scale."""
    return nn.Identity()


class ExteroceptiveNorm(nn.Module):
    """Normaliser for the exteroceptive block's PROTOTYPE (raw-centroid) part.

    The exteroceptive block is laid out ``[prototype centroid | relational
    summary | class-scores vector]`` (TigerBrain._exteroceptive_block), and the
    leading part lives on a very different footing than everything after it:

    - the prototype centroid enters the DM nets as RAW absolute hidden
      coordinates, whose scale drifts as the encoder keeps training online --
      so its leading ``proto_dim`` columns are LayerNorm'd here (zero-mean /
      unit-variance per sample, with a learnable affine).
    - the relational summary and the class-scores vector (each present only
      when their respective mode/flag is on) are ALREADY bounded/scaled
      per-feature by construction -- log-score similarity stats at O(tens) and
      tanh reward channels in (-1, 1) for the former, a softmax distribution or
      raw log-similarities for the latter -- with meaningful structural zeros
      (a cold class's reward, an as-yet-unbought class's sentinel score).
      LayerNorm would destroy those, so every trailing column is passed
      through untouched regardless of which of the two features produced it.

    ``proto_dim`` is always just the raw-centroid width (TigerBrain sets
    ``exteroceptive_proto_dim`` independently of whether relational_state /
    include_class_scores are on), so this realises every `state` mode /
    include_class_scores combination with one module:
      - 'prototype', class scores off: proto_dim == full exteroceptive width
        -> whole block normed.
      - 'relational', class scores off: proto_dim == 0 -> a no-op.
      - any mode with a nonzero trailing part (relational summary and/or
        class-scores vector): only the leading centroid is normed, everything
        after it is left in peace.
    """

    def __init__(self, proto_dim):
        super().__init__()
        self.proto_dim = int(proto_dim)
        self.norm = nn.LayerNorm(self.proto_dim) if self.proto_dim > 0 else None

    def forward(self, exteroceptive):
        if self.norm is None:
            return exteroceptive
        if self.proto_dim >= exteroceptive.shape[-1]:
            # Whole exteroceptive block is the raw centroid (no relational
            # summary or class-scores vector active this run).
            return self.norm(exteroceptive)
        # Normalise the leading centroid, pass everything after it through
        # (relational summary and/or class-scores vector, whichever is active).
        proto = self.norm(exteroceptive[:, :self.proto_dim])
        trailing = exteroceptive[:, self.proto_dim:]
        return torch.cat((proto, trailing), dim=1)


def _make_extero_norm(kwargs):
    """Normaliser for the exteroceptive block, selected implicitly by the
    `state` mode via the injected `exteroceptive_proto_dim` (the width of the
    raw-centroid part; TigerBrain.init_agents). See ExteroceptiveNorm. In pure
    'relational' mode proto_dim is 0 and this is a no-op, so the already
    feature-scaled relational summary is fed to the net unchanged."""
    return ExteroceptiveNorm(int(kwargs.get('exteroceptive_proto_dim', 0)))


class PolicyNet(nn.Module):
    def __init__(self, kwargs):
        super(PolicyNet, self).__init__()
        self.proprio_norm = _make_proprio_norm(kwargs)
        self.extero_norm = _make_extero_norm(kwargs)
        # Full proprioceptive-tail width (fixed scalars + per-class acquired-CTI
        # map). Stored so forward() splits at the run's true boundary rather than
        # the scalar-only module constant.
        self.proprio_state_size = proprio_tail_size(kwargs)
        hidden_size = int(kwargs['hidden_size'])
        self.fc1 = nn.Linear(int(kwargs['state_size']), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, int(kwargs['action_size']))

    def forward(self, x):
        if len(x.shape)<2:
            x = x.unsqueeze(0)
        exteroceptive_part = self.extero_norm(x[:,:-self.proprio_state_size])
        proprioceptive_part = self.proprio_norm(x[:,-self.proprio_state_size:])
        x = torch.cat((exteroceptive_part, proprioceptive_part), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=len(x.shape)-1)
        return x


class ValueNet(nn.Module):
    def __init__(self, kwargs):
        super(ValueNet, self).__init__()
        self.proprio_norm = _make_proprio_norm(kwargs)
        self.extero_norm = _make_extero_norm(kwargs)
        # Full proprioceptive-tail width (fixed scalars + per-class acquired-CTI
        # map). Stored so forward() splits at the run's true boundary rather than
        # the scalar-only module constant.
        self.proprio_state_size = proprio_tail_size(kwargs)
        hidden_size = int(kwargs['hidden_size'])
        self.fc1 = nn.Linear(int(kwargs['state_size']), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if len(x.shape)<2:
            x = x.unsqueeze(0)
        exteroceptive_part = self.extero_norm(x[:,:-self.proprio_state_size])
        proprioceptive_part = self.proprio_norm(x[:,-self.proprio_state_size:])
        x = torch.cat((exteroceptive_part, proprioceptive_part), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class NEFENet(nn.Module):
    def __init__(self, kwargs):
        """
        Thought to booststrap the value in term of the NEGATIVE EXPECTED FREE ENERGY
        """
        super(NEFENet, self).__init__()
        self.proprio_norm = _make_proprio_norm(kwargs)
        self.extero_norm = _make_extero_norm(kwargs)
        # Full proprioceptive-tail width (fixed scalars + per-class acquired-CTI
        # map). Stored so forward() splits at the run's true boundary rather than
        # the scalar-only module constant.
        self.proprio_state_size = proprio_tail_size(kwargs)
        hidden_size = int(kwargs['hidden_size'])
        self.fc1 = nn.Linear(int(kwargs['state_size']), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, int(kwargs['action_size']))

    def forward(self, x):
        if len(x.shape)<2:
            x = x.unsqueeze(0)
        exteroceptive_part = self.extero_norm(x[:,:-self.proprio_state_size])
        proprioceptive_part = self.proprio_norm(x[:,-self.proprio_state_size:])
        x = torch.cat((exteroceptive_part, proprioceptive_part), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQN(nn.Module):
    def __init__(self, kwargs):
        super(DQN, self).__init__()
        self.proprio_norm = _make_proprio_norm(kwargs)
        self.extero_norm = _make_extero_norm(kwargs)
        # Full proprioceptive-tail width (fixed scalars + per-class acquired-CTI
        # map). Stored so forward() splits at the run's true boundary rather than
        # the scalar-only module constant.
        self.proprio_state_size = proprio_tail_size(kwargs)
        hidden_size = int(int(kwargs['hidden_size']))
        self.fc1 = nn.Linear(int(kwargs['state_size']), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, int(int(kwargs['action_size'])))

    def forward(self, x):
        if len(x.shape)<2:
            x = x.unsqueeze(0)
        exteroceptive_part = self.extero_norm(x[:,:-self.proprio_state_size])
        proprioceptive_part = self.proprio_norm(x[:,-self.proprio_state_size:])
        x = torch.cat((exteroceptive_part, proprioceptive_part), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQN(nn.Module):
    def __init__(self, kwargs):
        super(DuelingDQN, self).__init__()
        self.proprio_norm = _make_proprio_norm(kwargs)
        self.extero_norm = _make_extero_norm(kwargs)
        # Full proprioceptive-tail width (fixed scalars + per-class acquired-CTI
        # map). Stored so forward() splits at the run's true boundary rather than
        # the scalar-only module constant.
        self.proprio_state_size = proprio_tail_size(kwargs)
        hidden_dim = int(int(kwargs['hidden_size']))
        self.fc1 = nn.Linear(int(kwargs['state_size']), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, int(kwargs['action_size']))
        )


    def forward(self, x):
        if len(x.shape)<2:
            x = x.unsqueeze(0)
        exteroceptive_part = self.extero_norm(x[:,:-self.proprio_state_size])
        proprioceptive_part = self.proprio_norm(x[:,-self.proprio_state_size:])
        x = torch.cat((exteroceptive_part, proprioceptive_part), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(MLP, self).__init__()
        hidden_size = (output_size - input_size) // 2
        hidden_size = output_size - hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    
class RecurrentModel(nn.Module):

    def __init__(self, input_size, hidden_size, dropout, recurrent_layers, device='cpu'):
        super(RecurrentModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, dropout=dropout, num_layers=int(recurrent_layers), batch_first=True)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through GRU layer
        out, _ = self.gru(x, h0)
        
        return F.relu(out[:, -1, :])


class MulticlassPrototypicalClassifier(nn.Module):

    def __init__(self, device='cpu'):
        super(MulticlassPrototypicalClassifier, self).__init__()
        self.device = device


    def get_oh_labels(
        self,
        decimal_labels,
        n_way):

        # create placeholder for one_hot encoding:
        labels_onehot = torch.zeros(
            [decimal_labels.size()[0],
            n_way], device=self.device)
        # transform to one_hot encoding:
        labels_onehot = labels_onehot.scatter(
            1,
            decimal_labels,
            1)
        return labels_onehot


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
    

    def forward(self, hidden_vectors, labels, known_attacks_count, query_mask):
        """
        known_attacks_count is the current number of known attacks. 
        """
        # get one_hot_labels of current batch:
        oh_labels = self.get_oh_labels(
            decimal_labels=labels.long(),
            n_way=known_attacks_count)

        # get latent centroids:
        centroids, _ = self.get_centroids(
            hidden_vectors[~query_mask],
            oh_labels[~query_mask])

        # compute scores:
        scores = 1 / (torch.cdist(hidden_vectors[query_mask], centroids) + 1e-10)

        return scores


class MultiClassFlowClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob, kr_heads=8,device='cpu', kwargs=None):
        super(MultiClassFlowClassifier, self).__init__()
        self.device=device
        self.normalizer = nn.BatchNorm1d(input_size)
        rnn_input_dim = input_size
        self.use_encoder = False
        if kwargs['use_encoder']:
            self.use_encoder = True
            rnn_input_dim = hidden_size
            self.encoder = MLP(input_size, hidden_size, dropout_prob)
        self.rnn = RecurrentModel(rnn_input_dim, hidden_size, dropout_prob, int(kwargs['recurrent_layers']), device=self.device)
        kernel_regressor_class = DistKernelRegressor if kwargs['kr_type'] == 'dist' else DotProdKernelRegressor            
        self.kernel_regressor = kernel_regressor_class(
            {'device': self.device,
            'dropout': dropout_prob,
            'n_heads': kr_heads,
            'in_features': hidden_size,
            'out_features': hidden_size})
        self.classifier = MulticlassPrototypicalClassifier(device=self.device)

    def forward(self, x, labels, curr_known_attack_count, query_mask):
        # nn.BatchNorm1d ingests (N,C,L), where N is the batch size, 
        # C is the number of features or channels, and L is the sequence length
        x = self.normalizer(x.permute((0,2,1))).permute((0,2,1))
        if self.use_encoder:
            x = self.encoder(x)
        hiddens = self.rnn(x)
        hiddens, predicted_kernel = self.kernel_regressor(hiddens)
        logits  = self.classifier(hiddens, labels, curr_known_attack_count, query_mask)
        return logits, hiddens, predicted_kernel


class TwoStreamMulticlassFlowClassifier(nn.Module):
    def __init__(self, flow_input_size, second_stream_input_size, hidden_size, dropout_prob=0.2, kr_heads=8, device='cpu', kwargs=None):
        super(TwoStreamMulticlassFlowClassifier, self).__init__()
        self.device = device
        self.flow_normalizer = nn.BatchNorm1d(flow_input_size)
        flow_rnn_input_dim = flow_input_size
        second_stream_rnn_input_dim = second_stream_input_size
        self.use_encoder = False
        if kwargs['use_encoder']:
            self.use_encoder = True
            flow_rnn_input_dim = hidden_size
            second_stream_rnn_input_dim = hidden_size
            self.flow_encoder = MLP(flow_input_size, hidden_size, dropout_prob)
            self.second_stream_encoder = MLP(second_stream_input_size, hidden_size, dropout_prob)

        self.flow_rnn = RecurrentModel(flow_rnn_input_dim, hidden_size, dropout_prob, int(kwargs['recurrent_layers']), device=self.device)
        self.second_stream_normalizer = nn.BatchNorm1d(second_stream_input_size)
        self.second_stream_rnn = RecurrentModel(second_stream_rnn_input_dim, hidden_size, dropout_prob, int(kwargs['recurrent_layers']), device=self.device)
        kernel_regressor_class = DistKernelRegressor if kwargs['kr_type'] == 'dist' else DotProdKernelRegressor 
        self.kernel_regressor = kernel_regressor_class(
            {'device': self.device,
            'dropout': dropout_prob,
            'n_heads': kr_heads,
            'in_features': hidden_size*2,
            'out_features': hidden_size*2})
        self.classifier = MulticlassPrototypicalClassifier(device=self.device)

    def forward(self, flows, second_domain_feats, labels, curr_known_attack_count, query_mask):
        
        flows = self.flow_normalizer(flows.permute((0,2,1))).permute((0,2,1))
        second_domain_feats = self.second_stream_normalizer(second_domain_feats.permute((0,2,1))).permute((0,2,1))

        if self.use_encoder:
            flows = self.flow_encoder(flows)
            second_domain_feats = self.second_stream_encoder(second_domain_feats)

        flows = self.flow_rnn(flows)
        second_domain_feats = self.second_stream_rnn(second_domain_feats)

        hiddens = torch.cat([flows, second_domain_feats], dim=1)

        hiddens, predicted_kernel = self.kernel_regressor(hiddens)
        logits  = self.classifier(hiddens, labels, curr_known_attack_count, query_mask)

        return logits, hiddens, predicted_kernel
 

class ThreeStreamMulticlassFlowClassifier(nn.Module):
    def __init__(self, flow_input_size, second_stream_input_size, third_stream_input_size, hidden_size, dropout_prob=0.2, kr_heads=8, device='cpu', kwargs=None):
        super(ThreeStreamMulticlassFlowClassifier, self).__init__()
        self.device = device
        self.flow_normalizer = nn.BatchNorm1d(flow_input_size)
        self.use_encoder = False
        flow_rnn_input_dim = flow_input_size
        second_stream_rnn_input_dim = second_stream_input_size
        third_stream_rnn_input_dim = third_stream_input_size
        if kwargs['use_encoder']:
            self.use_encoder = True
            self.flow_rnn_input_dim = hidden_size
            self.second_stream_rnn_input_dim = hidden_size
            self.third_stream_rnn_input_dim = hidden_size
            self.flow_encoder = MLP(flow_input_size, hidden_size, dropout_prob)
            self.second_stream_encoder = MLP(second_stream_input_size, hidden_size, dropout_prob)
            self.third_stream_encoder = MLP(third_stream_input_size, hidden_size, dropout_prob)

        self.flow_rnn = RecurrentModel(flow_rnn_input_dim, hidden_size, dropout_prob, int(kwargs['recurrent_layers']), device=self.device)
        self.second_stream_normalizer = nn.BatchNorm1d(second_stream_input_size)
        self.second_stream_rnn = RecurrentModel(second_stream_rnn_input_dim, hidden_size, dropout_prob, int(kwargs['recurrent_layers']), device=self.device)
        self.third_stream_normalizer = nn.BatchNorm1d(third_stream_input_size)
        self.third_stream_rnn = RecurrentModel(third_stream_rnn_input_dim, hidden_size, dropout_prob, int(kwargs['recurrent_layers']), device=self.device)
        kernel_regressor_class = DistKernelRegressor if kwargs['kr_type'] == 'dist' else DotProdKernelRegressor 
        self.kernel_regressor = kernel_regressor_class(
            {'device': self.device,
            'dropout': dropout_prob,
            'n_heads': kr_heads,
            'in_features': hidden_size*3,
            'out_features': hidden_size*3})
        self.classifier = MulticlassPrototypicalClassifier(device=self.device)

    def forward(self, flows, second_domain_feats, third_domain_feats, labels, curr_known_attack_count, query_mask):
        
        flows = self.flow_normalizer(flows.permute((0,2,1))).permute((0,2,1))
        second_domain_feats = self.second_stream_normalizer(second_domain_feats.permute((0,2,1))).permute((0,2,1))
        third_domain_feats = self.third_stream_normalizer(third_domain_feats.permute((0,2,1))).permute((0,2,1))

        if self.use_encoder:
            flows = self.flow_encoder(flows)
            second_domain_feats = self.second_stream_encoder(second_domain_feats)
            third_domain_feats = self.third_stream_encoder(third_domain_feats)

        flows = self.flow_rnn(flows)
        second_domain_feats = self.second_stream_rnn(second_domain_feats)
        third_domain_feats = self.third_stream_rnn(third_domain_feats)

        hiddens = torch.cat([flows, second_domain_feats, third_domain_feats], dim=1)

        hiddens, predicted_kernel = self.kernel_regressor(hiddens)
        logits  = self.classifier(hiddens, labels, curr_known_attack_count, query_mask)

        return logits, hiddens, predicted_kernel
 

class ConfidenceDecoder(nn.Module):

    def __init__(
            self,
            device):

        super(ConfidenceDecoder, self).__init__()
        self.device = device


    def forward(
            self,
            scores):

        scores = (1 - scores.unsqueeze(-1)).min(1)[0]

        unknown_indicators = torch.sigmoid(scores)
        return unknown_indicators
    

class KernelRegressionLoss(nn.Module):

    def __init__(
            self,
            repulsive_weigth: int = 1, 
            attractive_weigth: int = 1,
            device: str = "cpu"):
        super(KernelRegressionLoss, self).__init__()
        self.r_w = repulsive_weigth
        self.a_w = attractive_weigth
        self.device = device

    def forward(self, baseline_kernel, predicted_kernel):
        # REPULSIVE force
        repulsive_CE_term = -(1 - baseline_kernel) * torch.log(1-predicted_kernel + 1e-10)
        repulsive_CE_term = repulsive_CE_term.sum(dim=1)
        repulsive_CE_term = repulsive_CE_term.mean()

        # The following acts as an ATTRACTIVE force for the embedding learning:
        attractive_CE_term = -(baseline_kernel * torch.log(predicted_kernel + 1e-10))
        attractive_CE_term = attractive_CE_term.sum(dim=1)
        attractive_CE_term = attractive_CE_term.mean()

        return (self.r_w * repulsive_CE_term) + (self.a_w * attractive_CE_term)



class TransitionNet(nn.Module):

    def __init__(
            self,
            kwargs):
        super(TransitionNet, self).__init__()

        self.transition_input_size = int(kwargs['proprioceptive_state_size']) + int(kwargs['action_size'])
        self.recurrent_layers = int(kwargs['recurrent_layers'])
        self.act = nn.LeakyReLU(kwargs['leakyrelu_alpha'])
        self.fc1 = nn.Linear(self.transition_input_size, int(kwargs['hidden_size']))
        self.fc2 = nn.Linear(int(int(kwargs['hidden_size'])), int(kwargs['proprioceptive_state_size']))

    def forward(self, x):
        
        # compatibility with batch processing
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        
        x_res = self.fc1(x)
        x_res = self.act(x_res)
        x_res = self.act(x_res)
        x_res = self.fc2(x_res)
        
        return x_res    



class NewTransitionNet(nn.Module):

    def __init__(
            self,
            kwargs):
        super(NewTransitionNet, self).__init__()
        
        self.action_size = int(kwargs['action_size'])
        self.proprioceptive_state_size = int(kwargs['proprioceptive_state_size'])
        self.state_size = int(kwargs['state_size'])

        self.transition_input_size = int(kwargs['state_size']) + int(kwargs['action_size'])
        self.act = nn.LeakyReLU(kwargs['leakyrelu_alpha'])


        self.action_stream_fc1 = nn.Linear(self.action_size, int(kwargs['hidden_size'])//4)
        self.action_stream_fc2 = nn.Linear(int(kwargs['hidden_size'])//4, int(kwargs['hidden_size'])//8)

        self.exteroceptive_state_stream_fc1 = nn.Linear(int(kwargs['state_size']) - self.proprioceptive_state_size, int(kwargs['hidden_size'])// 4)
        self.exteroceptive_state_stream_fc2 = nn.Linear(int(kwargs['hidden_size']) // 4, int(kwargs['hidden_size'])// 8)

        self.final_fc_1 = nn.Linear(int(kwargs['hidden_size']) // 4 + self.proprioceptive_state_size, self.proprioceptive_state_size)
        self.final_fc_2 = nn.Linear(self.proprioceptive_state_size, self.proprioceptive_state_size)

    def forward(self, x):
        
        # compatibility with batch processing
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        
        action_part = x[:,-self.action_size:]
        state_part = x[:,:-self.action_size]
        exteroceptive_state_part = state_part[:,:-self.proprioceptive_state_size]
        proprioceptive_state_part = state_part[:,-self.proprioceptive_state_size:]

        action_part = self.act(self.action_stream_fc1(action_part))
        action_part = self.act(self.action_stream_fc2(action_part))

        exteroceptive_state_part = self.act(self.exteroceptive_state_stream_fc1(exteroceptive_state_part))
        exteroceptive_state_part = self.act(self.exteroceptive_state_stream_fc2(exteroceptive_state_part))

        x_res = torch.cat([action_part, exteroceptive_state_part, proprioceptive_state_part], dim=1)

        x_res = self.act(self.final_fc_1(x_res))
        x_res = self.act(self.final_fc_2(x_res))

        output = proprioceptive_state_part + x_res
        
        return output


class VariationalTransitionNet(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        
        self.action_size = int(kwargs['action_size'])
        self.proprioceptive_state_size = int(kwargs['proprioceptive_state_size'])
        self.state_size = int(kwargs['state_size'])

        self.transition_input_size = int(kwargs['state_size']) + int(kwargs['action_size'])
        self.act = nn.LeakyReLU(kwargs['leakyrelu_alpha'])


        self.action_stream_fc1 = nn.Linear(self.action_size, int(kwargs['hidden_size'])//4)
        self.action_stream_fc2 = nn.Linear(int(kwargs['hidden_size'])//4, int(kwargs['hidden_size'])//8)

        self.exteroceptive_state_stream_fc1 = nn.Linear(int(kwargs['state_size']) - self.proprioceptive_state_size, int(kwargs['hidden_size'])// 4)
        self.exteroceptive_state_stream_fc2 = nn.Linear(int(kwargs['hidden_size']) // 4, int(kwargs['hidden_size'])// 8)

        self.final_fc_mean_1 = nn.Linear(int(kwargs['hidden_size']) // 4 + self.proprioceptive_state_size, self.proprioceptive_state_size)
        self.final_fc_mean_2 = nn.Linear(self.proprioceptive_state_size, self.proprioceptive_state_size)

        self.final_fc_logvar_1 = nn.Linear(int(kwargs['hidden_size']) // 4 + self.proprioceptive_state_size, self.proprioceptive_state_size)
        self.final_fc_logvar_2 = nn.Linear(self.proprioceptive_state_size, self.proprioceptive_state_size)


    def forward(self, x):

        # compatibility with batch processing
        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        action_part = x[:,-self.action_size:]
        state_part = x[:,:-self.action_size]
        exteroceptive_state_part = state_part[:,:-self.proprioceptive_state_size]
        proprioceptive_state_part = state_part[:,-self.proprioceptive_state_size:]

        action_part = self.act(self.action_stream_fc1(action_part))
        action_part = self.act(self.action_stream_fc2(action_part))

        exteroceptive_state_part = self.act(self.exteroceptive_state_stream_fc1(exteroceptive_state_part))
        exteroceptive_state_part = self.act(self.exteroceptive_state_stream_fc2(exteroceptive_state_part))

        x_res = torch.cat([action_part, exteroceptive_state_part, proprioceptive_state_part], dim=1)

        x_res_mean = self.act(self.final_fc_mean_1(x_res))
        x_res_mean = self.act(self.final_fc_mean_2(x_res_mean))
        mean = proprioceptive_state_part + x_res_mean

        x_res_logvar = self.act(self.final_fc_logvar_1(x_res))
        x_res_logvar = self.act(self.final_fc_logvar_2(x_res_logvar))
        logvar = x_res_logvar

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mean + eps * std

        return sample, mean, logvar
    
    
class SimmilarityNet(nn.Module):
    def __init__(
            self,
            hidden_size):
        super(SimmilarityNet, self).__init__()

        self.act = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x1, x2):
        input_to_symm = torch.abs(x1 - x2)
        symm = self.fc1(input_to_symm)
        symm = self.act(symm)
        symm = self.fc2(symm)
        return symm


class DotProdKernelRegressor(nn.Module):

    def __init__(
            self,
            kwargs):

        super(DotProdKernelRegressor, self).__init__()

        self.act = nn.Sigmoid()
        self.device = kwargs['device']

    def forward(
            self,
            hiddens):
        
        kernel = self.act(hiddens @ hiddens.T)

        return hiddens, kernel



class DistKernelRegressor(nn.Module):

    def __init__(
            self,
            kwargs):

        super(DistKernelRegressor, self).__init__()

        self.device = kwargs['device']
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(-0.5))
        self.similarity_network = SimmilarityNet(hidden_size=int(kwargs['in_features']))

    def forward(
            self,
            hiddens):
        
        n_nodes = hiddens.shape[0]

        h_pivot = hiddens.repeat(
            n_nodes,
            1)

        h_interleave = hiddens.repeat_interleave(
            n_nodes,
            dim=0)
        
        energies = self.similarity_network(h_pivot, h_interleave)

        kernel = torch.sigmoid(energies)

        kernel = kernel.reshape(n_nodes,n_nodes)

        return hiddens, kernel


class SimpleKernelRegressor(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_heads: int,
            is_concat: bool = False,
            dropout: float = 0.0,
            leaky_relu_negative_slope: float = 0.2,
            share_weights: bool = True,
            device: str = "cpu"):

        super(SimpleKernelRegressor, self).__init__()

        self.device = device
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(-0.5))

    def forward(
            self,
            hiddens):
        
        n_nodes = hiddens.shape[0]
        
        energies = 1/ (torch.cdist(hiddens, hiddens)+1e-10)

        kernel = torch.sigmoid(energies)

        kernel = kernel.reshape(n_nodes,n_nodes)

        return hiddens, kernel


class KernelRegressor(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_heads: int,
            is_concat: bool = False,
            dropout: float = 0.0,
            leaky_relu_negative_slope: float = 0.2,
            share_weights: bool = True,
            device: str = "cpu"):

        super(KernelRegressor, self).__init__()

        self.regressor = GraphAttentionV2Layer(
            in_features=in_features,
            out_features=out_features,
            n_heads=n_heads,
            is_concat=is_concat,
            dropout=dropout,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
            share_weights=share_weights
        )
        self.device = device


    def forward(
            self,
            hiddens):

        return self.regressor(hiddens)


class GraphAttentionV2Layer(nn.Module):


    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int,
                 is_concat: bool = False,
                 dropout: float = 0.1,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = True):

        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_l = nn.Linear(
            in_features,
            self.n_hidden * n_heads,
            bias=False)

        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(
                in_features,
                self.n_hidden * n_heads,
                bias=False)

        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(
            self.n_hidden,
            1,
            bias=False)

        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(
            negative_slope=leaky_relu_negative_slope)

        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)

        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)


    def forward(self,
                h: torch.Tensor):

        # Number of nodes
        n_nodes = h.shape[0]

        # The initial GAT transformations,
        # We do two linear transformations and then split it up for each head.
        g_l = self.linear_l(h).view(
            n_nodes,
            self.n_heads,
            self.n_hidden)

        g_r = self.linear_r(h).view(
            n_nodes,
            self.n_heads,
            self.n_hidden)

        # #### Calculate attention score
        g_l_repeat = g_l.repeat(
            n_nodes,
            1,
            1)

        g_r_repeat_interleave = g_r.repeat_interleave(
            n_nodes,
            dim=0)

        g_sum = g_l_repeat + g_r_repeat_interleave

        g_sum = g_sum.view(
            n_nodes,
            n_nodes,
            self.n_heads,
            self.n_hidden)

        # get energies
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        """
        # We assume a fully connected adj_mat
        assert adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == n_nodes
        adj_mat = adj_mat.unsqueeze(-1)
        adj_mat = adj_mat.repeat(1, 1, self.n_heads)

        e = e.masked_fill(adj_mat == 0, float('-inf'))
        """

        # Normalization
        a = self.softmax(e)

        a = self.dropout(a)

        # Calculate final output for each head
        hiddens = torch.einsum('ijh,jhf->ihf', a, g_r)

        
        if self.is_concat:
            # Concatenate the heads
            hiddens = hiddens.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            # Take the mean of the heads
            hiddens = hiddens.mean(dim=1)
        
    
        a =  a.mean(dim=2)

        # we are making discrete kernel regression. 
        # A node might have many neighbours:
        a = a / (a.max(dim=1)[0] + 1e-10)
        a = a.clamp(min=0, max=1)

        return hiddens, a