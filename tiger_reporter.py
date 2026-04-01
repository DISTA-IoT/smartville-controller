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

import plotly.express as px
from sklearn.decomposition import PCA
from smartController.brain_utils import TRAINING, INFERENCE, CLOSED_SET, ANOMALY_DETECTION

class TigerReporter:
    """
    Class responsible for generating reports and plots for TigerBrain.
    """

    def __init__(self, kwargs, wb_run, encoder, logger, seed):
        self.kwargs = kwargs
        self.wb_run = wb_run
        self.encoder = encoder
        self.logger = logger
        self.seed = seed
        self.wbt = kwargs['wandb']['wb_tracking']


    def report(self, preds, hiddens, labels, predicted_clusters, query_mask, phase,
               training_cs_cm=None, training_os_cm=None,
               custom_cs_cm=None, custom_os_cm=None):
        """
        Generates a dictionary of plots to be logged to Weights & Biases.
        """
        if phase == TRAINING:
            cs_cm_to_plot = training_cs_cm
            os_cm_to_plot = training_os_cm
        else:
            # INFERENCE
            cs_cm_to_plot = custom_cs_cm
            os_cm_to_plot = custom_os_cm

        log_dict = {}

        if self.wbt and self.kwargs['wandb']['plots']:
            # Closed Set Confusion Matrix
            if cs_cm_to_plot is not None:
                cs_conf_mat = self.plot_confusion_matrix(
                    mod=CLOSED_SET,
                    cm=cs_cm_to_plot,
                    phase=phase,
                    norm=False,
                    classes=self.encoder.get_labels())
                if cs_conf_mat:
                    log_dict[f'{phase}Plots/{CLOSED_SET} Confusion Matrix'] = cs_conf_mat

            # Anomaly Detection Confusion Matrix
            if os_cm_to_plot is not None:
                os_conf_mat = self.plot_confusion_matrix(
                    mod=ANOMALY_DETECTION,
                    cm=os_cm_to_plot,
                    phase=phase,
                    norm=False,
                    classes=['Known', 'ZdA'])
                if os_conf_mat:
                    log_dict[f'{phase}Plots/{ANOMALY_DETECTION} Confusion Matrix'] = os_conf_mat

            # Hidden Space Projection
            fig_gt, fig_pred = self.plot_hidden_space(
                hiddens=hiddens,
                labels=labels,
                predicted_labels=predicted_clusters,
                phase=phase)
            if fig_gt:
                log_dict[f"{phase}Plots/Ground-truth clusters"] = fig_gt
            if fig_pred:
                log_dict[f"{phase}Plots/Predicted clusters"] = fig_pred

            # Scores Vectors PCA
            fig_scores = self.plot_scores_vectors(
                score_vectors=preds,
                labels=labels[query_mask],
                phase=phase)
            if fig_scores:
                log_dict[f"{phase}Plots/PCA of ass. scores"] = fig_scores

        self.logger.debug(f'{phase} CS Conf matrix: \n {cs_cm_to_plot}')
        self.logger.debug(f'{phase} AD Conf matrix: \n {os_cm_to_plot}')

        return log_dict


    def plot_confusion_matrix(self, mod, cm, phase, norm=True, classes=None):
        """
        Generates a Plotly heatmap for a confusion matrix.
        """
        if self.wb_run is None:
            return None

        cm_np = cm.detach().cpu().numpy().astype(float)

        if norm:
            # Normalize and prevent division by zero
            denom = cm_np.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            cm_np = cm_np / denom
            fmt_str = '.2f'
        else:
            fmt_str = '.0f'

        # Ensure classes are strings
        str_classes = [str(c) for c in classes]

        # Generate a Plotly Heatmap
        fig = px.imshow(
            cm_np,
            x=str_classes,
            y=str_classes,
            labels=dict(x="Predicted", y="Baseline", color="Count"),
            color_continuous_scale="Blues",
            text_auto=fmt_str,
            title=f'{phase} {mod} Confusion Matrix'
        )

        fig.update_xaxes(side="bottom")
        return fig


    def plot_hidden_space(self, hiddens, labels, predicted_labels, phase):
        """
        Generates scatter plots of the hidden space projected to 2D using PCA.
        """
        if self.wb_run is None:
            return None, None

        hiddens_np = hiddens.detach().cpu().numpy()
        hiddens_np = self.project_to_2d(hiddens_np, context_name="hidden vectors")
        if hiddens_np is None:
            return None, None

        labels_np = labels.squeeze(1).detach().cpu().numpy()
        nl_labels = self.encoder.inverse_transform_to_str(labels_np)

        pred_labels_np = []
        if predicted_labels is not None:
            pred_labels_np = [str(lbl) for lbl in predicted_labels.detach().cpu().numpy()]
        else:
            # If no predicted clusters are provided, we can't plot them
            pred_labels_np = ["N/A"] * len(nl_labels)

        # Prepare a lightweight dictionary for Plotly
        data_dict = {
            "PCA_1": hiddens_np[:, 0],
            "PCA_2": hiddens_np[:, 1],
            "Ground Truth": nl_labels,
            "Predicted Cluster": pred_labels_np
        }

        # Plot 1: Ground Truth
        fig_gt = px.scatter(
            data_dict, x="PCA_1", y="PCA_2", color="Ground Truth",
            title=f'{phase} Ground-truth clusters'
        )
        # Enlarge the markers a bit
        fig_gt.update_traces(marker=dict(size=10, opacity=0.7))

        # Plot 2: Predicted Clusters
        fig_pred = None
        if predicted_labels is not None:
            fig_pred = px.scatter(
                data_dict, x="PCA_1", y="PCA_2", color="Predicted Cluster",
                title=f'{phase} Predicted clusters'
            )
            fig_pred.update_traces(marker=dict(size=10, opacity=0.7))

        return fig_gt, fig_pred


    def plot_scores_vectors(self, score_vectors, labels, phase):
        """
        Generates a scatter plot of the score vectors projected to 2D using PCA.
        """
        if self.wb_run is None:
            return None

        scores_np = score_vectors.detach().cpu().numpy()
        scores_np = self.project_to_2d(scores_np, context_name="score vectors")
        if scores_np is None:
            return None

        labels_np = labels.squeeze(1).detach().cpu().numpy()
        nl_labels = self.encoder.inverse_transform_to_str(labels_np)

        # Lightweight dictionary
        data_dict = {
            "Score_X": scores_np[:, 0],
            "Score_Y": scores_np[:, 1],
            "Ground Truth": nl_labels
        }

        fig = px.scatter(
            data_dict, x="Score_X", y="Score_Y", color="Ground Truth",
            title=f'{phase} PCA reduction of association scores'
        )
        fig.update_traces(marker=dict(size=10, opacity=0.7))

        return fig


    def project_to_2d(self, vectors_np, context_name):
        """
        Robust and fast 2D projection.
        Uses randomized SVD when possible for steadier runtime across different hidden sizes.
        """
        if vectors_np.shape[1] < 2:
            self.logger.warning(
                f'PCA not applied to {context_name} because they are too low dimensional'
            )
            return None

        if vectors_np.shape[1] == 2:
            return vectors_np

        min_dim = min(vectors_np.shape[0], vectors_np.shape[1])
        solver = 'randomized' if min_dim > 2 else 'full'

        try:
            pca = PCA(
                n_components=2,
                svd_solver=solver,
                random_state=self.seed,
                copy=False
            )
            return pca.fit_transform(vectors_np)
        except Exception as e:
            self.logger.warning(
                f'Error during PCA ({solver}) applied to {context_name}: {e}. Falling back to full solver.'
            )
            try:
                return PCA(n_components=2, svd_solver='full', copy=False).fit_transform(vectors_np)
            except Exception as inner_e:
                self.logger.warning(f'Fallback PCA failed on {context_name}: {inner_e}')
                return None
