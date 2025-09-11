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
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

from smartController.attr_dict import AttrDict


TRAINING_METRICS_SECTION_NAME = "Training Metrics"
EVALUATION_METRICS_SECTION_NAME = "Evaluation Metrics"
INFERENCE_METRICS_SECTION_NAME = "Inference Metrics"
TRAIN_PLOTS_SECTION_NAME = "Training Plots"
INFERENCE_PLOTS_SECTION_NAME = "Inference Plots"

from enum import Enum

class MetricNames(Enum):
    EvalKR = "Mean EVAL KR PREC"
    EvalCS = "Mean EVAL CS ACC"
    EvalAD = "Mean EVAL AD ACC"

    TrainKRNMI = "Training_KR_NMI"
    TrainKRARI = "Training_KR_ARI"
    TrainKRLoss = "Training_KR_LOSS"
    TrainAcc = "Training_Acc"
    TrainLoss = "Training_Loss"
    TrainAnomalyBalance = "Training_ANOMALY_BALANCE"
    TrainADLoss = "Training_AD_LOSS"
    TrainADAcc = "Training_AD_Acc"

    InfLoss = "Inference_Loss"
    InfAcc = "Inference_Acc"
    InfKRNMI = "Inference_KR_NMI"
    InfKRARI = "Inference_KR_ARI"
    InfKRLoss = "Inference_KR_LOSS"
    InfAnomalyBalance = "Inference_ANOMALY_BALANCE"
    InfADLoss = "Inference_AD_LOSS"
    InfADAcc = "Inference_AD_Acc"


class PlotNames(Enum):
    TrainPCA_assoc = "Training PCA of ass. scores"
    TrainLSR = "Training Latent Space Representations"
    TrainCSFM = "Training CS Confusion Matrix"
    TrainADCM = "Training AD Confusion Matrix"

    InfPCA_assoc = "Inference PCA of ass. scores"
    InfLSR = "Inference Latent Space Representations"
    InfCSFM = "Inference CS Confusion Matrix"
    InfADCM = "Inference AD Confusion Matrix"


class WandBTracker():

    def __init__(self, kwargs):
        args = AttrDict(kwargs)
        self.logger = args.logger
        self.wb_logger = wandb.init(
            # Set the project where this run will be logged
            project=args.wandb.wb_project_name,
            name=args.wandb.wb_run_name,
            # Track hyperparameters and run metadata
            config=kwargs,
            mode=("online" if args.wandb.wb_tracking else "disabled"),
            )
        
        if args.wandb.wb_tracking:
            try:
                self.workspace =  ws.Workspace.from_url(
                    f"https://wandb.ai/{args.wandb.entity}/{args.wandb.wb_project_name}?nw={args.wandb.view_name}")
                
            except Exception as e:

                if e.args[0] == f'Workspace `{args.wandb.view_name}` not found in project `{args.wandb.wb_project_name}`':
                    self.logger.warning(e.args[0])
                    self.logger.info(f"Creating workspace: {args.wandb.view_name}")
                    self.workspace = ws.Workspace(
                        name=args.wandb.view_name,
                        entity=args.wandb.entity,
                        project=args.wandb.wb_project_name
                    )
                else:
                    self.logger.error(f"Failed to load workspace: {e}")
                    if self.wb_logger is not None:
                        self.wb_logger.finish()
                    raise RuntimeError(f"Failed to load workspace: {e}")

            try:
                self.set_workspace()
            except Exception as e:
                self.logger.error(f"Failed to set workspace: {e}")
                if self.wb_logger is not None:
                    self.wb_logger.finish()
                raise RuntimeError(f"Failed to set workspace: {e}")
            

    def set_workspace(self):
    
        modified_workspace = self.set_sections()


        try:
            if modified_workspace:
                self.workspace.save()
        except Exception as e:
            self.logger.error(f"Failed to save workspace: {e}")
            if self.wb_logger is not None:
                self.wb_logger.finish()
            raise RuntimeError(f"Failed to save workspace: {e}")
        finally:
            if modified_workspace:
                self.logger.info(f"Workspace modified.")
            else:
                self.logger.info(f"Workspace left unchanged. ")
            self.logger.info(f"Workspace available at {self.workspace.url}")


    def set_sections(self):
        
        modified_workspace = False
        section_names = [section.name for section in self.workspace.sections]
        
        if INFERENCE_METRICS_SECTION_NAME not in section_names:                                            
            self.workspace.sections.insert(1,
                ws.Section(
                    name=INFERENCE_METRICS_SECTION_NAME,
                    panels=[
                        wr.LinePlot(x="Step", y=[MetricNames.InfLoss.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.InfAcc.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.InfKRNMI.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.InfKRARI.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.InfKRLoss.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.InfAnomalyBalance.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.InfADLoss.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.InfADAcc.value])
                    ],
                    is_open=True,
                )
            )
            self.logger.info(f"Will try to add {INFERENCE_METRICS_SECTION_NAME} section")
            modified_workspace = True

        if INFERENCE_PLOTS_SECTION_NAME not in section_names:
            self.workspace.sections.insert(2,
                ws.Section(
                    name=INFERENCE_PLOTS_SECTION_NAME,
                    panels=[
                        wr.MediaBrowser(
                            media_keys=[PlotNames.InfPCA_assoc.value]),
                        wr.MediaBrowser(
                            media_keys=[PlotNames.InfLSR.value]),
                        wr.MediaBrowser(
                            media_keys=[PlotNames.InfCSFM.value]),
                        wr.MediaBrowser(
                            media_keys=[PlotNames.InfADCM.value])
                    ],
                    is_open=True,
                )
            )
            self.logger.info(f"Will try to add {INFERENCE_PLOTS_SECTION_NAME} section")
            modified_workspace = True

        if TRAINING_METRICS_SECTION_NAME not in section_names:
            self.workspace.sections.insert(3,
                ws.Section(
                    name=TRAINING_METRICS_SECTION_NAME,
                    panels=[
                        wr.LinePlot(x="Step", y=[MetricNames.TrainLoss.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.TrainAcc.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.TrainKRNMI.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.TrainKRARI.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.TrainKRLoss.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.TrainAnomalyBalance.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.TrainADLoss.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.TrainADAcc.value])
                    ],
                    is_open=True,
                )
            )
            self.logger.info(f"Will try to add {TRAINING_METRICS_SECTION_NAME} section")
            modified_workspace = True

        if TRAIN_PLOTS_SECTION_NAME not in section_names:
            self.workspace.sections.insert(4,
                ws.Section(
                    name=TRAIN_PLOTS_SECTION_NAME,
                    panels=[
                        wr.MediaBrowser(
                            media_keys=[PlotNames.TrainPCA_assoc.value]),
                        wr.MediaBrowser(
                            media_keys=[PlotNames.TrainLSR.value]),
                        wr.MediaBrowser(
                            media_keys=[PlotNames.TrainCSFM.value]),
                        wr.MediaBrowser(
                            media_keys=[PlotNames.TrainADCM.value])
                    ],
                    is_open=True,
                )
            )
            self.logger.info(f"Will try to add {TRAIN_PLOTS_SECTION_NAME} section")
            modified_workspace = True

        if EVALUATION_METRICS_SECTION_NAME not in section_names:
            self.workspace.sections.insert(5,
                ws.Section(
                    name=EVALUATION_METRICS_SECTION_NAME,
                    panels=[
                        wr.LinePlot(x="Step", y=[MetricNames.EvalKR.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.EvalCS.value]),
                        wr.LinePlot(x="Step", y=[MetricNames.EvalAD.value])
                    ],
                    is_open=True,
                )
            )
            self.logger.info(f"Will try to add {EVALUATION_METRICS_SECTION_NAME} section")
            modified_workspace = True
        
        return modified_workspace
        