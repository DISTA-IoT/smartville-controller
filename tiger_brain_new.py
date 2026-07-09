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
import inspect
import importlib
import traceback
from collections import Counter
from functools import wraps
from contextlib import contextmanager

from smartController.replay_buffer import RawReplayBuffer, Batch
from smartController.wandb_tracker import WandBTracker
from smartController.tiger_environment_new import NewTigerEnvironment
from smartController.neural_modules import PROPRIOCEPTIVE_STATE_SIZE
from smartController.tiger_agents import (
    ValueLearningAgent, DAIP_Agent, DAIA_Agent,
    DAIF_Agent, DAISA_Agent, PPO_Agent, A2C_Agent
)
from smartController.attr_dict import AttrDict, coerce_config_types
from smartController.data_recorder import FlowDataRecorder

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

    # Width of the fixed-size relational exteroceptive state block produced by
    # _relational_summary when `relational_state` is enabled. Kept in sync with
    # the number of statistics that method stacks: the 7 similarity stats plus
    # the running-average accept-reward of the nearest (r_max) and runner-up
    # (r_runnerup) known class.
    RELATIONAL_STATE_DIM = 9

    def __init__(self, kwargs, wb_tracker=None):
        """
        Initializes the TigerBrain module with provided configuration and optional WandB tracker.
        """
        # Config values reach us as JSON / HTML-form text (from the dashboard
        # online, from manifest.json + CLI overrides offline), so numeric
        # hyperparameters arrive as strings and boolean flags as "true"/"false".
        # Coerce the two RL/ML config blocks once here, at the shared entry
        # point for both the online and offline paths, so every downstream
        # reader (this class, NewTigerEnvironment, the agents, the neural
        # modules) sees already-typed values instead of relying on a correct
        # int()/float() cast at every scattered call site. Mutating kwargs in
        # place also covers self.intrusion_detection_kwargs and the AttrDict
        # built below (both alias these same dicts).
        for _cfg_block in ('intrusion_detection', 'neural_modules'):
            if isinstance(kwargs.get(_cfg_block), dict):
                kwargs[_cfg_block] = coerce_config_types(kwargs[_cfg_block])

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
        self.packets_per_sample = int(args.intrusion_detection.packets_per_sample)
        # Caps how many packet-feature samples a single flow can inject into
        # one tick's batch (see stack_flow_tensors). Without this, a flow
        # with a deep backlog (e.g. a dense burst the controller fell behind
        # on) could blow a single batch up to thousands of rows, which is
        # quadratic-ish death for the kernel-regression/prototypical step and
        # can OOM the process. Excess backlog just drains over later ticks.
        self.max_packet_samples_per_flow_per_tick = int(
            args.intrusion_detection.get('max_packet_samples_per_flow_per_tick', 32))
        self.hidden_size = int(args.neural_modules.hidden_size)
        self.multi_class = args.intrusion_detection.multi_class
        self.kernel_regression = args.intrusion_detection.kernel_regression
        self.device = args.device

        # Training and Evaluation parameters
        self.seed = int(args.intrusion_detection.seed)
        random.seed(self.seed)
        # Seed torch here too (not just in init_inference_neural_modules), since
        # init_agents() below constructs the DM agent's policy/value networks and
        # their weight init draws from torch's global RNG. Without this, varying
        # `seed` changes exploration/sampling order but not the DM's starting
        # weights, undermining seed-controlled multi-run statistics.
        torch.manual_seed(self.seed)
        self.k_shot = int(args.intrusion_detection.k_shot)
        self.batch_size = int(args.intrusion_detection.batch_size)
        self.report_step_freq = int(args.intrusion_detection.report_step_freq)
        self.plot_step_freq = int(args.intrusion_detection.plot_step_freq)
        self.online_eval_step_freq = int(args.intrusion_detection.online_eval_step_freq)
        self.use_neural_AD = args.intrusion_detection.use_neural_AD
        self.use_neural_KR = args.intrusion_detection.use_neural_KR
        self.use_neural_CS = args.intrusion_detection.use_neural_CS
        self.online_evaluation = args.intrusion_detection.online_evaluation
        self.online_eval_rounds = int(args.intrusion_detection.online_evaluation_rounds)
        self.load_pretrained_inference_module = args.intrusion_detection.pretrained_inference
        self.clustering_loss_backprop = args.intrusion_detection.clustering_loss_backprop
        self.attractive_weight = float(args.intrusion_detection.attractive_weight)
        self.repulsive_weight = float(args.intrusion_detection.repulsive_weight)
        self.learning_rate = float(args.intrusion_detection.learning_rate)
        self.replay_buffer_max_capacity = int(args.intrusion_detection.replay_buffer_max_capacity)
        # K1: decouple the inference-module training cadence from the (inference)
        # tick. online_inference still runs every tick for live metrics; training
        # can run more or fewer gradient steps against the balanced buffers.
        # Defaults (1 step, every tick) reproduce the original one-step-per-tick
        # behaviour exactly. train_steps_per_tick>1 -> more gradient steps per
        # training tick (faster convergence, costlier tick); train_every_n_ticks>1
        # -> train only every n-th training-eligible tick (cheaper tick).
        self.train_steps_per_tick = max(0, int(args.intrusion_detection.get('train_steps_per_tick', 1)))
        self.train_every_n_ticks = max(1, int(args.intrusion_detection.get('train_every_n_ticks', 1)))
        # K2: when a flow captured no fresh packet this tick, whether it still
        # contributes a sample built from its sticky last-packets window (True,
        # legacy behaviour) or is skipped for this tick (False). Only has any
        # effect when packet features are in use (see stack_flow_tensors).
        self.cache_flows_with_no_packets = bool(
            args.intrusion_detection.get('cache_flows_with_no_packets', True))
        self.pretrained_models_dir = args.intrusion_detection.pretrained_models_dir
        self.update_target_freq = int(args.intrusion_detection.update_target_freq)
        self.unknown_accept_reward_scale = self.intrusion_detection_kwargs.get('unknown_accept_reward_scale', 1.0)
        self.unknown_malicious_accept_penalty_scale = self.intrusion_detection_kwargs.get('unknown_malicious_accept_penalty_scale', 1.0)
        self.useless_epistemic_penalty = float(self.intrusion_detection_kwargs.get('useless_epistemic_penalty', 0.0))
        # --- Supervision-value reward components (hidden by default). Two
        # optional, additive reward terms that credit a label-buying
        # (supervised) policy for the downstream quality gains its CTI
        # purchases produce -- so a supervised policy can out-earn an
        # unsupervised one on the merits of better perception, instead of
        # through hand-crafted penalties on the unsupervised baseline:
        #   * cluster_impurity_penalty_weight: per unknown-traffic cluster,
        #     subtracts weight * (1 - purity) from that cluster's reward, where
        #     purity is the fraction of the cluster's members sharing its
        #     majority true label (see brain_utils/get_clusters + the
        #     majority-vote purity already computed for CTI targeting). A policy
        #     that has bought more CTI leaves fewer distinct unknown classes
        #     bleeding together, so its collective-anomaly clusters are purer
        #     and it is penalised less.
        #   * classification_accuracy_reward_weight: per known-traffic group,
        #     adds weight * (that group's closed-set classification accuracy)
        #     to its reward. A policy with more Known classes classifies its
        #     known traffic more accurately and so earns more.
        # Both default to 0.0, which fully HIDES them: no reward contribution
        # and no wandb series emitted, reproducing existing runs exactly. The
        # gating knobs live in tiger/config/default.yaml and the controller's
        # pre_recorded_data/manifest.json (intrusion_detection block).
        self.cluster_impurity_penalty_weight = float(
            self.intrusion_detection_kwargs.get('cluster_impurity_penalty_weight', 0.0))
        self.classification_accuracy_reward_weight = float(
            self.intrusion_detection_kwargs.get('classification_accuracy_reward_weight', 0.0))
        # When True, the online anomaly detector scores each sample only
        # against the *true* Known-class prototypes, excluding G1 columns from
        # its known-set. By default G1 is folded into the inference known-set
        # (G1 carries zda_label 0 at inference), so a real zero-day (G2) that
        # happens to sit near a G1 prototype is scored as non-anomalous and
        # missed. Excluding G1 removes that specific blind spot. Default False
        # (legacy behaviour: G1 counts as known at inference).
        self.exclude_g1_from_ad_known_set = bool(
            self.intrusion_detection_kwargs.get('exclude_g1_from_ad_known_set', False))
        # Decision threshold applied to the confidence decoder's anomaly
        # probability (zda_predictions) to obtain the binary Known/anomaly
        # verdict, both for the online decision path and for the AD
        # confusion-matrix/accuracy metrics.
        self.ad_threshold = float(
            self.intrusion_detection_kwargs.get('ad_threshold', 0.75))
        # Exteroceptive-state encoding for the Decision Module, selected by the
        # `state` config knob (one of 'prototype', 'relational', 'mixed'):
        #  - 'prototype' (default, legacy raw-centroid state): the DM's
        #    exteroceptive block is the absolute hidden-space centroid of the
        #    group/cluster (width scales with the active feature streams).
        #  - 'relational': a fixed-size relational summary of the group's/
        #    cluster's similarity to the known-class prototypes (see
        #    _relational_summary), instead of the absolute centroid. Keeps the
        #    decision layer consistent with the prototypical/relational-
        #    bottleneck inductive bias of the perception layers, is invariant
        #    to the number of known classes, and is far more stable than
        #    absolute coordinates under representation drift.
        #  - 'mixed': the concatenation of both, [centroid | relational
        #    summary], so the DM sees the absolute embedding and the class-
        #    relative summary at once.
        # Legacy configs that only set the old boolean `relational_state`
        # (True -> 'relational') are still honoured when `state` is absent.
        state_mode = self.intrusion_detection_kwargs.get(
            'state',
            'relational' if self.intrusion_detection_kwargs.get('relational_state', False) else 'prototype')
        state_mode = str(state_mode).strip().lower()
        if state_mode not in ('prototype', 'relational', 'mixed'):
            raise ValueError(
                "intrusion_detection.state must be one of 'prototype', "
                f"'relational', 'mixed'; got {state_mode!r}")
        self.state_mode = state_mode
        # Derived flags naming which encodings contribute to the exteroceptive
        # block. `relational_state` stays a boolean alias driving all the
        # relational-summary machinery (reward tracker, summary construction,
        # wandb series) -- the summary is built for both 'relational' and
        # 'mixed'.
        self.use_prototype_state = state_mode in ('prototype', 'mixed')
        self.relational_state = state_mode in ('relational', 'mixed')
        # Temperature (alpha) on the reward channels of the relational
        # (exteroceptive) summary: r_max / r_runnerup are tanh(mean_r /
        # (alpha * |reward| scale)) (see _class_reward_feature). alpha > 1
        # widens the tanh's near-linear region so the extreme (very good /
        # very costly) classes keep more dynamic range instead of saturating;
        # alpha = 1.0 recovers the un-tempered squash. Default 2.0. Guarded to
        # a small positive so it can never zero or flip the divisor.
        self.reward_temperature = max(
            float(self.intrusion_detection_kwargs.get('reward_temperature', 2.0)), 1e-8)
        # When True, the proprioceptive tail of the state vector is
        # normalised per-feature at assembly time instead of by the net's
        # pooled LayerNorm(PROPRIOCEPTIVE_STATE_SIZE): the two unbounded
        # channels (anomaly/known counts and the accumulated budget) are
        # individually squashed, while the already-bounded confidences, the
        # acquired-CTI fraction and the CTI flag pass through
        # untouched -- so their structural zeros (which double as the
        # known-vs-unknown regime indicator) stay clean, and a diverging
        # budget can no longer inflate the pooled variance and blank out the
        # other five channels for that sample (see assembly_state_vector /
        # _proprio_budget_feature). The net drops its LayerNorm when this is
        # on (neural_modules: proprio_norm -> Identity), reading the same flag
        # from its kwargs. Default False (legacy behaviour: pooled LayerNorm).
        self.proprio_feature_scaling = bool(
            self.intrusion_detection_kwargs.get('proprio_feature_scaling', False))
        # Per-episode running average of the empirical accept-reward earned per
        # predicted known class, appended to each relational summary
        # (see _relational_summary / _update_relational_reward).
        self._reset_relational_reward_tracker()
        # Per-episode running |budget| scale used to normalise the budget
        # channel of the proprioceptive tail when proprio_feature_scaling is
        # on. Reset alongside the reward tracker so, like the reward memory,
        # the budget scale is re-estimated from each episode's own experience
        # rather than carried across episodes.
        self._reset_proprio_scale_tracker()
        # Ren et al. (2021) define relative Mahalanobis distance as a
        # post-hoc, frozen-feature diagnostic: the encoder is trained first,
        # then RMD is computed against its (fixed) representation. Our
        # confidence decoders instead sit inside the training graph -- the
        # AD/BCE loss backprops through `query_h` in
        # train_inf_module_single_batch every episode, so the encoder's
        # geometry is continuously reshaped to make pseudo-anomalies (G1s)
        # score as anomalous, not just diagnosed post-hoc. This knob lets us
        # detach the confidence decoder's inputs (hidden vectors and known-
        # class scores) from the encoder's graph, so the AD/BCE loss can
        # still update the decoder's own parameters (if any) but can no
        # longer shape the encoder. Default True preserves existing
        # behaviour (gradients flow to the encoder).
        self.ad_loss_backprop_to_encoder = bool(
            self.intrusion_detection_kwargs.get('ad_loss_backprop_to_encoder', True))
        # Max global grad-norm for the IM (classifier + confidence_decoder)
        # update. The Mahalanobis confidence decoder backprops into the encoder
        # through detached, floored whitening denominators (see
        # im_models/mahalanobis.py): the surviving gradient into the hidden
        # vectors is scaled by 1/within_var, which is bounded only by
        # 1/var_floor = 1e4 and grows without bound as the representation
        # collapses (Mahalanobis distance is scale-invariant in the forward
        # pass, but its gradient w.r.t. the un-whitened coordinates scales as
        # 1/scale). Over offline_replay's repeated passes this runs the encoder
        # off to inf/NaN, which later surfaces as a NaN value/critic loss the
        # moment those weights are loaded for an agency run (the DM's
        # exteroceptive state is a summary of these hidden vectors). Clipping
        # the update norm breaks that runaway. Set <= 0 to disable.
        self.grad_clip_max_norm = float(
            self.intrusion_detection_kwargs.get('grad_clip_max_norm', 10.0))
        # for the threshold_cti ablation:
        self.cti_confidence_threshold = float(self.intrusion_detection_kwargs.get('cti_confidence_threshold', 0.5))
        # hard_g2s ablation: a blocklist of G2 class labels whose CTI purchase is
        # forbidden. A would-be epistemic buy (action 2) whose targeted label
        # (the cluster's majority true label -- the class that would actually be
        # bought, see act_on_unknown_clusters) falls in this list is remapped to
        # a block (1) instead. Empty by default (no class blocked from purchase),
        # so it is a no-op unless explicitly set. Unlike greedy_cti/cti_period/
        # fixed_threshold_cti/no_epistemic_actions -- which are mutually-exclusive
        # policies over the epistemic slot -- this is an orthogonal FILTER meant
        # to be LAYERED on top of one of them (typically greedy_cti): greedy would
        # buy every available G2, and hard_g2s carves out the ones it must block
        # instead, yielding a "buy only the G2s that matter" oracle policy.
        self.hard_g2s = list(self.intrusion_detection_kwargs.get('hard_g2s', []) or [])
        self.logger_instance.info(
            "\033[1m[TigerBrain] unknown_accept_reward_scale=%s, unknown_malicious_accept_penalty_scale=%s\033[0m, "
            " useless_epistemic_penalty=%s, cluster_impurity_penalty_weight=%s, "
            "classification_accuracy_reward_weight=%s, exclude_g1_from_ad_known_set=%s, state=%s, "
            "reward_temperature=%s, proprio_feature_scaling=%s, "
            "ad_loss_backprop_to_encoder=%s, grad_clip_max_norm=%s, ad_threshold=%s, cti_threshold=%s\033[0m",
            self.unknown_accept_reward_scale, self.unknown_malicious_accept_penalty_scale,
            self.useless_epistemic_penalty, self.cluster_impurity_penalty_weight,
            self.classification_accuracy_reward_weight, self.exclude_g1_from_ad_known_set, self.state_mode,
            self.reward_temperature, self.proprio_feature_scaling,
            self.ad_loss_backprop_to_encoder, self.grad_clip_max_norm, self.ad_threshold, self.cti_confidence_threshold)

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
        self._label_names_lookup = None
        self._g1_codes_tensor = None
        self._g2_codes_tensor = None

        # Data collection (offline-replay capture mode)
        self.data_collection_mode = bool(args.intrusion_detection.get('data_collection_mode', False))
        self.data_collection_skip_training = bool(args.intrusion_detection.get('data_collection_skip_training', True))
        self.data_recorder = None
        if self.data_collection_mode:
            self.logger_instance.info(
                "[TigerBrain] data_collection_mode=True: every batch seen by process_input "
                "will be recorded to disk instead of (or in addition to) being used for "
                f"training. data_collection_skip_training={self.data_collection_skip_training}."
            )
            try:
                self.data_recorder = FlowDataRecorder(
                    out_dir=args.intrusion_detection.get('data_collection_dir', '/tmp/tiger_data_collection/'),
                    logger=self.logger_instance,
                    shard_size=args.intrusion_detection.get('data_collection_shard_size', 2000),
                    flush_interval_secs=args.intrusion_detection.get('data_collection_flush_interval_secs', 30),
                    use_packet_feats=bool(args.intrusion_detection.get('data_collection_use_packet_feats', False)) and self.use_packet_feats,
                    use_node_feats=bool(args.intrusion_detection.get('data_collection_use_node_feats', False)) and self.use_node_feats,
                    run_metadata=self._build_data_collection_manifest(),
                )
            except Exception:
                self.logger_instance.error(
                    "[TigerBrain] Failed to initialize FlowDataRecorder; disabling "
                    "data_collection_mode for this run so process_input falls back to "
                    "normal training behavior."
                )
                self.data_collection_mode = False
                self.data_recorder = None

    def _build_data_collection_manifest(self):
        """
        Snapshot of everything an offline replay script would need to reconstruct
        this exact TigerBrain/NewTigerEnvironment deterministically. Written once
        per collection run by FlowDataRecorder, alongside the shards themselves.
        """
        try:
            return {
                "intrusion_detection": self.kwargs.get('intrusion_detection', {}),
                "neural_modules": self.kwargs.get('neural_modules', {}),
                "knowledge": self.kwargs.get('knowledge', {}),
                "rewards": self.kwargs.get('rewards', {}),
                "health": self.kwargs.get('health', {}),
                "wandb": self.kwargs.get('wandb', {}),
                "traffic_dict": self.traffic_dict,
                "container_ips": self.container_ips,
                "ips_containers": self.ips_containers,
                "use_packet_feats": self.use_packet_feats,
                "use_node_feats": self.use_node_feats,
                "flow_feat_dim": self.flow_feat_dim,
                "packet_feat_dim": self.packet_feat_dim,
                "hidden_size": self.hidden_size,
                "device": self.device,
                "inference_model_variant": self.kwargs.get('inference_model_variant', 'default'),
            }
        except Exception:
            self.logger_instance.error(
                f"[TigerBrain] Failed to build data collection manifest: {traceback.format_exc()}"
            )
            return {}
        finally:
            self.logger_instance.info(
                "[TigerBrain] Built data collection manifest: "
                f"device={self.device} use_packet_feats={self.use_packet_feats} "
                f"use_node_feats={self.use_node_feats} flow_feat_dim={self.flow_feat_dim} "
                f"packet_feat_dim={self.packet_feat_dim} hidden_size={self.hidden_size} "
                f"inference_model_variant={self.kwargs.get('inference_model_variant', 'default')} "
                f"container_ips_count={len(self.container_ips or {})} "
                f"ips_containers_count={len(self.ips_containers or {})} "
                f"traffic_dict_keys={list((self.traffic_dict or {}).keys())}"
            )

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
        Shuts down monitoring threads, WandB tracker, and the data recorder
        (if data collection mode is active -- flushes the final partial shard
        so no in-flight samples are lost when an experiment is stopped).
        """
        if hasattr(self, '_stop_monitoring'):
            self._stop_monitoring.set()
            for t in self._monitoring_threads:
                if t.is_alive():
                    t.join(timeout=2.0)

        if self.data_recorder is not None:
            self.logger_instance.info("[TigerBrain] Shutting down: closing data recorder.")
            self.data_recorder.close()

        if self.wb_tracker:
            self.wb_tracker.shutdown()
            
    def init_intelligence(self):
        """
        Initializes metrics, confusion matrices, and the environment.
        """
        self.eval_queue = queue.Queue()
        self.current_known_classes_count = 0
        self.batch_processing_allowed = False
        # K1: counts training-eligible ticks since the last training step, so
        # training can fire once every train_every_n_ticks such ticks.
        self._ticks_since_train = 0
        self.best_cs_accuracy = 0
        self.best_AD_accuracy = 0
        self.best_KR_accuracy = 0
        # CTI-acquisition-latency tracker: label -> step_counter at which that
        # G2 was bought this episode, kept until the first later online tick
        # whose batch actually contains that class again (which closes it and
        # emits epistemic_delay/<label>). See perform_epistemic_action /
        # _log_epistemic_delays.
        self._pending_epistemic_delays = {}
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
        # A new episode forgets all bought CTI (env.reset restores the G2 set),
        # so any purchase still waiting on its class to reappear is void: drop
        # the pending epistemic-delay markers rather than closing them against
        # the next episode's traffic.
        self._pending_epistemic_delays = {}
        # New episode: forget the accumulated per-class accept-reward averages,
        # so the DM must re-estimate them from this episode's own experience.
        # This mirrors the IM reset above and the no-absolute-prototypes stance
        # -- reward memory must not leak across episodes any more than the
        # prototypes do.
        self._reset_relational_reward_tracker()
        # Same rationale for the proprioceptive budget scale: re-estimate it
        # from this episode's own budget trajectory rather than leaking the
        # previous episode's magnitude across the reset.
        self._reset_proprio_scale_tracker()
        self.episode_count += 1

    def init_agents(self, args):
        """
        Initializes the mitigation agent (e.g., DQN, DAI variants).
        """
        # Exteroceptive block of the state vector, selected by the `state`
        # config knob (see __init__). It is the concatenation of whichever
        # encodings are active, in a fixed order [prototype | relational]:
        #  - prototype part (state in {'prototype', 'mixed'}): the raw
        #    hidden-space centroid of the group/cluster, whose width scales
        #    with the number of active feature streams (flow [+node] [+packet]).
        #  - relational part (state in {'relational', 'mixed'}): a fixed-size
        #    relational summary of the group's/cluster's similarity to the
        #    known-class prototypes (RELATIONAL_STATE_DIM dims), independent of
        #    stream count and of how many known classes exist. See
        #    _relational_summary.
        # 'mixed' concatenates both; the build order here must match how each
        # group's/cluster's exteroceptive vector is assembled in
        # act_on_known_traffic / act_on_unknown_clusters.
        # Width of the prototype (raw-centroid) sub-block of the exteroceptive
        # block: the leading columns, before any relational summary. 0 in pure
        # 'relational' mode. The DM nets LayerNorm exactly this leading slice
        # (see neural_modules._make_extero_norm): the centroid enters as raw
        # absolute coordinates whose scale drifts as the encoder trains, while
        # the relational summary that may follow it is already bounded/scaled
        # per-feature (log-score stats + tanh reward channels) and must be left
        # untouched.
        self.exteroceptive_proto_dim = 0
        if self.use_prototype_state:
            self.exteroceptive_proto_dim += self.hidden_size
            if self.use_node_feats:
                self.exteroceptive_proto_dim += self.hidden_size
            if self.use_packet_feats:
                self.exteroceptive_proto_dim += self.hidden_size
        self.exteroceptive_dim = self.exteroceptive_proto_dim
        if self.relational_state:
            self.exteroceptive_dim += self.RELATIONAL_STATE_DIM
        self.state_space_dim = self.exteroceptive_dim

        # State space components:
        # 0. exteroceptive block: for unknown traffic, the centroid (or
        #    relational summary) of the current anomaly cluster; for known
        #    traffic, that of the current predicted-class group
        #    (a -1-filled placeholder only when there is no next group/cluster
        #    to chain to within the tick).
        # 1. number of anomalies inferred in the tick's batch
        # 2. confidence of the current anomaly cluster's members (unknown
        #    traffic); zeroed for known-traffic states, doubling as the
        #    regime indicator alongside slot 4.
        # 3. number of known-class samples inferred in the tick's batch
        # 4. confidence of the current predicted-class group's members (known
        #    traffic); zeroed for unknown-cluster states, doubling as the
        #    regime indicator alongside slot 2.
        # 5. acquired-CTI fraction: fraction of this episode's G2 (zero-day)
        #    pool already bought and delivered (env.acquired_cti_fraction()).
        #    Gives the value function an explicit, monotone memory of how much
        #    intelligence the agent has purchased, so the delayed known-traffic
        #    payoff of a buy is attributable to the epistemic action.
        # 6. available CTI options (boolean flag)
        # 7. current system budget
        self.state_space_dim += PROPRIOCEPTIVE_STATE_SIZE

        agent_mapping = {
            'DQN': ValueLearningAgent,
            'DDQN': ValueLearningAgent,
            'DuelingDQN': ValueLearningAgent,
            'DuelingDDQN': ValueLearningAgent,
            'DAI_P': DAIP_Agent,
            'DAI_A': DAIA_Agent,
            'DAI_SA': DAISA_Agent,
            'DAI_F': DAIF_Agent,
            'PPO': PPO_Agent,
            'A2C': A2C_Agent
        }
        
        agent_type = args.intrusion_detection.agent
        if agent_type not in agent_mapping:
            raise ValueError(f'Unknown agent type: {agent_type}')

        agent_class = agent_mapping[agent_type]
        args.intrusion_detection.action_size = 3  # block, pass or TCI acquisition
        args.intrusion_detection.state_size = self.state_space_dim
        # Width of the prototype sub-block the DM nets LayerNorm (0 in pure
        # 'relational' mode); read by neural_modules._make_extero_norm.
        args.intrusion_detection.exteroceptive_proto_dim = self.exteroceptive_proto_dim
        
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

    def load_models_from_variant(self):
        """
        Loads the ASAP model classes (classifier, confidence decoder, kernel
        regression loss) from the im_models sub-package, picking the variant
        (e.g. 'default', 'mahalanobis', 'optim') named in kwargs.
        """
        variant_name = self.kwargs.get('inference_model_variant', 'default')
        try:
            module = importlib.import_module(f'smartController.im_models.{variant_name}')
        except ImportError as e:
            raise RuntimeError(
                f"Unknown inference_model_variant '{variant_name}': no module "
                f"im_models/{variant_name}.py found: {e}")

        model_classes = {}
        for name, obj in vars(module).items():
            if (isinstance(obj, type) and
                hasattr(obj, '__bases__') and
                any('Module' in base.__name__ for base in obj.__bases__ if hasattr(base, '__name__'))):
                model_classes[name] = obj

        return model_classes

    def init_inference_neural_modules(self):
        """
        Initializes neural modules (classifier, confidence decoder, criterion).
        Loads pre-trained weights if specified.
        """
        torch.manual_seed(self.seed)
        model_classes = self.load_models_from_variant()

        if CONFIDENCE_DECODER_CLASS_NAME not in model_classes:
                raise RuntimeError(f"A class named {CONFIDENCE_DECODER_CLASS_NAME} was not found in your models.py file")
        
        self.confidence_decoder = model_classes[CONFIDENCE_DECODER_CLASS_NAME](device=self.device)
        # The confidence decoder's forward signature varies by model file: the
        # legacy prototypical decoders take only `scores` (the known-class
        # similarity slice), while the Mahalanobis decoder takes the raw
        # hiddens + support structure to build per-class Gaussians. Cache which
        # kwargs this decoder actually accepts so _call_confidence_decoder can
        # feed each one exactly what it declares, without either coupling the
        # brain to a specific decoder or editing the other model files.
        cd_params = inspect.signature(self.confidence_decoder.forward).parameters
        self._cd_accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in cd_params.values())
        self._cd_param_names = set(cd_params.keys())

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
            self.kwargs['neural_modules']['device'] = self.device
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

        if self.kwargs['neural_modules']['custom_inference_model_path']:
            self.classifier_path = self.pretrained_models_dir +self.kwargs['neural_modules']['custom_classifier_path']
            self.confidence_decoder_path = self.pretrained_models_dir + self.kwargs['neural_modules']['custom_confidence_decoder_path']
        else:
            self.classifier_path = f"{self.pretrained_models_dir}multiclass_flow{feats_str}_classifier_pretrained_h{self.hidden_size}.pt"
            self.confidence_decoder_path = f"{self.pretrained_models_dir}flow{feats_str}_confidence_decoder_pretrained_h{self.hidden_size}.pt"

        if self.load_pretrained_inference_module:
            if os.path.exists(self.pretrained_models_dir):
                if os.path.exists(self.classifier_path):
                    self.classifier.load_state_dict(torch.load(self.classifier_path, weights_only=True))
                    self._warn_if_non_finite(self.classifier, self.classifier_path, "classifier")
                    self.logger_instance.info(f"Pre-trained weights loaded from {self.classifier_path}")
                else:
                    self.logger_instance.error(f"Pre-trained folder not found at {self.pretrained_models_dir}.")

                if self.multi_class and os.path.exists(self.confidence_decoder_path):
                    self.confidence_decoder.load_state_dict(torch.load(self.confidence_decoder_path, weights_only=True))
                    self._warn_if_non_finite(self.confidence_decoder, self.confidence_decoder_path, "confidence decoder")
                    self.logger_instance.info(f"Pre-trained weights loaded from {self.confidence_decoder_path}")
                else:
                    self.logger_instance.error(f"Pre-trained folder not found at {self.pretrained_models_dir}.")
            else:
                self.logger_instance.info(f"Pre-trained folder not found at {self.pretrained_models_dir}.")

    def _warn_if_non_finite(self, model, path, name):
        """Loudly flag (and sanitise) a pretrained checkpoint that already
        contains NaN/Inf weights -- e.g. one written by a diverged Mahalanobis
        pretraining run before the save guard existed. Left unsanitised these
        weights NaN the DM's value loss the moment inference runs under agency.
        We nan_to_num them so the run can at least start, but the model is
        effectively untrained: retrain the IM with a lower learning rate or a
        tighter intrusion_detection.grad_clip_max_norm."""
        bad = [k for k, v in model.state_dict().items()
               if torch.is_tensor(v) and not torch.isfinite(v).all()]
        if bad:
            self.logger_instance.error(
                f'\033[91mLoaded {name} from {path} has non-finite (NaN/Inf) weights in '
                f'{bad} -- this checkpoint is from a diverged pretraining run. Sanitising to '
                f'zeros so the run can start, but retrain the IM (lower learning_rate / '
                f'grad_clip_max_norm) before trusting these results.\033[0m')
            with torch.no_grad():
                for p in model.parameters():
                    torch.nan_to_num_(p, nan=0.0, posinf=0.0, neginf=0.0)

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

            # Partial-capture channel (RC 3.4): a corrupted acquired class admits
            # only a fraction of its samples into its buffer, so its prototype is
            # built from sparser, noisier evidence. Randomly drop the rest before
            # pushing. Guarded by `enabled` so the clean regime draws no RNG and
            # admits every sample, exactly as before.
            if self.env.cti_delivery.enabled and masked_flow.shape[0] > 0:
                capture = self.env.cti_delivery.capture_for(
                    self.encoder.inverse_transform(label.view(1))[0])
                if capture < 1.0:
                    keep = torch.rand(masked_flow.shape[0], device=masked_flow.device) < capture
                    masked_flow = masked_flow[keep]
                    masked_packet = masked_packet[keep] if self.use_packet_feats else None
                    masked_node = masked_node[keep] if self.use_node_feats else None
                    masked_labels = masked_labels[keep]

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
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def get_label_names_from_encoded_labels(self, encoded_labels):
        """
        Resolves per-sample string label names from their encoded class
        labels, mirroring get_rewards_from_encoded_labels's caching.
        """
        if self._label_names_lookup is None or len(self._label_names_lookup) != len(self.encoder.get_mapping()):
            self._label_names_lookup = {
                class_idx: class_name
                for class_name, class_idx in self.encoder.get_mapping().items()
            }
        return [self._label_names_lookup[label.item()] for label in encoded_labels]

    def _class_index_to_name(self, class_idx):
        """
        Resolve a single encoded class index (int or 0-dim tensor) to its
        string label name, reusing the same cached inverse mapping as
        get_label_names_from_encoded_labels. Returns None for an index not in
        the current encoder mapping. Used by the per-class decision/belief
        wandb series (reward_beliefs, known_acceptances/known_blocks), whose
        keys are indexed by predicted class rather than by true label.
        """
        if self._label_names_lookup is None or len(self._label_names_lookup) != len(self.encoder.get_mapping()):
            self._label_names_lookup = {
                class_idx2: class_name
                for class_name, class_idx2 in self.encoder.get_mapping().items()
            }
        return self._label_names_lookup.get(int(class_idx))

    def _log_reward_beliefs(self):
        """
        Emit the current per-known-class running-average accept-reward -- the
        "spurious" reward beliefs the DM has attached to each predicted known
        class this episode, which feed the relational summary's r_max /
        r_runnerup channels (see _class_reward_feature / _update_relational_
        reward). They are spurious in that a group's membership is decided by
        prototypical *similarity*, so the reward credited to a class can be
        earned by traffic that merely resembles it. One series per class ->
        reward_beliefs/<class_name>. Only populated (and only meaningful) when
        relational_state is on, so the series stay hidden otherwise.
        """
        if not (self.wbt and self.relational_state):
            return
        beliefs = {}
        for idx, count in self._class_reward_count.items():
            if count == 0:
                continue
            name = self._class_index_to_name(idx)
            if name is None:
                continue
            beliefs[f'reward_beliefs/{name}'] = self._class_reward_sum[idx] / count
        if beliefs:
            self.reporter.log_scalars(beliefs, step=self.wb_tracker.step_counter)

    def _g1_column_mask(self, num_cols):
        """
        Boolean mask over the classifier's `num_cols` logit columns marking
        the columns that belong to G1 classes. Logit columns are indexed by
        the encoder's class code (get_oh_labels scatters by class label), so a
        G1 class's column index is exactly its encoded code.
        """
        mask = torch.zeros(num_cols, dtype=torch.bool, device=self.device)
        g1_codes = [c for c in self.encoder.get_codes_for_labels(self.env.current_knowledge['G1s']) if c < num_cols]
        if g1_codes:
            mask[torch.tensor(g1_codes, device=self.device)] = True
        return mask

    def _reset_relational_reward_tracker(self):
        """
        (Re)initialise the per-episode running average of the empirical
        accept-reward earned per predicted known class. Called at construction
        and at the start of every episode (reset_environment): the averages are
        deliberately NOT persisted across episodes, so the DM re-estimates them
        from each episode's own experience rather than memorising a fixed
        setting -- the same reason the state uses relational summaries instead
        of absolute prototypes.

        `_class_reward_sum` / `_class_reward_count` hold the sum and count of
        accepted-group rewards keyed by predicted class index (the same index
        space as the prototypical-logit columns). `_reward_abs_*` track the
        running mean of |reward|, used only to rescale the reward channel onto
        the similarity-logit scale before it enters the DM (whose LayerNorm
        covers only the proprioceptive tail, not this exteroceptive block).
        """
        self._class_reward_sum = {}
        self._class_reward_count = {}
        self._reward_abs_total = 0.0
        self._reward_abs_n = 0
        self._reward_abs_scale = 0.0

    def _reset_proprio_scale_tracker(self):
        """
        (Re)initialise the per-episode running |budget| scale used to
        normalise the budget channel of the proprioceptive tail when
        proprio_feature_scaling is on AND there is no bankruptcy floor to
        anchor to (env.disable_budget_bankrupt_termination is True). In the
        default regime the budget channel anchors to min_budget/init_budget
        instead (see _proprio_budget_feature), but the running scale is kept
        warm on every call so it is always available as the no-floor fallback.
        `_budget_abs_total` / `_budget_abs_n` accumulate the sum and count of
        |budget| observed at each state assembly this episode; `_budget_abs_
        scale` is their ratio (the running mean magnitude), the divisor the
        budget is squashed against in that fallback.

        Note this is a *dedicated* budget scale, not the reward |scale|:
          * the reward |scale| is a per-decision magnitude and is only ever
            populated when relational_state is on (see _update_relational_
            reward), whereas proprio scaling must work regardless of that flag;
          * budget is the running *accumulation* of per-decision rewards, so
            its magnitude is on a different (and larger) scale than a single
            decision's reward.
        Dividing budget by its own running magnitude keeps the channel ~O(1)
        in normal operation and lets tanh bound it under divergence.
        """
        self._budget_abs_total = 0.0
        self._budget_abs_n = 0
        self._budget_abs_scale = 0.0

    def _proprio_budget_feature(self, budget, device, dtype):
        """
        The bounded budget feature for the proprioceptive tail, on (-1, 1).
        Two regimes, forked on whether the bankruptcy floor is live:

          * Bankruptcy termination ENABLED (env.disable_budget_bankrupt_
            termination is False, the default): the budget has a meaningful
            absolute anchor -- min_budget is the death floor and init_budget
            the starting bankroll -- so encode absolute *runway* rather than a
            self-relative magnitude. z = (budget - init_budget) / (init_budget
            - min_budget) is centred on the starting bankroll: z = 0 at the
            start, z = -1 (tanh -> ~-0.76) at the min_budget floor, positive
            when ahead. tanh's resolution is best near the floor -- exactly
            where the buy/afford/bankruptcy decisions need it -- and saturates
            harmlessly when very solvent. This is independent of the raw
            budget magnitude, so budget being one or two orders larger than a
            single reward is irrelevant.

          * Bankruptcy termination DISABLED: min_budget is no longer a boundary
            the episode respects, so there is no floor to anchor to. Fall back
            to self-normalising by the running |budget| magnitude,
            tanh(budget / running_scale) -- the reward-style squash -- which
            still bounds a runaway budget without assuming an absolute anchor.

        The running |budget| scale is maintained on every call regardless of
        regime, so it is always warm as the fallback divisor. Degenerate
        denominators (~0 start-to-death runway, or ~0 running scale before any
        budget is seen) fall back to 1.0.
        """
        b = float(budget)
        # Always maintained: the fallback divisor for the no-floor regime.
        self._budget_abs_total += abs(b)
        self._budget_abs_n += 1
        self._budget_abs_scale = self._budget_abs_total / self._budget_abs_n
        if not self.env.disable_budget_bankrupt_termination:
            # Floor is live: anchor to it (absolute runway, centred on start).
            runway = self.env.init_budget - self.env.min_budget
            denom = runway if abs(runway) > 1e-8 else 1.0
            z = (b - self.env.init_budget) / denom
        else:
            # No floor: self-normalise by the running |budget| magnitude.
            scale = self._budget_abs_scale if self._budget_abs_scale > 1e-8 else 1.0
            z = b / scale
        return torch.tanh(torch.tensor(z, device=device, dtype=dtype))

    def _update_relational_reward(self, class_idx, group_reward):
        """
        Fold one accepted known-traffic group's empirical reward into the
        running average for its predicted class. Only accepted groups are
        recorded -- a blocked group earns a structural zero (see
        _decision_reward) that would otherwise drag every class's estimate
        toward zero and mask the real accept-value of the ones the policy
        happens to block.
        """
        idx = int(class_idx)
        self._class_reward_sum[idx] = self._class_reward_sum.get(idx, 0.0) + group_reward
        self._class_reward_count[idx] = self._class_reward_count.get(idx, 0) + 1
        self._reward_abs_total += abs(group_reward)
        self._reward_abs_n += 1
        self._reward_abs_scale = self._reward_abs_total / self._reward_abs_n

    def _class_reward_feature(self, class_idx, device, dtype):
        """
        The scale-normalised, bounded reward feature for one known class: the
        running-average accept-reward of `class_idx`, divided by the running
        |reward| scale (times the reward_temperature) and tanh-squashed onto
        (-1, 1) so it sits on the same footing as the similarity stats. A class
        not yet accepted this episode (count 0) returns a neutral 0.0 -- which
        coincides with the reward of a block, i.e. "no evidence it is worth
        accepting yet".

        reward_temperature (alpha) widens the near-linear region of the tanh:
        the input is mean_r / (alpha * scale), so a larger alpha pushes typical
        ratios further from saturation and preserves more dynamic range among
        the extreme (very good / very costly) classes, at the cost of a gentler
        response overall. Default 2.0; alpha=1.0 recovers the un-tempered
        squash.
        """
        idx = int(class_idx)
        count = self._class_reward_count.get(idx, 0)
        if count == 0:
            return torch.zeros((), device=device, dtype=dtype)
        mean_r = self._class_reward_sum[idx] / count
        scale = self._reward_abs_scale if self._reward_abs_scale > 1e-8 else 1.0
        scale = scale * self.reward_temperature
        return torch.tanh(torch.tensor(mean_r / scale, device=device, dtype=dtype))

    def _relational_summary(self, score_slice):
        """
        Fixed-size, permutation-invariant relational summary of a group's or
        cluster's prototypical similarity scores to the known-class prototypes
        (`score_slice`: [n_members, K] the
        `logits` the prototypical classifier produces). Returns a
        RELATIONAL_STATE_DIM-vector that depends ONLY on the relation to known
        classes -- never on absolute hidden coordinates -- so it stays
        consistent with the prototypical / relational-bottleneck inductive
        bias of the perception layer, is invariant to how many known classes
        currently exist (K can grow as CTI is bought), and is far more stable
        than an absolute centroid under representation drift.

        The similarity stats (as before): closeness to the nearest / farthest /
        mean prototype, spread across prototypes, the top-2 margin (ambiguity),
        the assignment entropy, and the energy (logsumexp).

        Interleaved with those, two reward stats bind the *value* of a decision
        to the relational structure without ever naming a class: r_max is the
        running-average accept-reward of the nearest prototype (the one s_max
        refers to) and r_runnerup is that of the runner-up prototype (the one
        the margin is measured against). So the DM reads "the class this most
        resembles has paid off well / badly" and "the ambiguity is between a
        good class and a bad one" -- ordered by relation, not by class identity,
        which would break the relational bottleneck.
        """
        # The prototypical CS logits fed here are inverse distances
        # (1/cdist, MulticlassPrototypicalClassifier) -- unbounded above and,
        # worse, they blow up hyperbolically exactly as the encoder tightens its
        # clusters (cdist -> 0 => score -> 1e10). Fed raw into the value net's
        # exteroceptive block (which is NOT normalised) that drove the DM's Q and
        # value_loss into the millions. Work in log-score space instead:
        # log(1/cdist) = -log(cdist) is bounded (~[-tens, +23]) and stays a
        # *monotone* function of similarity, so every stat below sits at
        # O(1..tens) as the design (SS 3.2) intended, while the decision-relevant
        # structure is preserved: nearest/runner-up ranking (hence
        # r_max/r_runnerup) is invariant under the monotone log, margin becomes a
        # scale-invariant log-ratio, and softmax/entropy stop saturating to
        # one-hot. clamp_min guards log(0) from a non-finite/zero upstream logit.
        s = torch.log(score_slice.clamp_min(1e-30)).mean(dim=0)  # [K] mean log-similarity
        k = s.shape[0]
        p = torch.softmax(s, dim=0)
        entropy = -(p * torch.log(p + 1e-10)).sum()
        energy = torch.logsumexp(s, dim=0)
        s_max = s.max()
        s_min = s.min()
        s_mean = s.mean()
        if k > 1:
            s_std = s.std(unbiased=False)
            top2 = torch.topk(s, 2)
            margin = top2.values[0] - top2.values[1]
            k_max, k_run = top2.indices[0], top2.indices[1]
        else:
            s_std = torch.zeros((), device=s.device, dtype=s.dtype)
            margin = s_max
            # Single known class: nearest and runner-up collapse to the same one.
            k_max = torch.argmax(s)
            k_run = k_max
        r_max = self._class_reward_feature(k_max, s.device, s.dtype)
        r_runnerup = self._class_reward_feature(k_run, s.device, s.dtype)
        return torch.stack(
            [s_max, r_max, s_mean, s_min, s_std, margin, r_runnerup, entropy, energy]
        ).to(dtype=score_slice.dtype)

    def _exteroceptive_block(self, centroid, score_slice):
        """
        Assemble one group's/cluster's exteroceptive state block according to
        the `state` mode, in the fixed order [prototype centroid | relational
        summary] -- the same order exteroceptive_dim is sized with in
        init_agents:
          - 'prototype': the raw hidden-space centroid alone.
          - 'relational': the fixed-size relational summary alone.
          - 'mixed': the concatenation of both.
        `centroid` is the group's/cluster's hidden-space centroid; `score_slice`
        is that group's/cluster's rows of prototypical similarity scores (fed to
        _relational_summary). Only the encodings the mode needs are computed.
        """
        if self.state_mode == 'prototype':
            return centroid
        summary = self._relational_summary(score_slice)
        if self.state_mode == 'relational':
            return summary
        # 'mixed': centroid first, then the relational summary, matching a
        # relational summary cast onto the centroid's device/dtype so the
        # concatenation is well-formed regardless of upstream dtype.
        return torch.cat(
            [centroid, summary.to(device=centroid.device, dtype=centroid.dtype)])

    def _call_confidence_decoder(self, decoder, scores, hidden_vectors, labels, query_mask, known_class_mask):
        """
        Invoke `decoder` with exactly the inputs its forward signature
        declares. Legacy prototypical decoders consume only `scores` (the
        known-class similarity slice); the Mahalanobis decoder consumes the raw
        hiddens plus the support structure (labels / query_mask /
        known_class_mask) to build its per-class Gaussians. The accepted-kwarg
        plan was cached for self.confidence_decoder in
        init_inference_neural_modules; the async-evaluation clone is the same
        class, so the same plan applies to it.
        """
        available = {
            'scores': scores,
            'hidden_vectors': hidden_vectors,
            'labels': labels,
            'query_mask': query_mask,
            'known_class_mask': known_class_mask,
        }
        if self._cd_accepts_kwargs:
            call_kwargs = available
        else:
            call_kwargs = {k: v for k, v in available.items() if k in self._cd_param_names}
        return decoder(**call_kwargs)

    def online_anomaly_detection(self, batch, logits, hiddens, one_hot_labels, query_mask):
        """
        Performs anomaly detection on the online batch.
        """
        if self.use_neural_AD:
            known_class_h_mask = self.get_known_classes_mask(batch, one_hot_labels)

            # Targeted fix for the "G2-near-G1" blind spot: at inference G1
            # carries zda_label 0, so get_known_classes_mask folds G1 columns
            # into the AD's known-set and a real zero-day sitting near a G1
            # prototype scores as non-anomalous. When enabled, drop the G1
            # columns so the detector scores only against true Known-class
            # prototypes. Skipped if it would empty the known-set.
            if self.exclude_g1_from_ad_known_set:
                candidate_mask = known_class_h_mask & ~self._g1_column_mask(known_class_h_mask.shape[0])
                if candidate_mask.any():
                    known_class_h_mask = candidate_mask

            try:
                zda_predictions = self._call_confidence_decoder(
                    self.confidence_decoder,
                    scores=logits[:, known_class_h_mask],
                    hidden_vectors=hiddens,
                    labels=batch.class_labels,
                    query_mask=query_mask,
                    known_class_mask=known_class_h_mask)
            except Exception as e:
                self.logger_instance.error(f'Confidence decoder error: {e}')
                raise RuntimeError(f'Confidence decoder error: {e}')
        
            predicted_zda_mask = (zda_predictions > self.ad_threshold).to(torch.bool).squeeze(-1)
        else:
            zda_predictions = batch.zda_labels[query_mask]
            predicted_zda_mask = zda_predictions.to(torch.bool).squeeze(-1)

        # Hard epistemic action: for any class whose AD oracle is active (granted
        # together with its label buy), stop using the IM's verdict and take the
        # ground-truth zda label (Known, since the class is already bought). Composes
        # with use_neural_AD -- a no-op when it is off (that branch already uses
        # ground truth). Overriding the whole query subset is safe: only the
        # online tail feeds downstream decisions/metrics.
        if self.env.ad_oracle_labels:
            codes = self.encoder.get_codes_for_labels(self.env.ad_oracle_labels)
            if codes:
                codes_t = torch.tensor(codes, device=self.device, dtype=batch.class_labels.dtype)
                oracle_mask = torch.isin(batch.class_labels[query_mask].squeeze(-1), codes_t)
                if oracle_mask.any():
                    gt = batch.zda_labels[query_mask]
                    zda_predictions = zda_predictions.clone()
                    zda_predictions[oracle_mask] = gt[oracle_mask].to(zda_predictions.dtype)
                    predicted_zda_mask = predicted_zda_mask.clone()
                    predicted_zda_mask[oracle_mask] = gt[oracle_mask].squeeze(-1).to(torch.bool)

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

        Returns, in addition to the correctness mask and tick-level accuracy,
        the per-sample predicted class and the logits slice the predictions
        came from -- callers (act_on_known_traffic) use these to group known
        samples by predicted class and compute a per-class-inference
        confidence, instead of the single tick-broadcast confidence this
        method also still sets on self.cs_classif_confidence (kept for
        tick-level W&B reporting in online_inference).
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

        self.cs_classif_confidence = self._cs_confidence_for_slice(
            interest_logits_slice, online_class_preds, number_of_known_classes)

        return known_correct_classification_mask, cs_acc, online_class_preds, interest_logits_slice, number_of_known_classes

    def _cs_confidence_for_slice(self, logits_slice, preds_slice, num_classes):
        """
        Closed-set classification confidence over an arbitrary subset of
        known-predicted samples (their logits + predicted class indices).
        Same four interchangeable strategies as before, just generalized so
        it can be applied either to a whole tick's known traffic (tick-level
        reporting) or to one predicted-class group within a tick (per-class-
        inference decisions in act_on_known_traffic).
        """
        n = logits_slice.shape[0]
        if n == 0:
            return torch.ones(1) * 10

        strategy = self.kwargs['intrusion_detection'].get('confidence_strategy', 'baseline')

        if strategy == 'entropy':
            probs = torch.softmax(logits_slice, dim=1)
            result = (probs * torch.log(probs + 1e-10)).sum(dim=1).mean().unsqueeze(-1)
        elif strategy == 'energy':
            result = torch.logsumexp(logits_slice, dim=1).mean().unsqueeze(-1)
        elif strategy == 'margin':
            probs = torch.softmax(logits_slice, dim=1)
            if probs.shape[1] > 1:
                top2 = torch.topk(probs, 2, dim=1).values
                result = (top2[:, 0] - top2[:, 1]).mean().unsqueeze(-1)
            else:
                result = probs.mean().unsqueeze(-1)
        else: # baseline
            non_choosed_mask = torch.ones(n, num_classes, device=logits_slice.device)
            non_choosed_mask[torch.arange(n, device=logits_slice.device), preds_slice] = 0
            mean_non_choosed_values = logits_slice[non_choosed_mask.to(torch.bool)].mean()
            mean_choosed_logits = logits_slice.max(1)[0].mean()
            # mean_non_choosed_values can land at/near zero or go negative
            # (these are raw logits, not probabilities), sending the ratio
            # to +-inf/0 and the log to NaN/Inf -- _clamp_confidence below
            # sanitises the result before it reaches callers.
            result = torch.log(mean_choosed_logits / mean_non_choosed_values).unsqueeze(-1)

        return self._clamp_confidence(result)

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


    def _zda_confidence_for_subset(self, probs):
        """
        Anomaly-detection confidence over an arbitrary subset of predicted-
        anomalous online samples (e.g. one collective-anomaly cluster's
        members). Thisis what act_on_unknown_clusters uses, so each cluster's
        decision sees its own members' confidence.

        Every member here is, by construction, predicted-anomalous (clusters
        are built only from the predicted-unknown subset), so the baseline
        strategy collapses to the mean anomaly probability of the cluster.
        """
        if probs.numel() == 0:
            return torch.ones(1) * 10

        probs = probs.view(-1, 1)
        strategy = self.kwargs['intrusion_detection'].get('confidence_strategy', 'baseline')

        if strategy == 'entropy':
            p = torch.cat([1 - probs, probs], dim=1)
            result = (p * torch.log(p + 1e-10)).sum(dim=1).mean().unsqueeze(-1)
        elif strategy == 'energy':
            p = probs.clamp(1e-10, 1 - 1e-10)
            l = torch.log(p / (1 - p))
            logits = torch.cat([-l, l], dim=1)
            result = torch.logsumexp(logits, dim=1).mean().unsqueeze(-1)
        elif strategy == 'margin':
            result = torch.abs(2 * probs - 1).mean().unsqueeze(-1)
        else: # baseline
            result = probs.mean().unsqueeze(-1)

        return self._clamp_confidence(result)

    def _clamp_confidence(self, value):
        """
        Single choke point for both confidence methods above. Confidence
        values are read raw by callers -- W&B scalars, cti_confidence_threshold
        comparisons in _select_unknown_cluster_action -- before they ever
        reach assembly_state_vector's own nan_to_num pass over the full state
        vector, so a NaN/Inf here (e.g. the baseline strategy's log of a
        degenerate logit ratio) needs sanitising at the source rather than
        relying on that later, blanket pass.

        Replacement values match assembly_state_vector's convention exactly
        (nan/posinf/neginf all -> 0.0): a degenerate confidence maps to the
        same neutral value the full-state pass would have produced anyway, so
        this only moves the sanitisation earlier without changing what the
        agent's LayerNorm'd proprioceptive block ends up seeing.
        """
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    def _decision_reward(self, accepted, group_rewards, accept_reward_scale=1.0, malicious_accept_penalty_scale=1.0):
        """
        Single reward rule shared by every DM decision, known-traffic group
        or unknown-traffic cluster alike: a decision is "accept" (let the
        group's traffic stand) or "block" (discard it).

        Accepted: per-flow, positive (benign) rewards are scaled by
        `accept_reward_scale` and negative (malicious) rewards are scaled by
        `malicious_accept_penalty_scale` -- so `accept_reward_scale` only
        discounts the upside of a *correct* accept, while
        `malicious_accept_penalty_scale` upscales (or, left at its default,
        leaves untouched) the downside of a *wrong* one. Both default to
        1.0 (no-op).
        
        Blocked:
        the benign reward the group would have earned is forgone (the
        malicious-content cost is zero, since it's blocked); blocking is
        never scaled.

        An epistemic (CTI-buying) decision reuses this same rule for its
        accept/block component; the caller subtracts the CTI price on top.
        """
        if accepted:
            positive = torch.relu(group_rewards)
            negative = group_rewards - positive
            return (accept_reward_scale * positive.sum() + malicious_accept_penalty_scale * negative.sum()).item()
        return 0

    def act_on_known_traffic(self, num_of_anomalies, num_known, hiddens, zda_mask, rewards,
                              class_preds, interest_logits_slice, number_of_known_classes,
                              true_label_names_known):
        """
        One DM decision per predicted closed-set class-inference group within
        this tick's known-traffic sub-batch, mirroring act_on_unknown_clusters's
        per-cluster loop: samples the IM assigned to the same class are
        grouped, and the agent accepts/rejects that specific class inference
        by looking at the hidden-space centroid of its members. Per-group
        confidence (cs_classif_confidence) and reward. The zda_confidence
        slot in this state vector is zeroed, since these samples were never
        evaluated for anomalousness -- the zero also signals to the agent
        that this state belongs to the known-traffic regime. Next state's
        exteroceptive part is the next group's centroid, chained sequentially
        within the tick exactly like the cluster loop.

        Every accepted group feeds `record_reappearances` for its reward side
        (net_values / reward_since_purchase); a blocked group never earns or
        costs the raw per-flow reward. The reappearance *counts* and their
        cti_shots/cti_misses split are tallied separately, once per tick by true
        label, in record_cti_reappearances -- so a bought-G2 sample the IM
        deemed anomalous (a miss, which never reaches this known-traffic path)
        is still counted.
        """
        if num_known == 0:
            return

        num_online = zda_mask.shape[0]
        known_hiddens = hiddens[-num_online:][~zda_mask]
        known_samples_costs = rewards[~zda_mask]

        unique_classes = torch.unique(class_preds)
        num_groups = unique_classes.shape[0]

        group_member_masks = [class_preds == cls for cls in unique_classes]
        group_centroids = [known_hiddens[mask].mean(dim=0) for mask in group_member_masks]
        # Exteroceptive state block per group: the raw centroid, the relational
        # summary of the group's similarity to the known prototypes, or (mixed)
        # both concatenated -- selected by the `state` mode (see
        # _exteroceptive_block).
        group_exteroceptive = [
            self._exteroceptive_block(centroid, interest_logits_slice[mask])
            for centroid, mask in zip(group_centroids, group_member_masks)
        ]

        classification_reward_total = 0.0
        last_action = None
        # Supervised classification-accuracy reward bookkeeping this tick: the
        # total accuracy reward added and the correct/total sample counts used
        # to report the tick's overall closed-set accuracy. All stay 0 -- and
        # no reward is added -- when the knob is off, keeping it hidden.
        classification_accuracy_reward_total = 0.0
        accuracy_correct_total = 0
        accuracy_samples_total = 0
        # Representation-scale diagnostic: L2 norm of each group's raw hidden
        # centroid (group_centroids, the exteroceptive block that enters the DM
        # value net unnormalised when relational_state is off). Tracks whether
        # the encoder's embedding norm -- and therefore the DQN's input scale --
        # is inflating over training, independently of any relational transform.
        centroid_norms = []
        # Value-net input scale actually fed to the DM, and the raw prototypical
        # logit (1/cdist) behind it. In relational_state mode the exteroceptive
        # block IS the relational summary (not the centroid above), so these are
        # the series that reveal a relational blow-up: exteroceptive_abs_max is
        # what the value net sees (should stay O(tens) after the log-score fix),
        # while proto_logit_raw_max is the underlying inverse-distance spike
        # (stays large regardless) -- the gap between them is the fix working.
        extero_abs_maxes = []
        proto_logit_raw_maxes = []

        for idx in range(num_groups):
            if self.intrusion_detection_kwargs['price_decay']: self.env.price_decay()

            member_mask = group_member_masks[idx]
            group_confidence = self._cs_confidence_for_slice(
                interest_logits_slice[member_mask], class_preds[member_mask], number_of_known_classes)

            centroid = group_exteroceptive[idx].unsqueeze(0)
            centroid_norms.append(group_centroids[idx].norm().item())
            extero_abs_maxes.append(group_exteroceptive[idx].abs().max().item())
            proto_logit_raw_maxes.append(interest_logits_slice[member_mask].max().item())
            state_vec = self.assembly_state_vector(
                centroid, num_of_anomalies, num_known,
                0.0, group_confidence.item(), self.env.current_budget)

            if self.intrusion_detection_kwargs['automatic_cs_acceptance']:
                action_signal = torch.tensor([0], device=self.device).long()
            else:
                # Known-traffic has no epistemic (CTI) semantics: the reward
                # logic only distinguishes accept (0) from not-accept (block).
                # Remap a self-chosen epistemic action (2) to block (1) BEFORE
                # it is scored, tallied, and stored, so the replay-buffer tuple
                # (state, action, reward, next_state) is consistent -- the
                # stored action is the block that actually happened, not a 2
                # that was silently treated as a block.
                action_signal = self._remap_epistemic_to_block(self.act(state_vec))

            last_action = action_signal

            self.env.steps_done += 1
            self.wb_tracker.step_counter += 1

            group_costs = known_samples_costs[member_mask]
            accepted_group = action_signal.item() == 0
            classification_reward = self._decision_reward(accepted_group, group_costs)

            # Pragmatic-decision tally for this group, keyed by its most-similar
            # (IM-predicted) class -- the finer-grained known-regime analogue of
            # the unknown-regime majority-label tally in act_on_unknown_clusters.
            # Counted in *samples* (group members), accumulated per-episode on
            # the env so its grand total is comparable to appearances; epistemic
            # actions (2) are not pragmatic and are skipped.
            most_similar_name = self._class_index_to_name(unique_classes[idx])
            if most_similar_name is not None:
                action_val = action_signal.item()
                member_count = int(member_mask.sum().item())
                if action_val == 0:
                    self.env.record_pragmatic_decision('known', most_similar_name, True, member_count)
                elif action_val == 1:
                    self.env.record_pragmatic_decision('known', most_similar_name, False, member_count)

            # Feed the empirical (pre-shaping) pragmatic reward of an accepted
            # group into the per-class running average that the next tick's
            # relational summaries read back as r_max / r_runnerup. Keyed by the
            # IM-predicted class -- the same index space as the prototype-logit
            # columns the summary indexes -- so lookup and update line up.
            if self.relational_state and accepted_group:
                self._update_relational_reward(unique_classes[idx], classification_reward)

            group_true_labels = [true_label_names_known[i] for i in member_mask.nonzero(as_tuple=False).squeeze(-1).tolist()]
            self.env.record_reappearances(group_true_labels, group_costs.tolist(), accepted=accepted_group)

            # Supervised classification-accuracy reward (hidden by default):
            # credit this known-traffic group weight * (fraction of its members
            # the IM classified correctly). Every member of a group shares the
            # same predicted class, so this fraction is that predicted class's
            # accuracy on the group; summed over the tick it is the overall
            # closed-set accuracy. A policy with more Known classes classifies
            # more accurately and so earns more, again crediting supervision on
            # its merits. No-op when the weight is 0 (default).
            if self.classification_accuracy_reward_weight > 0.0 and group_true_labels:
                group_pred_names = self.get_label_names_from_encoded_labels(class_preds[member_mask])
                group_correct = sum(1 for t, p in zip(group_true_labels, group_pred_names) if t == p)
                group_accuracy = group_correct / len(group_true_labels)
                accuracy_reward = self.classification_accuracy_reward_weight * group_accuracy
                classification_reward += accuracy_reward
                classification_accuracy_reward_total += accuracy_reward
                accuracy_correct_total += group_correct
                accuracy_samples_total += len(group_true_labels)

            self.env.current_budget += classification_reward

            if not self.intrusion_detection_kwargs['automatic_cs_acceptance']:
                new_state = state_vec.detach().clone()
                if idx < num_groups - 1:
                    new_state[:-PROPRIOCEPTIVE_STATE_SIZE] = group_exteroceptive[idx + 1]
                else:
                    new_state[:-PROPRIOCEPTIVE_STATE_SIZE] = -1 * torch.ones_like(new_state[:-PROPRIOCEPTIVE_STATE_SIZE])
                new_state[-1] = self.env.current_budget
                end_signal = torch.tensor([self.env.has_episode_ended()], device=self.device, dtype=torch.long)
                self.mitigation_agent.remember(
                    state_vec.detach(), action_signal,
                    torch.tensor([classification_reward], device=self.device),
                    new_state, end_signal, self.wb_tracker.step_counter)

            self.env.episode_rewards.append(classification_reward)
            self.env.episode_budgets.append(self.env.current_budget)

            classification_reward_total += classification_reward

        if self.wbt:
            known_scalars = {
                AGENT+'/'+'generic_reward': classification_reward_total,
                AGENT+'/'+'classification_reward': classification_reward_total,
                AGENT+'/'+'budget': self.env.current_budget,
                AGENT+'/'+'known_traffic_groups': num_groups,
            }
            # Representation-scale diagnostic (known-traffic regime).
            if centroid_norms:
                known_scalars['diagnostics/centroid_norm_known_mean'] = \
                    sum(centroid_norms) / len(centroid_norms)
                known_scalars['diagnostics/centroid_norm_known_max'] = max(centroid_norms)
            if extero_abs_maxes:
                known_scalars['diagnostics/exteroceptive_abs_max_known'] = max(extero_abs_maxes)
            if proto_logit_raw_maxes:
                known_scalars['diagnostics/proto_logit_raw_max_known'] = max(proto_logit_raw_maxes)
            # Supervision-value component: total accuracy reward added this tick
            # and the tick's overall closed-set accuracy behind it. Emitted only
            # when the knob is on, so the series stay hidden by default.
            if self.classification_accuracy_reward_weight > 0.0:
                known_scalars[AGENT+'/'+'classification_accuracy_reward'] = \
                    classification_accuracy_reward_total
                known_scalars[AGENT+'/'+'classification_accuracy'] = \
                    (accuracy_correct_total / accuracy_samples_total
                     if accuracy_samples_total > 0 else 0.0)
            self.reporter.log_scalars(known_scalars, step=self.wb_tracker.step_counter)

        # Reward beliefs (the per-class spurious accept-rewards feeding the
        # relational exteroceptive summary) are updated by the accepted groups
        # above, so emit their current state once the tick's decisions are in.
        self._log_reward_beliefs()

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
    
    def _remap_epistemic_to_block(self, action):
        """
        Ablation helper: turns a CTI-purchase action (2) into a block (1).
        Used by the scripted-CTI ablation modes in `_select_unknown_cluster_action`
        to neutralize the agent's own choice of action 2 outside their forced slot.
        """
        if action == 2:
            return torch.tensor([1], device=self.device).long()
        return action

    def _can_afford_cti(self):
        """
        Affordability guard for the greedy-CTI ablation: a CTI purchase must
        not bankrupt the agent. Bankruptcy is `current_budget < min_budget`
        (see NewTigerEnvironment.has_episode_ended), so a buy is affordable
        only if paying its price leaves the budget at or above min_budget
        (equal survives; strictly below dies).

        This runs before the cluster's majority label -- and thus the exact
        label that will be charged -- is known, so it guards the worst case:
        the most expensive CTI action currently on offer (the priciest
        purchasable G2 listed in env.current_cti_options). Under
        hard_epistemic_action a buy still costs that single price -- the AD oracle
        rides on it for free -- so no extra candidates need folding in. If even
        the priciest keeps budget >= min_budget, any actual buy this tick is safe.

        When bankruptcy termination is disabled (disable_budget_bankrupt_
        termination), there is no "dying", so every buy is affordable.
        """
        if self.env.disable_budget_bankrupt_termination:
            return True

        # Purchasable G2s (skip the high-cost placeholders update_cti_options
        # inserts when no G2 remains).
        prices = [
            price for label, price in self.env.current_cti_options.items()
            if not str(label).startswith('placeholder_')
        ]

        if not prices:
            return False

        worst_case_price = max(prices)
        return (self.env.current_budget - worst_case_price) >= self.env.min_budget

    def _select_unknown_cluster_action(self, state_vec, cluster_confidence=None):
        """
        Chooses the action for one unknown-traffic cluster.

        The default path is the learned policy (`self.act`) -- this is what
        runs in practice, since `greedy_cti`, `cti_period`, and
        `no_epistemic_actions` are all off by default in
        tiger/config/default.yaml (`greedy_cti: False`, `cti_period: -1`,
        `no_epistemic_actions: false`). The three branches below are
        mutually-exclusive ABLATION KNOBS, set via tiger/config overrides,
        used to study the value of the epistemic-action channel itself
        rather than the learned policy:

        - cti_period != -1: a scripted periodic-CTI policy. Forces action 2
          every `cti_period` steps, provided a G2 class is still available to
          buy (`epistemic_actions_available == 1`); in every other step (and
          whenever no G2 class remains, even on a periodic step), the agent is
          still queried, but any 2 it returns is remapped to 1 (block), since
          CTI is reserved for the periodic slot.
        - greedy_cti: buys (action 2) whenever there is still an unbought
          G2 class available (`self.env.epistemic_actions_available == 1`)
          AND the buy is affordable -- i.e. paying the price would not
          bankrupt the agent (`_can_afford_cti`, budget stays >= min_budget).
          When a G2 is available but the buy is unaffordable, or no G2
          remains, it behaves like the periodic case (query + remap 2 -> 1),
          so a broke greedy agent blocks instead of buying itself to death.
        - fixed_threshold_cti: buys (action 2) whenever the cluster's
          confidence is below cti_confidence_threshold (and a G2 is available
          and affordable); otherwise queries the agent with 2 remapped to 1.
        - no_epistemic_actions: queries the agent normally but remaps any 2
          it returns to 1, fully disabling epistemic actions as a no-CTI
          baseline.

        Distinct from those ablations, guard_learned_cti_affordability is an
        opt-in guard on the LEARNED path itself (default off): when on, a CTI
        buy the agent chooses for itself but cannot afford is remapped to a
        block, so the learned policy is held to the same affordability
        constraint the scripted baselines are (`_can_afford_cti`). Off by
        default -- and a no-op whenever bankruptcy termination is disabled --
        so the baseline lets the agent's own action 2 through and learn the
        budget constraint from the bankruptcy consequence itself.

        Only when none of the ablations is active does the agent's own action
        2 survive -- and then only subject to the optional affordability guard.
        """
        # Coerce to int before the -1 guard: config numbers travel as JSON/
        # form strings (see int(...) at every other use site, e.g.
        # update_target_freq / agent_memory_size), so cti_period can arrive as
        # the string "-1". Comparing that raw string to the int -1 is always
        # True, which would wrongly enter the periodic-CTI branch and (since
        # step % int("-1") == 0 for every step) force action 2 on every step --
        # bypassing no_epistemic_actions entirely.
        cti_period = int(self.intrusion_detection_kwargs.get('cti_period', -1))
        if cti_period != -1:
            if self.env.steps_done % cti_period == 0 \
                    and self.env.epistemic_actions_available == 1 \
                        and self._can_afford_cti():
                return torch.tensor([2], device=self.device).long()
            return self._remap_epistemic_to_block(self.act(state_vec))

        if self.intrusion_detection_kwargs.get('greedy_cti'):
            if self.env.epistemic_actions_available == 1 and self._can_afford_cti():
                return torch.tensor([2], device=self.device).long()
            return self._remap_epistemic_to_block(self.act(state_vec))

        # fixed_threshold_cti (RC 3.6(b)): buy when the cluster's confidence is
        # below cti_confidence_threshold, else defer the pragmatic choice.
        if self.intrusion_detection_kwargs.get('fixed_threshold_cti'):
            if cluster_confidence is not None and cluster_confidence > self.cti_confidence_threshold \
                    and self.env.epistemic_actions_available == 1 \
                    and self._can_afford_cti():
                return torch.tensor([2], device=self.device).long()
            return self._remap_epistemic_to_block(self.act(state_vec))

        action = self.act(state_vec)
        if self.intrusion_detection_kwargs['no_epistemic_actions']:
            return self._remap_epistemic_to_block(action)
        if self.env.epistemic_actions_available == 0:
            # No G2 class left to buy: a CTI purchase here would be a no-op,
            # so it's not offered as a real choice -- treat it as a block.
            return self._remap_epistemic_to_block(action)
        # Optional affordability guard for the LEARNED policy (default OFF).
        # When on, a self-chosen CTI buy the agent can't cover is remapped to a
        # block, exactly as the scripted greedy/periodic/threshold baselines
        # are guarded. The remap only touches action 2, and the returned action
        # is what gets executed and stored in replay (see mitigation_agent.
        # remember below), so the agent learns from the block it actually took.
        # When off (and always a no-op while bankruptcy termination is
        # disabled), action 2 survives and the agent must learn the budget
        # constraint from the bankruptcy consequence itself.
        if self.intrusion_detection_kwargs.get('guard_learned_cti_affordability') \
                and not self._can_afford_cti():
            return self._remap_epistemic_to_block(action)
        return action

    def act_on_unknown_clusters(self, clusters_oh, centroids, missing, num_anom, num_known, zda_mask, rewards, online_anomaly_probs, true_label_names_zda, anomalous_logits=None):
        """
        Performs mitigation actions (block/pass/CTI) on detected unknown clusters.
        Each cluster's decision uses its own members' zda_confidence (via
        _zda_confidence_for_subset). The cs_classif_confidence slot in this
        state vector is zeroed, since these samples were never classified
        into a known class -- the zero also signals to the agent that this
        state belongs to the unknown-cluster regime.

        `anomalous_logits` are the predicted-anomalous online samples' rows of
        prototypical similarity scores (same ordering as clusters_oh's rows),
        used only when `relational_state` is enabled to build each cluster's
        relational exteroceptive state block.
        """
        num_identified = centroids[~missing].shape[0]
        anomalous_rewards = rewards[zda_mask]
        rewards_per_cluster = (clusters_oh * anomalous_rewards.unsqueeze(-1)).sum(0)

        # clusters_oh's rows are the predicted-anomalous online samples, in
        # the same order as online_anomaly_probs[zda_mask] (and, by the same
        # construction, true_label_names_zda) -- so indexing all of them by
        # the same cluster column lines a cluster up with its members' own
        # anomaly probabilities and true labels (see collective_anomaly_detection).
        anomalous_probs = online_anomaly_probs[zda_mask].view(-1, 1)
        non_missing_columns = (~missing).nonzero(as_tuple=False).squeeze(-1)

        # Exteroceptive state block per cluster: the raw centroid, the
        # relational summary of the cluster's similarity to the known
        # prototypes, or (mixed) both concatenated -- selected by the `state`
        # mode (see _exteroceptive_block).
        cluster_centroids = centroids[~missing]
        cluster_exteroceptive = [
            self._exteroceptive_block(
                cluster_centroids[i],
                anomalous_logits[clusters_oh[:, non_missing_columns[i]].bool()])
            for i in range(num_identified)
        ]

        clustering_reward = 0
        epistemic_actions_taken = 0
        wasted_epistemic_actions_taken = 0
        epistemic_costs = 0
        rewards_per_accepted_clusters = 0
        rewards_per_blocked_clusters = 0
        # Per-cluster zda_confidence values this tick, for threshold calibration.
        cluster_confidences = []
        # Sum of per-cluster impurity (1 - purity) this tick, used both to
        # charge the cluster-impurity penalty (when enabled) and to report the
        # mean impurity. Stays 0.0 -- and no penalty is applied -- when the knob
        # is off, so the component is fully hidden by default.
        cluster_impurity_sum = 0.0
        # Representation-scale diagnostic: L2 norm of each cluster's raw hidden
        # centroid (cluster_centroids, the exteroceptive block that enters the
        # DM value net unnormalised when relational_state is off). Same purpose
        # as in act_on_known_traffic, for the unknown/zero-day regime -- where
        # echo/doorlock live until bought, so this is the scale most relevant to
        # the over-blocking of those classes.
        centroid_norms = []
        # See act_on_known_traffic: value-net input scale actually fed to the DM
        # (the relational summary when relational_state is on) and the raw
        # prototypical logit (1/cdist) behind it, for the unknown/zero-day regime.
        extero_abs_maxes = []
        proto_logit_raw_maxes = []

        for idx in range(num_identified):
            if self.intrusion_detection_kwargs['price_decay']: self.env.price_decay()
            accepted_cluster = False
            epistemic_action = False

            column = non_missing_columns[idx]
            member_mask = clusters_oh[:, column].bool()
            cluster_zda_confidence = self._zda_confidence_for_subset(anomalous_probs[member_mask])
            cluster_confidences.append(cluster_zda_confidence.item())

            centroid_norms.append(cluster_centroids[idx].norm().item())
            extero_abs_maxes.append(cluster_exteroceptive[idx].abs().max().item())
            if anomalous_logits is not None:
                proto_logit_raw_maxes.append(anomalous_logits[member_mask].max().item())
            state_vec = self.assembly_state_vector(
                cluster_exteroceptive[idx].unsqueeze(0), num_anom, num_known,
                cluster_zda_confidence.item(), 0.0, self.env.current_budget)

            action = self._select_unknown_cluster_action(state_vec, cluster_zda_confidence.item())

            # Ground-truth labels of this cluster's members -- needed both
            # for the accept-path bookkeeping below and, on the epistemic
            # path, to target the CTI purchase at this cluster's actual
            # majority class rather than at whatever G2 happens to be next
            # in the curriculum list (the centroid is exteroceptive and can
            # be spurious/mixed-class, so the *targeted* label is decided
            # by majority vote among the cluster's true labels, not by the
            # centroid itself). Computed BEFORE the action is consumed because
            # the hard_g2s remap below keys off the targeted (majority) label.
            member_labels = [true_label_names_zda[i] for i in member_mask.nonzero(as_tuple=False).squeeze(-1).tolist()]

            # Majority true label AND its purity (fraction of the cluster that
            # is the majority class), computed once per cluster here so it can
            # feed both the epistemic path (CTI targeting + RC 3.4 quality
            # coupling) and the cluster-impurity penalty below. purity is the
            # hidden cluster-quality signal: a messy, low-purity cluster both
            # yields worse intelligence when bought and, when the penalty is
            # enabled, costs the policy more for having left it unresolved.
            if member_labels:
                majority_label, majority_count = Counter(member_labels).most_common(1)[0]
                cluster_purity = majority_count / len(member_labels)
            else:
                majority_label, cluster_purity = None, None

            # hard_g2s filter: a would-be CTI buy (action 2) whose targeted
            # label -- the majority true label computed just above, i.e. the
            # exact class perform_epistemic_action would purchase -- is on the
            # hard_g2s blocklist is remapped to a block (1). This is what lets a
            # greedy_cti run be turned into a "buy only the G2s that matter"
            # oracle: greedy would buy every available G2, and this carves out
            # the ones that must be blocked instead. A no-op when hard_g2s is
            # empty (default) or the target is off-list; the remapped action is
            # what feeds the accept/reward/tally logic and replay below, so the
            # agent is credited with the block it actually took.
            if action == 2 and majority_label is not None and majority_label in self.hard_g2s:
                action = self._remap_epistemic_to_block(action)

            if action == 0:
                accepted_cluster = True
            elif action == 2:
                epistemic_action = True
                accepted_cluster = not self.intrusion_detection_kwargs['epistemic_is_blocking']

            current_reward = self._decision_reward(
                accepted_cluster, rewards_per_cluster[~missing][idx],
                accept_reward_scale=self.unknown_accept_reward_scale,
                malicious_accept_penalty_scale=self.unknown_malicious_accept_penalty_scale)

            # Pragmatic-decision tally for this cluster, keyed by its majority
            # true label. Counted in *samples* (cluster members), accumulated
            # per-episode on the env so its grand total is comparable to
            # appearances; epistemic buys (action 2) are excluded -- they are
            # the epistemic channel, tracked separately as "Epistemic Actions
            # taken".
            if majority_label is not None:
                action_val = action.item()
                member_count = int(member_mask.sum().item())
                if action_val == 0:
                    self.env.record_pragmatic_decision('unknown', majority_label, True, member_count)
                elif action_val == 1:
                    self.env.record_pragmatic_decision('unknown', majority_label, False, member_count)

            if accepted_cluster:
                member_rewards_tensor = anomalous_rewards[member_mask]
                positive = torch.relu(member_rewards_tensor) * self.unknown_accept_reward_scale
                negative = (member_rewards_tensor - torch.relu(member_rewards_tensor)) * self.unknown_malicious_accept_penalty_scale
                scaled_member_rewards = (positive + negative).tolist()
                self.env.record_unsupervised_pass(member_labels, scaled_member_rewards)

            # Cluster-impurity penalty (hidden by default): charge this
            # cluster's decision weight * (1 - purity). A policy whose CTI
            # purchases have thinned the unknown-class pool leaves purer
            # collective-anomaly clusters and is penalised less, so the reward
            # credits better perception rather than penalising the unsupervised
            # baseline directly. No-op when the weight is 0 (default).
            if cluster_purity is not None:
                cluster_impurity = 1.0 - cluster_purity
                cluster_impurity_sum += cluster_impurity

                if self.cluster_impurity_penalty_weight > 0.0:
                    current_reward -= self.cluster_impurity_penalty_weight * cluster_impurity

            if epistemic_action:
                updates_dict = self.perform_epistemic_action(
                    majority_label, purity=cluster_purity,
                    confidence=cluster_zda_confidence.item())
                current_reward -= updates_dict['price_payed']
                # Instant delivery of a real acquisition: log the corruption it
                # arrived with (delayed deliveries log the same in
                # _poll_cti_deliveries). No-op-valued (0.0 / 1.0) in the clean
                # regime; only emitted for genuine acquisitions.
                if updates_dict['updated_label'] is not None and self.wbt \
                        and self.env.cti_delivery.enabled:
                    lbl = updates_dict['updated_label']
                    self.reporter.log_scalars({
                        f'cti_effective_noise/{lbl}': self.env.cti_delivery.noise_for(lbl),
                        f'cti_capture/{lbl}': self.env.cti_delivery.capture_for(lbl),
                    }, step=self.wb_tracker.step_counter)
                if updates_dict.get('wasted', False):
                    wasted_epistemic_actions_taken += 1
                    # The buy acquired nothing -- penalise it beyond the price
                    # already paid, so a policy that reads the cluster before
                    # buying beats one that buys blindly.
                    current_reward -= self.useless_epistemic_penalty

            self.env.current_budget += current_reward
            next_state = state_vec.detach().clone()
            
            if idx < num_identified - 1:
                next_state[:-PROPRIOCEPTIVE_STATE_SIZE] = cluster_exteroceptive[idx+1]
            else:
                next_state[:-PROPRIOCEPTIVE_STATE_SIZE] = -1 * torch.ones_like(next_state[:-PROPRIOCEPTIVE_STATE_SIZE])

            # env state already reflects any epistemic action taken above
            # (perform_epistemic_action ran before this), so re-reading the
            # acquired-CTI fraction here captures the post-buy bump: the buy's
            # next_state shows the higher ownership its purchase just produced,
            # which is what lets the bootstrap credit that value to action 2.
            next_state[-3] = self.env.acquired_cti_fraction()
            next_state[-2] = self.env.epistemic_actions_available
            next_state[-1] = self.env.current_budget

            self.env.steps_done += 1
            self.wb_tracker.step_counter += 1

            end_signal = torch.tensor([self.env.has_episode_ended()], device=self.device, dtype=torch.long)

            self.mitigation_agent.remember(state_vec.detach(), action, current_reward, next_state, end_signal, self.wb_tracker.step_counter)
            self.env.episode_rewards.append(current_reward.item() if hasattr(current_reward, 'item') else current_reward)
            self.env.episode_budgets.append(self.env.current_budget)

            reward_val = current_reward.item() if hasattr(current_reward, 'item') else current_reward

            clustering_reward += reward_val
            epistemic_actions_taken += int(epistemic_action)
            epistemic_costs += reward_val if epistemic_action else 0
            rewards_per_accepted_clusters += reward_val if accepted_cluster else 0
            rewards_per_blocked_clusters += reward_val if not accepted_cluster else 0

        if len(centroids[~missing]) > 0 and self.wbt:

            cluster_scalars = {
                AGENT+'/'+'generic_reward': clustering_reward,
                AGENT+'/'+'clustering_reward': clustering_reward,
                AGENT+'/'+'budget': self.env.current_budget,
                AGENT+'/'+'Epistemic Actions taken': epistemic_actions_taken,
                AGENT+'/'+'Wasted Epistemic Actions taken': wasted_epistemic_actions_taken,
                AGENT+'/'+'epistemic_costs': epistemic_costs,
                AGENT+'/'+'rewards_per_accepted_clusters': rewards_per_accepted_clusters,
                AGENT+'/'+'rewards_per_blocked_clusters': rewards_per_blocked_clusters,
                AGENT+'/'+'mean_cluster_impurity': cluster_impurity_sum / num_identified
            }
            # Representation-scale diagnostic (unknown/zero-day regime).
            if centroid_norms:
                cluster_scalars['diagnostics/centroid_norm_unknown_mean'] = \
                    sum(centroid_norms) / len(centroid_norms)
                cluster_scalars['diagnostics/centroid_norm_unknown_max'] = max(centroid_norms)
            if extero_abs_maxes:
                cluster_scalars['diagnostics/exteroceptive_abs_max_unknown'] = max(extero_abs_maxes)
            if proto_logit_raw_maxes:
                cluster_scalars['diagnostics/proto_logit_raw_max_unknown'] = max(proto_logit_raw_maxes)
            # Supervision-value component: total impurity penalty deducted this
            # tick and the mean cluster impurity behind it. Emitted only when
            # the knob is on, so the series stay hidden in the default regime.
            if self.cluster_impurity_penalty_weight > 0.0:
                cluster_scalars[AGENT+'/'+'cluster_impurity_penalty'] = \
                    self.cluster_impurity_penalty_weight * cluster_impurity_sum
                
            # Per-cluster zda_confidence distribution this tick, for threshold
            # calibration: summary scalars plus a histogram of the raw values.
            if cluster_confidences:
                import wandb
                conf_t = torch.tensor(cluster_confidences)
                cluster_scalars[AGENT+'/'+'cluster_zda_confidence_mean'] = conf_t.mean().item()
                cluster_scalars[AGENT+'/'+'cluster_zda_confidence_min'] = conf_t.min().item()
                cluster_scalars[AGENT+'/'+'cluster_zda_confidence_max'] = conf_t.max().item()
                cluster_scalars[AGENT+'/'+'cluster_zda_confidence_std'] = \
                    conf_t.std(unbiased=False).item()
            self.reporter.log_scalars(cluster_scalars, step=self.wb_tracker.step_counter)

    def _log_epistemic_delays(self, present_labels):
        """
        For every G2 bought in an earlier tick whose class now appears among
        this tick's online true labels (`present_labels`), emit
        epistemic_delay/<label> = current step_counter - purchase step_counter,
        then stop tracking it. This is the CTI-acquisition latency: how many DM
        steps elapse between paying for a class and the model next actually
        seeing that class on the wire (where it can start being classified /
        trained on as a Known). One point per acquisition -> a no-aggregation
        line per G2.
        """
        closed = {}
        for label, buy_step in list(self._pending_epistemic_delays.items()):
            if label in present_labels:
                closed[f'epistemic_delay/{label}'] = self.wb_tracker.step_counter - buy_step
                del self._pending_epistemic_delays[label]
        if closed and self.wbt:
            self.reporter.log_scalars(closed, step=self.wb_tracker.step_counter)

    def _log_zda_probs(self, true_label_names, online_anomaly_probs):
        """
        For every G2  this episode whose traffic is present this tick, 
        compute that class's own samples' anomaly probabilities 
        -- selected by TRUE label, over all online samples regardless of the known/anomaly
        split -- and emit supervised_scores/<label>. 

        `online_anomaly_probs` is zda_predictions[-num_online:] (per-online-sample
        anomaly probability), aligned element-for-element with `true_label_names`.
        """
        scores = {}
        for label in set(true_label_names):
            stats = self.env.acquired_g2_stats.get(label)
            if stats is None:
                continue
            
            member_idx = [i for i, name in enumerate(true_label_names) if name == label]
            member_probs = online_anomaly_probs[member_idx]
            if not stats['bought']:
                scores[f'unsupervised_scores/{label}'] = member_probs.mean().unsqueeze(-1).item()
            else:
                scores[f'supervised_scores/{label}'] = member_probs.mean().unsqueeze(-1).item()

        if scores and self.wbt:
            self.reporter.log_scalars(scores, step=self.wb_tracker.step_counter)

    def online_inference(self, online_batch):
        """
        Executes the online inference loop.
        """
        
        self.classifier.eval()
        self.confidence_decoder.eval()

        # Deliver any purchased CTI whose (delayed) arrival is now due, before
        # this tick reads the knowledge state -- so a class delivered this tick
        # is treated as Known from here on. No-op in the instant-delivery regime.
        if self.agency:
            self._poll_cti_deliveries()

        online_batch_tuple = self.prepare_online_batch(online_batch)
        if online_batch_tuple is None: return
        
        merged_batch, merged_query_mask, accuracy_mask = online_batch_tuple
        
        with torch.no_grad():
            with self.profile("onl_inf_forward_pass"):
                logits, hiddens, predicted_kernel = self.infer(self.classifier, merged_batch, self.current_known_classes_count, query_mask=merged_query_mask)
            
            one_hot_labels = self.get_oh_labels(merged_batch, self.current_known_classes_count)

            with self.profile("onl_inf_AD"):
                zda_predictions, predicted_zda_mask = self.online_anomaly_detection(merged_batch, logits, hiddens, one_hot_labels, merged_query_mask)
        
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
        true_label_names = self.get_label_names_from_encoded_labels(merged_batch.class_labels[-num_online:].squeeze(-1))
        # Episode-wide appearance tally over every online sample's true label,
        # regardless of how it was predicted -- counts all classes (Knowns,
        # G1s, G2s) on the wire this episode.
        self.env.record_appearances(true_label_names)

        # Post-buyin CTI hit/miss accounting per already-bought G2, split by the
        # IM's anomaly-detection verdict (deemed Known = shot, still deemed
        # anomaly = miss), independent of the DM's later decision. Counted by
        # TRUE label over the whole online batch so misses (which flow to the
        # unknown-cluster path, not act_on_known_traffic) are included, keeping
        # reappearances == cti_shots + cti_misses. Per-episode scalars, reported
        # at episode end. Runs before act_on_unknown_clusters performs this
        # tick's own buys, so a class bought this very tick isn't counted until
        # it next reappears (matching "after buying the label").
        if self.agency:
            self.env.record_cti_reappearances(true_label_names, pred_online_zda_mask.tolist())

        # Close out CTI-acquisition latencies here -- before act_on_unknown_clusters
        # can perform this tick's own purchases -- so a class bought this very tick
        # never self-triggers a zero delay; only a *later* tick that actually
        # carries the bought class on the wire closes it.
        if self.agency and self._pending_epistemic_delays:
            self._log_epistemic_delays(set(true_label_names))

        # Anomaly-detection probabilities per G2s.
        # Computed here so it sees every occurrence of the
        # class, independent of the known/anomaly split downstream.
        if self.agency:
            self._log_zda_probs(true_label_names, zda_predictions[-num_online:])

        _, cs_acc, class_preds, interest_logits_slice, number_of_known_classes = \
            self.perform_cs_inference(merged_batch, logits, pred_online_zda_mask, num_online, num_known)


        kr_metrics = {}
        if num_known > 0:
            true_label_names_known = [name for name, is_zda in zip(true_label_names, pred_online_zda_mask.tolist()) if not is_zda]
            pred_label_names_known = self.get_label_names_from_encoded_labels(class_preds)
            classification_metrics = self.env.record_classification_stats(true_label_names_known, pred_label_names_known)
            if self.wbt and classification_metrics:
                self.reporter.log_scalars(classification_metrics, step=self.wb_tracker.step_counter)
            if self.agency:
                self.act_on_known_traffic(
                    num_anom, num_known, hiddens, pred_online_zda_mask, rewards,
                    class_preds, interest_logits_slice, number_of_known_classes,
                    true_label_names_known)

        if num_anom > 0:
            with self.profile("onl_inf_CAD"):
                clusters_oh, centroids, missing, kr_metrics = self.collective_anomaly_detection(merged_batch, predicted_kernel, one_hot_labels, pred_online_zda_mask, num_online, hiddens)
            if self.agency:
                online_anomaly_probs = zda_predictions[-num_online:]
                true_label_names_zda = [name for name, is_zda in zip(true_label_names, pred_online_zda_mask.tolist()) if is_zda]
                # Anomalous online samples' prototypical-score rows, in the
                # same order as clusters_oh's rows -- used by relational_state.
                anomalous_logits = logits[-num_online:][pred_online_zda_mask]
                self.act_on_unknown_clusters(clusters_oh, centroids, missing, num_anom, num_known, pred_online_zda_mask, rewards, online_anomaly_probs, true_label_names_zda, anomalous_logits=anomalous_logits)

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
            }
            all_metrics.update(ad_metrics)
            all_metrics.update(cs_metrics)
            all_metrics.update(kr_metrics)

            self.reporter.log_scalars(all_metrics, step=self.wb_tracker.step_counter)
            
        self.classifier.train()
        self.confidence_decoder.train()

        if self.agency and self.env.has_episode_ended():
            if self.wbt:
                steps = self.env.steps_done
                episode_return = torch.Tensor(self.env.episode_rewards).sum()
                # Episode outcome flags, each gated by its termination knob so a
                # flag is set only when that condition actually ends the episode.
                # With both win/bankrupt termination on, exactly one is true.
                # Logged as 0/1 so their means read as win/failure/timeout rate.
                ended_bankrupt = (not self.env.disable_budget_bankrupt_termination) \
                    and self.env.current_budget < self.env.min_budget
                ended_timeout = steps >= self.env.max_episode_steps
                ended_win = (not self.env.disable_budget_win_termination) \
                    and self.env.current_budget > self.env.max_budget
                episode_metrics = {
                    'episode_count': self.episode_count,
                    'mean_episode_reward': torch.Tensor(self.env.episode_rewards).mean(),
                    'sum_episode_rewards': episode_return,
                    # Uncapped per-step return: unlike sum_episode_rewards this
                    # is not bounded by the termination threshold, so a more
                    # efficient (e.g. label-buying) agent shows a higher value.
                    'return_per_step': episode_return / max(1, steps),
                    'mean_episode_budget': torch.Tensor(self.env.episode_budgets).mean(),
                    'final_episode_budget': self.env.current_budget,
                    'epistemic_actions_per_episode': self.env.epistemic_actions,
                    'wasted_epistemic_actions_per_episode': self.env.wasted_epistemic_actions,
                    # CTI delivery fidelity (RC 3.4): how many real acquisitions
                    # this episode arrived imperfectly (any label-noise, partial
                    # capture, or delivery delay), and their share of all real
                    # acquisitions. Both 0 in the clean regime.
                    'cti_corrupt_buys_per_episode': self.env.corrupt_cti_buys,
                    'cti_bad_buy_rate': (self.env.corrupt_cti_buys / self.env.total_cti_buys
                                         if self.env.total_cti_buys > 0 else 0.0),
                    'steps_per_episode': steps,
                    'episode_outcome_win': int(ended_win),
                    'episode_outcome_bankrupt': int(ended_bankrupt),
                    'episode_outcome_timeout': int(ended_timeout),
                }
                # Fixed-key per-G2 CTI-ROI series: same 7 labels every run
                # (knowledge is static), so these render as 7 line plots
                # under each of the wandb sections below. net_values tracks
                # only the reward earned on accepted reappearances *after*
                # buyin -- it never nets out the CTI purchase price, so a
                # benign G2 can be at worst zero (if blocked/absent after
                # buyin) but never negative. The price paid to acquire the
                # CTI is reported separately under price_payed/<label>, so the
                # cost of buyin is visible without contaminating the reward
                # series.
                for label, stats in self.env.acquired_g2_stats.items():
                    episode_metrics[f'reappearances/{label}'] = stats['reappearances']
                    episode_metrics[f'net_values/{label}'] = stats['reward_since_purchase']
                    episode_metrics[f'price_payed/{label}'] = stats['price_paid']
                    # Post-buyin IM verdict split of the reappearances:
                    # cti_shots + cti_misses == reappearances.
                    episode_metrics[f'cti_shots/{label}'] = stats['cti_shots']
                    episode_metrics[f'cti_misses/{label}'] = stats['cti_misses']
                    # Reencounters after the AD oracle became active (granted with the label buy).
                    episode_metrics[f'oracled_reappearances/{label}'] = stats['oracled_reappearances']
                # Pre-purchase (unsupervised) per-G2 net cost/reward: what
                # accepting that G2's traffic cost/earned this episode while
                # it was still unbought, i.e. exactly the gap a no-epistemic-
                # actions baseline would also have paid/earned.
                for label, net_value in self.env.unsupervised_costs.items():
                    episode_metrics[f'unsupervised_costs/{label}'] = net_value
                # Per-class net value for the non-G2 classes (Knowns, G1s),
                # tracked from episode init -- how much each known class's
                # accepted traffic earned/cost this episode. G2 net_values are
                # emitted above (post-buyin) from acquired_g2_stats; together
                # the two loops cover every class exactly once.
                for label, net_value in self.env.net_values.items():
                    episode_metrics[f'net_values/{label}'] = net_value
                # Per-class episode appearance counts over every class on the
                # wire (Knowns, G1s and G2s, the latter both before and after
                # being bought).
                for label, count in self.env.appearances.items():
                    episode_metrics[f'appearances/{label}'] = count
                # Per-class pragmatic-decision sample counts this episode, keyed
                # like the per-tick decisions that fed them (known by predicted
                # class, unknown by cluster majority label). Counted in wire
                # samples, so summing all four over every class recovers total
                # appearances minus the epistemically-bought cluster samples.
                for label, count in self.env.known_acceptances.items():
                    episode_metrics[f'known_acceptances/{label}'] = count
                for label, count in self.env.known_blocks.items():
                    episode_metrics[f'known_blocks/{label}'] = count
                for label, count in self.env.unknown_acceptances.items():
                    episode_metrics[f'unknown_acceptances/{label}'] = count
                for label, count in self.env.unknown_blocks.items():
                    episode_metrics[f'unknown_blocks/{label}'] = count
                self.reporter.log_scalars(episode_metrics, step=self.wb_tracker.step_counter)
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

    def assembly_state_vector(self, centroid, num_anom, num_known, zda_confidence, cs_classif_confidence, curr_budget):
        """
        Assembles the state vector for the agent. zda_confidence and
        cs_classif_confidence are passed in explicitly by the caller: each
        decision carries the confidence of the specific cluster or class-
        inference-group it is actually about in the slot relevant to its
        regime, and zero in the other slot, so the zero/non-zero pair tells
        the agent which regime (known vs. unknown traffic) this state
        belongs to.

        The seven proprioceptive channels are, in order: anomaly count, ZDA
        confidence, known count, classification confidence, acquired-CTI
        fraction, CTI-available flag, budget. With proprio_feature_scaling on,
        each is normalised per-feature here rather than by the net's pooled
        LayerNorm(PROPRIOCEPTIVE_STATE_SIZE): the two unbounded channels are
        squashed (counts via log1p, budget via the floor-anchored or
        running-|budget| tanh in _proprio_budget_feature), while the two
        already-bounded confidences, the already-bounded acquired-CTI fraction
        (in [0, 1]) and the boolean flag are left exactly as-is -- crucially
        preserving the structural zero in whichever confidence slot marks the
        off-regime. The net's proprio_norm becomes an Identity in this mode
        (neural_modules), so the tail is normalised exactly once.

        The acquired-CTI fraction (env.acquired_cti_fraction()) is the state's
        memory of how much of this episode's zero-day pool the agent has bought
        and delivered so far; it lets the value function attribute a purchase's
        delayed known-traffic payoff back to the epistemic action.
        """
        if self.proprio_feature_scaling:
            device, dtype = centroid.device, centroid.dtype
            proprio = torch.stack([
                torch.log1p(torch.tensor(float(num_anom), device=device, dtype=dtype)),
                torch.tensor(float(zda_confidence), device=device, dtype=dtype),
                torch.log1p(torch.tensor(float(num_known), device=device, dtype=dtype)),
                torch.tensor(float(cs_classif_confidence), device=device, dtype=dtype),
                torch.tensor(float(self.env.acquired_cti_fraction()), device=device, dtype=dtype),
                torch.tensor(float(self.env.epistemic_actions_available), device=device, dtype=dtype),
                self._proprio_budget_feature(curr_budget, device, dtype),
            ])
        else:
            proprio = torch.tensor([
                float(num_anom), float(zda_confidence),
                float(num_known), float(cs_classif_confidence),
                float(self.env.acquired_cti_fraction()),
                float(self.env.epistemic_actions_available), float(curr_budget)
            ], device=centroid.device, dtype=centroid.dtype)
        state_vec = torch.cat([
            centroid.squeeze(0),
            proprio
        ])
        # The exteroceptive block is a summary of the IM's hidden vectors; if a
        # loaded encoder is corrupt (non-finite weights from a diverged
        # pretraining run) this can be NaN/Inf, which propagates straight into
        # the replay buffer and NaNs the agent's value/critic loss. Sanitise
        # here, at the single point every state passes through, so the DM update
        # stays finite regardless of the IM's health.
        return torch.nan_to_num(state_vec, nan=0.0, posinf=0.0, neginf=0.0)
    
    def act(self, state_vec):
        """Gets action from the mitigation agent."""
        action = self.mitigation_agent.act(state_vec)
        return torch.Tensor([action]).long()

    def process_input(self, flows, node_feats: dict = None):
        """Main entry point for processing new network flows."""
        if len(flows) > 0:
            with self.profile("process_input_total"):
                if self.data_collection_mode and self.data_recorder is not None:
                    with self.profile("input_assembly"):
                        batch, row_flow_indices = self.stack_flow_tensors(flows, node_feats)
                    # batch is None when no flow produced a row this tick (only
                    # reachable with use_packet_feats=True and
                    # cache_flows_with_no_packets=False): nothing to record/train on.
                    if batch is None:
                        return
                    tick = self.wb_tracker.step_counter
                    self._record_flows_for_data_collection(batch, flows, row_flow_indices, tick)
                    if self.data_collection_skip_training:
                        if not self.agency:
                            self.wb_tracker.step_counter += 1
                        return
                    # data collection + training both run: still need encoded labels.
                    # (get_labels must NOT be called while holding self._lock: it can
                    # call add_class_to_knowledge_base, which is @thread_safe and would
                    # deadlock re-acquiring the same non-reentrant lock.)
                    batch.class_labels = self.get_labels(flows, row_flow_indices)
                else:
                    with self.profile("input_assembly"):
                        batch = self.assembly_input_tensor(flows, node_feats)
                    if batch is None:
                        return

                self._process_batch(batch)

        if self.agency and self.wb_tracker.step_counter % self.update_target_freq == 0:
            self.mitigation_agent.update_target_model()

    def _process_batch(self, batch):
        """
        Shared downstream path for a fully-assembled, fully-labelled batch:
        replay buffer push, online inference, single-batch training, step
        counter bookkeeping. Called identically by process_input (online,
        live flows) and process_input_from_record (offline replay of
        recorded shards), so an offline replay sees exactly what the
        original online run saw -- no re-derivation of this logic.
        """
        with self._lock:
            self.push_to_replay_buffers(batch.flow_features, batch.packet_features, batch.node_features, batch_labels=batch.class_labels)

            if self.batch_processing_allowed:
                with self.profile("online_inference_total"):
                    self.online_inference(batch)

            # we check again if batch_processing allowed because
            # knowledge can change during online inference.
            # K1: inference above runs every tick (live metrics); training below
            # is throttled to once every train_every_n_ticks eligible ticks, and
            # then runs train_steps_per_tick gradient steps. Defaults (1, 1)
            # reproduce the original single-step-every-tick behaviour.
            if self.batch_processing_allowed:
                self._ticks_since_train += 1
                if self._ticks_since_train >= self.train_every_n_ticks:
                    self._ticks_since_train = 0
                    for _ in range(self.train_steps_per_tick):
                        with self.profile("train_inf_module_single_batch"):
                            self.train_inf_module_single_batch()

            if not self.agency:
                # If there's no agency, the step increments here,
                # otherwise it increments with each action
                self.wb_tracker.step_counter += 1

    def process_input_from_record(self, flow_features, packet_features, node_features, element_classes, tick=None):
        """
        Offline-replay entry point: takes tensors/labels already recorded by
        FlowDataRecorder (one tick's worth of samples) and runs them through
        the exact same downstream path (_process_batch) that live process_input
        uses, so an offline run trains/evaluates identically to what would
        have happened online for this tick. Unlike process_input, callers
        are expected to have already grouped samples by recorded tick.
        """
        with self.profile("process_input_from_record_total"):
            if packet_features is not None and packet_features.dtype != torch.float32:
                # FlowDataRecorder stores packet_features as uint8 on disk (they're
                # integral byte values in [0,255]) to save space; classifiers expect
                # the float32 dtype the live online path always fed them.
                packet_features = packet_features.to(torch.float32)
            batch = Batch(flow_features=flow_features, packet_features=packet_features, node_features=node_features)
            batch.class_labels = self.get_labels_from_strings(list(element_classes))
            self.logger_instance.info(
                f"[TigerBrain] offline replay: tick={tick} n_samples={flow_features.shape[0]} "
                f"label_histogram={dict(Counter(element_classes))} "
                f"known_classes_count={self.current_known_classes_count} "
                f"batch_processing_allowed={self.batch_processing_allowed}"
            )
            self._process_batch(batch)


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

            test_zda_batch_labels = zda_batch_labels = torch.zeros(samples_per_class, 1, device=self.device)
            if class_nl_label in frozen_knowledge.get('G2s', set()):
                test_zda_batch_labels = zda_batch_labels = torch.ones(samples_per_class, 1, device=self.device)

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
            test_zda_labels = zda_labels = torch.zeros(samples_per_class, 1, device=self.device)

            if mode == TRAINING:
                if nl_label in self.env.current_knowledge['G2s']: continue
                if nl_label in self.env.current_knowledge['G1s']: zda_labels = torch.ones(samples_per_class, 1, device=self.device)

            if mode == INFERENCE:
                if nl_label in self.env.current_knowledge['G2s']:
                    test_zda_labels = zda_labels = torch.ones(samples_per_class, 1, device=self.device)
            
            try:
                f, p, n, l = replay_buff.sample(samples_per_class)
            except:
                continue

            # Label-noise channel (RC 3.4): a corrupted acquired class's training
            # labels are flipped to a wrong known class with its effective noise
            # probability, poisoning its prototype. Guarded by `enabled` so the
            # clean regime draws no RNG and stays byte-identical to before.
            if mode == TRAINING and self.env.cti_delivery.enabled:
                noise = self.env.cti_delivery.noise_for(nl_label)
                if noise > 0.0:
                    l = self._apply_cti_label_noise(l, noise, int(l.view(-1)[0].item()))

            all_flow.append(f)
            all_labels.append(l)
            all_zda.append(zda_labels)
            all_test_zda.append(test_zda_labels)
            if p is not None: all_packet.append(p)
            if n is not None: all_node.append(n)

        if not all_flow: return None
        return Batch(flow_features=torch.cat(all_flow, dim=0), packet_features=(torch.cat(all_packet, dim=0) if all_packet else None), node_features=(torch.cat(all_node, dim=0) if all_node else None), class_labels=torch.cat(all_labels, dim=0), zda_labels=torch.cat(all_zda, dim=0), test_zda_labels=torch.cat(all_test_zda, dim=0))

    def _apply_cti_label_noise(self, labels, noise, true_code):
        """
        Flip a `noise` fraction of a corrupted CTI class's training-label draws
        to a wrong known class (RC 3.4 label-noise channel). 'symmetric' picks a
        uniformly-random other currently-known class; 'nearest' targets the class
        this one is most confused with in the current training confusion matrix
        (a more plausible, adversarial-leaning corruption), falling back to a
        random other class when no confusion signal exists yet. Returns the
        (possibly cloned) label tensor; a no-op when no row is selected to flip.
        """
        n = labels.shape[0]
        if n == 0:
            return labels
        flip_mask = torch.rand(n, device=labels.device) < noise
        k = int(flip_mask.sum().item())
        if k == 0:
            return labels
        candidates = [c for c in self.encoder.get_codes_for_labels(self.env.current_knowledge['Knowns'])
                      if c != true_code and c < self.current_known_classes_count]
        if not candidates:
            return labels
        labels = labels.clone()
        if self.env.cti_delivery.label_noise_mode == 'nearest':
            wrong = self._most_confused_known_class(true_code, candidates)
            wrong_codes = torch.full((k,), wrong, device=labels.device, dtype=labels.dtype)
        else:
            cand_t = torch.tensor(candidates, device=labels.device, dtype=labels.dtype)
            wrong_codes = cand_t[torch.randint(0, len(candidates), (k,), device=labels.device)]
        flat = labels.view(-1)
        flat[flip_mask] = wrong_codes
        return labels

    def _most_confused_known_class(self, true_code, candidates):
        """
        The candidate known class most often confused with `true_code` in the
        current training confusion matrix (training_cs_cm, rows = true labels,
        cols = predictions); the first candidate if the matrix carries no signal
        for this class yet.
        """
        cm = getattr(self, 'training_cs_cm', None)
        if cm is not None and true_code < cm.shape[0]:
            row = cm[true_code]
            best, best_val = None, 0.0
            for c in candidates:
                if c < row.shape[0] and row[c].item() > best_val:
                    best_val, best = row[c].item(), c
            if best is not None:
                return best
        return candidates[0]

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

            batch_os_cm = efficient_os_cm(preds=(zda_predictions[accuracy_mask].detach() > self.ad_threshold).long(), targets_onehot=onehot_zda_labels[accuracy_mask].long())

            cummulative_os_cm = (self.training_os_cm if mode == TRAINING else self.eval_os_cm)
            cummulative_os_cm += batch_os_cm
            zda_balance = zda_labels[accuracy_mask].to(torch.float32).mean().item()
            neg_w = 1 - zda_balance
            cummulative_os_acc = get_balanced_accuracy(cummulative_os_cm, negative_weight=neg_w)

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
            # See ad_loss_backprop_to_encoder docstring in __init__: when
            # False, feed the decoder detached copies of the encoder's
            # outputs so the AD/BCE loss below can still train the decoder's
            # own parameters (if any) but cannot backprop into the encoder
            # via `query_h`/`scores`, turning RMD back into the frozen-
            # feature diagnostic Ren et al. (2021) describe.
            cd_hiddens = hiddens if self.ad_loss_backprop_to_encoder else hiddens.detach()
            cd_scores = logits[:, known_h_mask] if self.ad_loss_backprop_to_encoder else logits[:, known_h_mask].detach()
            zda_preds = self._call_confidence_decoder(
                self.confidence_decoder,
                scores=cd_scores,
                hidden_vectors=cd_hiddens,
                labels=training_batch.class_labels,
                query_mask=query_mask,
                known_class_mask=known_h_mask)
            if self.multi_class:
                zda_loss, _, ad_metrics = self.evaluate_anomaly_detection(training_batch.zda_labels[query_mask], zda_preds, torch.ones(query_mask.sum(), device=self.device).to(torch.bool), TRAINING)
                loss += zda_loss

        kr_loss, pred_clusters, kr_metrics = self.evaluate_kernel_regression(pred_kernel, one_hot_labels, TRAINING)
        if self.clustering_loss_backprop: loss += kr_loss
        
        classif_loss, cs_acc, cs_metrics = self.evaluate_closed_set(training_batch.class_labels[query_mask], logits, TRAINING)
        loss += classif_loss

        self.training_cs_cm += efficient_cm(preds=logits.detach(), targets_onehot=one_hot_labels[query_mask])

        self.optimizer.zero_grad()
        loss.backward()
        # Clip the global grad-norm before stepping. Without this the
        # Mahalanobis AD loss can amplify the encoder's gradient by 1/within_var
        # (up to 1e4, unbounded as the representation collapses) and diverge the
        # encoder to inf/NaN over offline_replay's repeated passes -- which then
        # NaNs the DM's value loss when those weights are loaded for agency.
        if self.grad_clip_max_norm and self.grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.classifier.parameters()) + list(self.confidence_decoder.parameters()),
                self.grad_clip_max_norm)
        self.optimizer.step()

        

        # Plot the training confusion matrices BEFORE the report block resets
        # them. plot_step_freq (default 150) is a multiple of report_step_freq
        # (default 5), so every plotting step is also a reporting step; running
        # the report block first (which calls reset_train_cms) would zero
        # training_cs_cm / training_os_cm before reporter.report reads them,
        # rendering an all-zero confusion matrix in wandb. Plotting first shows
        # the matrices accumulated over the current report window, then the
        # report block windows the scalar metrics and resets, as before.
        if self.wb_tracker.step_counter % self.plot_step_freq == 0:
            plots = self.reporter.report(logits[:,known_h_mask], hiddens.detach(), training_batch.class_labels, pred_clusters, query_mask, TRAINING, training_cs_cm=self.training_cs_cm, training_os_cm=self.training_os_cm)
            if self.wbt: self.wb_run.log(plots, step=self.wb_tracker.step_counter)

        if self.wb_tracker.step_counter % self.report_step_freq == 0:
            all_metrics = {}
            all_metrics.update(ad_metrics)
            all_metrics.update(kr_metrics)
            all_metrics.update(cs_metrics)
            self.reset_train_cms()
            self.reporter.log_scalars(all_metrics, step=self.wb_tracker.step_counter)
        
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
    def perform_epistemic_action(self, target_label=None, purity=None, confidence=None):
        """Acquires a CTI label, updating the knowledge base and replay buffers.

        Delegates to the environment. Under the CTI delivery model (RC 3.4) the
        purchase is charged now but its delivery may be degraded, partially
        captured, and/or delayed; `purity`/`confidence` are the cluster-quality
        signals corruption severity is coupled to. When delivery is instant (the
        clean regime, or a zero delay) `updated_label` is set and the class is
        registered here; when it is deferred, `updated_label` is None and the
        registration happens later via _poll_cti_deliveries once the delay
        elapses.
        """
        updates = self.env.perform_epistemic_action(
            target_label, current_step=self.wb_tracker.step_counter,
            purity=purity, confidence=confidence)
        new_label = updates['updated_label']
        if new_label is not None:
            self._register_delivered_cti(new_label)
        return updates

    def _register_delivered_cti(self, new_label):
        """
        Register a just-delivered CTI class with the label encoder, open its
        replay buffer if it is genuinely new, and stamp its acquisition-latency
        clock. Shared by instant delivery (perform_epistemic_action) and delayed
        delivery (_poll_cti_deliveries), so both paths register a class the same
        way.
        """
        if self.encoder.update_label(new_label=new_label, logger=self.logger_instance):
            self.current_known_classes_count += 1
            self.add_replay_buffer(new_label)
            self.reset_train_cms()
            self.reset_test_cms()
        # Stamp the delivery step so online_inference can later emit
        # epistemic_delay/<label> = (steps until this class next shows up among
        # an online batch's true labels, i.e. once it can actually be trained
        # on). For instant delivery this is the buying step; for delayed
        # delivery it is the step the CTI actually arrives.
        self._pending_epistemic_delays[new_label] = self.wb_tracker.step_counter

    def _poll_cti_deliveries(self):
        """
        Deliver any purchased CTI whose (delayed) arrival is now due, registering
        each newly-trainable class and logging the corruption it arrived with.
        Called once per online tick; a no-op (nothing pending) in the
        instant-delivery regime.
        """
        if not self.env.cti_delivery.has_pending():
            return
        for label in self.env.deliver_pending_cti(self.wb_tracker.step_counter):
            self._register_delivered_cti(label)
            if self.wbt:
                self.reporter.log_scalars({
                    f'cti_effective_noise/{label}': self.env.cti_delivery.noise_for(label),
                    f'cti_capture/{label}': self.env.cti_delivery.capture_for(label),
                }, step=self.wb_tracker.step_counter)

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
                        zda_p = self._call_confidence_decoder(
                            decoder_clone,
                            scores=logits[:, k_mask],
                            hidden_vectors=hiddens,
                            labels=eval_batch.class_labels,
                            query_mask=q_mask,
                            known_class_mask=k_mask)
                        zda_l = eval_batch.zda_labels[q_mask]
                        oh_zda = torch.zeros(size=(zda_l.shape[0], 2), device=self.device).long().scatter(1, zda_l.long().view(-1, 1), 1)
                        b_os_cm = efficient_os_cm(preds=(zda_p > self.ad_threshold).long(), targets_onehot=oh_zda)
                        l_os_cm += b_os_cm
                        pos_w = zda_l.to(torch.float32).mean().item()
                        neg_w = 1 - pos_w
                        ad_acc = get_balanced_accuracy(b_os_cm, negative_weight=neg_w).item()
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
        wandb_run_name = self.kwargs['wandb']['wb_run_name']
        if curr_cs > self.best_cs_accuracy:
            self.best_cs_accuracy = curr_cs
            self.save_model(
                self.classifier,
                self.classifier_path[:-3] + 'single_'+wandb_run_name+'.pt',
                "flow classifier")
        if curr_ad > self.best_AD_accuracy:
            self.best_AD_accuracy = curr_ad
            self.save_model(
                self.confidence_decoder, 
                self.confidence_decoder_path[:-3] + 'single_'+wandb_run_name+'.pt',
                "confidence decoder")
        if curr_kr > self.best_KR_accuracy:
            self.best_KR_accuracy = curr_kr
            self.save_model(
                self.classifier,
                self.classifier_path[:-3] + 'coupled_'+wandb_run_name+'.pt',
                "flow classifier (coupled)")
            if self.multi_class: 
                self.save_model(
                    self.confidence_decoder,
                    self.confidence_decoder_path[:-3] + 'coupled'+wandb_run_name+'.pt', 
                    "confidence decoder (coupled)")

    def save_model(self, model, path, name):
        """Saves a model's state dictionary to a file."""
        state_dict = model.state_dict()
        # Never persist a diverged checkpoint. A NaN/Inf in the classifier or
        # confidence-decoder weights (e.g. after the Mahalanobis AD loss ran the
        # encoder off to infinity) would otherwise be written to disk and later
        # loaded for an agency run, where it NaNs the DM's value loss. Refuse to
        # overwrite the last-good checkpoint with a corrupt one.
        bad = [k for k, v in state_dict.items()
               if torch.is_tensor(v) and not torch.isfinite(v).all()]
        if bad:
            self.logger_instance.error(
                f'\033[91mRefusing to save {name}: non-finite (NaN/Inf) weights in '
                f'{bad} -- the model has diverged. Keeping the previous checkpoint at '
                f'{path}. Lower the learning rate or intrusion_detection.grad_clip_max_norm.\033[0m')
            return
        torch.save(state_dict, path)
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

    def get_labels_from_strings(self, labels):
        """
        Encodes raw natural-language string labels into integers, mutating the
        dynamic label encoder / knowledge base as new classes are discovered.
        Core logic shared by the online path (get_labels) and offline replay
        (process_input_from_record), which has only recorded strings, not
        live Flow objects.
        """
        for cl in self.encoder.fit(labels): self.add_class_to_knowledge_base(cl)
        return self.encoder.transform(labels).to(device=self.device, dtype=torch.long)

    def get_labels(self, flows, row_flow_indices=None):
        """
        Encodes string labels from flows into integers. If row_flow_indices is
        given (one entry per output batch row, see stack_flow_tensors), the
        per-flow label is repeated once for every row that flow contributed.
        """
        if row_flow_indices is None:
            row_flow_indices = range(len(flows))
        return self.get_labels_from_strings([flows[i].element_class for i in row_flow_indices])

    def stack_flow_tensors(self, flows, node_feats):
        """
        Stacks the per-flow feature tensors into a batch, with no side effects
        on the dynamic label encoder (unlike assembly_input_tensor). Used both
        by the normal training path (assembly_input_tensor wraps this and adds
        class_labels) and by data collection mode, which records raw tensors
        and string labels without mutating encoder/replay-buffer state.

        Flow-stats only refresh every flowstats_freq_secs, but packets can be
        captured several at a time during a sampling burst. Rather than using
        only the single most-recently-seen packet per flow (and silently
        dropping the rest of the burst), every queued packet for a flow is
        turned into its own row, all sharing that flow's current (still valid)
        flow_feat window. A flow with no freshly-queued packets this tick still
        contributes its one sticky last-known-packet row, so it isn't starved --
        unless cache_flows_with_no_packets is False (K2), in which case such a
        flow is skipped for this tick instead of re-emitting a stale row.

        Returns (batch, row_flow_indices) where row_flow_indices[i] is the
        index into `flows` that produced batch row i -- needed downstream to
        repeat labels/flow_ids correctly since flows no longer map 1:1 to rows.
        Returns (None, []) when no flow produced any row this tick (only
        reachable with use_packet_feats=True and cache_flows_with_no_packets=False).
        """
        flow_feat_rows = []
        packet_feat_rows = [] if self.use_packet_feats else None
        node_feat_rows = [] if self.use_node_feats else None
        row_flow_indices = []

        for flow_idx, flow in enumerate(flows):
            flow_feat_window = flow.get_flow_features()
            node_feat_vec = get_metrics_tensor(node_feats, flow.dest_ip, self.kwargs['health']) if self.use_node_feats else None

            packet_chunks = []
            if self.use_packet_feats:
                packet_chunks = flow.drain_packet_feature_chunks(
                    self.packets_per_sample, max_chunks=self.max_packet_samples_per_flow_per_tick)
                if packet_chunks:
                    n_pending = len(flow.pending_packet_feats)
                    backlog_note = (
                        f"; {n_pending} packet(s) still backlogged ")
                    self.logger_instance.debug(
                        f"[TigerBrain] flow {flow.flow_id}: consumed "
                        f"({len(packet_chunks) * self.packets_per_sample} packet(s) total, "
                        f"{backlog_note}")
                    flow.packet_count += len(packet_chunks) * self.packets_per_sample
                elif self.cache_flows_with_no_packets:
                    # Nothing freshly captured this tick: fall back to the
                    # sticky last-known packet so the flow still contributes.
                    self.logger_instance.warning(
                        f"[TigerBrain] flow {flow.flow_id}: no packets "
                        f"captured this tick; using sticky last packet"
                    )
                    packet_chunks = [flow.get_packet_features()]
                else:
                    # K2 (cache_flows_with_no_packets=False): skip this flow for
                    # this tick rather than re-emitting its stale sticky packet(s).
                    self.logger_instance.debug(
                        f"[TigerBrain] flow {flow.flow_id}: no packets captured "
                        f"this tick; skipping (cache_flows_with_no_packets=False)"
                    )
                    continue

            n_rows_for_flow = len(packet_chunks) if self.use_packet_feats else 1
            for row_i in range(n_rows_for_flow):
                flow_feat_rows.append(flow_feat_window)
                if self.use_packet_feats: packet_feat_rows.append(packet_chunks[row_i])
                if self.use_node_feats: node_feat_rows.append(node_feat_vec)
                row_flow_indices.append(flow_idx)

        if not flow_feat_rows:
            # No flow produced a row this tick (only reachable with
            # use_packet_feats=True and cache_flows_with_no_packets=False, when
            # every flow's pending queue happened to be empty). Signal "nothing
            # to do" so the caller skips this tick cleanly.
            return None, []

        f_batch = torch.stack(flow_feat_rows)
        # Packet rows are uint8 (raw bytes) from build_packet_tensor all the way
        # through the per-flow window/queue; cast to float32 once here, per
        # batch, instead of once per packet on the PacketIn thread. .to() is
        # device-preserving, so this stays correct for a GPU deployment where
        # the rows may already live on the accelerator.
        p_batch = torch.stack(packet_feat_rows).to(torch.float32) if self.use_packet_feats else None
        n_batch = torch.stack(node_feat_rows) if self.use_node_feats else None
        return Batch(flow_features=f_batch, packet_features=p_batch, node_features=n_batch), row_flow_indices

    def assembly_input_tensor(self, flows, node_feats):
        """Assemblies a batch from current flow observations. Returns None when
        stack_flow_tensors produced no rows this tick (see its docstring)."""
        batch, row_flow_indices = self.stack_flow_tensors(flows, node_feats)
        if batch is None:
            return None
        batch.class_labels = self.get_labels(flows, row_flow_indices)
        return batch

    def _record_flows_for_data_collection(self, batch, flows, row_flow_indices, tick):
        """
        Hands a freshly-assembled batch (tensors only, no encoder side effects)
        plus the raw ground-truth labels off to the FlowDataRecorder. Never
        records flow.zda/flow.test_zda: those are derived from this run's
        curriculum state (current_knowledge['G1s'/'G2s']), which mutates as
        the agent buys CTI -- see NewTigerEnvironment.perform_epistemic_action.
        Only element_class (a fact about the traffic, not the curriculum) is
        persisted; zda/test_zda must be recomputed at replay time.

        row_flow_indices maps each batch row back to its source flow (see
        stack_flow_tensors): a flow that contributed multiple packet-fanned
        rows this tick has its element_class/flow_id repeated accordingly.
        """
        element_classes = [flows[i].element_class for i in row_flow_indices]
        flow_ids = [flows[i].flow_id for i in row_flow_indices]
        try:
            self.data_recorder.record(batch, element_classes, tick=tick, flow_ids=flow_ids)
        except Exception:
            self.logger_instance.error(
                f"[TigerBrain] data recorder failed on tick={tick}: {traceback.format_exc()}"
            )
