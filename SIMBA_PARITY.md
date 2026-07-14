# TIGER ↔ SIMBA parity map

SIMBA (`simba/`) is the simplified brain that produced the good offline
results on the pre-recorded real traces (`simba/REAL_DATA_CALIBRATION.md`).
This documents how the full TIGER stack is configured to reproduce SIMBA's
calibrated regime in the live GNS3 simulation, what code was added to make
that possible, and which differences deliberately remain. The activating
config lives in `tiger/config/overrides/dista_tiger.yaml` (every value there
is annotated with the `SimbaConfig` field it mirrors).

## Parameter map (SimbaConfig → TIGER intrusion_detection key)

| SIMBA field (value) | TIGER key | note |
|---|---|---|
| `im_learning_rate` (1e-4) | `im_learning_rate` | **new key** — IM optimizer decoupled from the DM's; the real-trace stability fix |
| `dm_learning_rate` (5e-4) | `dm_learning_rate` | **new key** — remapped onto the agent's optimizer lr |
| `im_support_size` (8) | `k_shot: 8` | support samples per class |
| `im_query_size` (8) | `batch_size: 16` | 16/class = 8 support + 8 query |
| `packets_per_sample` (1) | `packets_per_sample: 1` | |
| `buffer_capacity` (512) | `replay_buffer_max_capacity: 512` | per-class IM buffer |
| `im_train_steps_per_tick` (1) | `train_steps_per_tick: 1` | |
| `buy_train_burst` (10) | `buy_train_burst: 10` | **new key** — IM burst on CTI delivery |
| `gamma` (0.998) | `agent_discount_rate: 0.998` | |
| `dm_hidden` (128) | `dm_hidden: 128` | **new key** — DM trunk width decoupled from IM `hidden_size` |
| `replay_capacity` (100k) | `agent_memory_size: 100000` | |
| `replay_batch_size` (64) | `replay_batch_size: 64` | |
| `target_update_freq` (500, hard) | `use_soft_update: false`, `update_target_freq: 500` | |
| `grad_clip` (10) | `dm_grad_clip_max_norm: 10.0` | already existed |
| `eps_start`/`eps_end` (1.0/0.05) | `init_epsilon_egreedy`/`greedy_min` | |
| `eps_decay_steps` (30k, linear) | `eps_linear_decay_steps: 30000` | **new key** — linear per-decision decay replaces per-replay multiplicative |
| `explore_buy_weight` (0.1) | `explore_buy_weight: 0.1` | **new key** — buy-averse exploration |
| `reward_scale` (0.02) | `dm_reward_scale: 0.02` | **new key** — TD-side reward scaling |
| `learn_start` (1000) | `dm_learn_start: 1000` | **new key** |
| replay per decision | `dm_replay_per_decision: true` | **new key** — SIMBA replays after every decision |
| no PER / 1-step TD / no Boltzmann | `use_per: false`, `n_step_rewards: 1`, `boltzmann_sampling: false` | |
| `init_budget` (2500) | `tiger_init_budget: 2500` | |
| `min_budget` (0) | `min_budget: 0` | |
| `bankrupt_terminates` (false) | `disable_budget_bankrupt_termination: true` | |
| `unknown_accept_discount` α (0.25) | `unknown_accept_reward_scale: 0.25` | |
| `block_benign_scale` β (1.0) | `block_benign_penalty_scale: 1.0` | **new key** — legacy TIGER scored every block as 0 |
| `waste_buy_penalty` (25) | `useless_epistemic_penalty: 25` | Knowns/G1s are priced 0, so a wasted buy costs ~25 |
| buy is non-blocking | `epistemic_is_blocking: false` | |
| `prices` / `rewards` / curriculum | `prices:` / `rewards:` / `knowledge:` blocks | already carried SIMBA's calibrated values |
| CE prototypical loss | `use_huber_cs: false` | |
| AD: dist > tau (G1-calibrated) | `inference_model_variant: simba_ad`, `ad_threshold: 0.5` | **new variant** `im_models/simba_ad.py` |
| DM state: stationary input-space centroid | `state: input` | **new mode** — SIMBA's `input_rep` (latest flow row + window mean + mean packet bytes, log1p / ÷255, running per-feature standardisation) |
| prototypes exclude G1 at inference | `exclude_g1_from_ad_known_set: true` | |
| no impurity shaping / no AD oracle / static prices | `cluster_impurity_penalty_weight: 0`, `hard_epistemic_action: false`, `price_decay: false` | |

## Code added for parity (all default to exact legacy no-ops)

* `tiger_brain_new.py` — `im_learning_rate` (IM optimizer), `dm_learning_rate`
  remap in `init_agents`, `buy_train_burst` in `_register_delivered_cti`,
  per-decision DM replay in `online_inference`, blocked-benign penalty in
  `_decision_reward`.
* `tiger_agents.py` (`ValueLearningAgent`) — linear per-decision epsilon
  schedule, buy-averse exploration draw, TD reward scaling, learn-start guard.
* `neural_modules.py` — `dm_hidden` decouples the DQN/DuelingDQN trunk width
  from the IM's `hidden_size`.
* `im_models/simba_ad.py` — SIMBA's anomaly rule as a model variant: anomaly
  probability `sigmoid((d_nearest_prototype − τ)/scale)` with τ learned by the
  G1-supervised AD/BCE loss (`p > 0.5 ⇔ d > τ`, SIMBA's `dist > tau`).
  Loads the existing h200 pretrained classifier checkpoint unchanged, and
  accepts the (empty) parameter-free decoder checkpoints non-strictly.

## Deliberate residual differences

* `neural_modules.hidden_size` stays **200** (SIMBA: 64) *until a
  SIMBA-geometry IM is pretrained*: the shipped pretrained IM checkpoints are
  h200 and cold-starting the IM online is the instability SIMBA's calibration
  warns about. The drift problem is addressed by `im_learning_rate: 1e-4`
  instead. To close the gap, pretrain with
  `tiger/config/overrides/tiger_pretraining.yaml` (now the SIMBA-geometry
  recipe: h64, single GRU layer, no dropout, `use_encoder: false`, `simba_ad`),
  rename the saved best checkpoints to
  `multiclass_flow_packet_classifier_pretrained_h64.pt` /
  `flow_packet_confidence_decoder_pretrained_h64.pt` in `tiger_models/`, and
  uncomment the "SIMBA-geometry IM" block in `dista_tiger.yaml`.
* Unknown-traffic **clustering** stays on the G1-supervised kernel-regression
  head (SIMBA clusters in the standardised raw-input space). Both derive their
  granularity from the G1 pseudo zero-days; the substrate differs.
* ~~The DM's exteroceptive state is the hidden-space centroid~~ **closed**:
  `state: input` now feeds the DM SIMBA's stationary input-space centroid.
  TIGER's proprio tail already carries the per-class acquired-CTI map, the
  analogue of SIMBA's per-class known-flags.
* Episode length is `max_episode_steps: 1000` DM decisions ≈ SIMBA's 51-tick
  real-trace episode at ~20 decisions/tick.

**The full remaining-difference audit lives in `tiger_vs_simba.md`** —
subsystem by subsystem, with impact ratings and the lever that would close
each gap.

## Running the offline experiment grid live

`tiger/tests/simba_parity_ablations.py` replays `simba_experiments.py`'s grid
(ablation modes × DM agents × seeds) on the live GNS3 stack via `dash_cli.py`,
under the `dista_tiger` profile: modes `drl` / `no_epistemic` / `greedy_cti` /
`fixed_threshold_cti` (one run per threshold in the sweep) / `oracle`
(greedy + `hard_g2s` blocklist of the malicious G2s), agents `DQN` / `DDQN` /
`DuelingDDQN`, default seeds 6 and 1.
