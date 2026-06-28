# TIGER's DRL — Detailed Analysis

Grounded entirely in `smartville-controller` (branch `TIGER_PAPER_DO_NOT_DELETE`, files `tiger_environment_new.py`, `tiger_brain_new.py`, `tiger_agents.py`, `neural_modules.py`) and the hyperparameters actually shipped in `tiger/config/default.yaml`. 

## 1. The two levels 
Two *coupled but asynchronously-updated* learning problems sharing one stream of network traffic:

- **Upper level — the Inference Module (IM)**: a supervised/self-supervised learner trained every tick by `train_inf_module_single_batch` via ordinary backprop on three losses (closed-set CE/Huber, anomaly-detection BCE, kernel-regression). It does **not** observe reward or budget. Its "opponent" is the traffic distribution and curriculum, not the DM.
- **Lower level — the Decision Module (DM)**: an RL agent that consumes the IM's *current* outputs (logits, hidden centroids, anomaly probability, confidence) as part of its state, and whose actions feed back into the IM's training distribution: action `2` (buy CTI) calls `perform_epistemic_action`, which moves a class from `G2s` to `Knowns`, mutates `DynamicLabelEncoder`, and allocates a brand-new replay buffer — i.e., the DM's epistemic action literally changes the IM's training-set curriculum and output dimensionality (`self.current_known_classes_count += 1`).

So the coupling is: **IM parameters → DM's state/reward each tick → DM's epistemic actions → IM's curriculum/training distribution for subsequent ticks**, with the IM frozen (`.eval()`) during the action-selection moment and only trained afterward, off a replay-buffer sample that is *decoupled* from the exact batch the DM just acted on (`train_inf_module_single_batch` samples fresh from buffers, not from `online_batch`). This is the literal mechanism of the "game": one player (IM) optimizes classification/clustering loss by gradient descent; the other (DM) optimizes long-run budget by RL, and the DM's exploration of action `2` is the only channel through which it can change the curriculum the IM is asked to learn.

## 2. The DM's MDP, precisely as coded

### 2.1 Action space

Always size 3:
- `0` = accept (pragmatic)
- `1` = block (pragmatic)
- `2` = buy CTI (epistemic)

Action selection is delegated per-agent-type to `self.mitigation_agent.act(state)`, wrapped by `TigerBrain.act`. For `ValueLearningAgent` (DQN/DDQN/Dueling variants): epsilon-greedy (`init_epsilon_egreedy=1.0`, decays by `greedy_decay=0.999` to `greedy_min=0.01`) **or** Boltzmann sampling over Q-values if `boltzmann_sampling: true` (default config has it `true`) — `tiger_agents.py`. For the DAI_* agents: categorical sampling either from the policy net or from a softmax over negative-EFE values, depending on `use_critic_to_act` (default `true`). For PPO/A2C: categorical sampling from the actor's softmax output.

There are three **forced-action overrides** that bypass the agent's own choice, applied per "unknown cluster" before `act()`'s output is used. 
1. `cti_period != -1` → a *periodic* CTI policy: every `cti_period` steps, action is hard-forced to `2`; otherwise the agent is queried but its `2` is remapped to `1` (block).
2. `greedy_cti=True` → forces `2` whenever `self.env.epistemic_actions_available == 1` (i.e., whenever an unbought G2 class still exists); otherwise queries the agent and remaps `2→1`.
3. `no_epistemic_actions=True` (and neither of the above set) → queries the agent normally but remaps any `2` to `1`, effectively disabling epistemic actions entirely (ablation knob).

These three are mutually exclusive ablation modes, not part of the "default" DM behavior — confirmed against `tiger/config/default.yaml`, where `greedy_cti: False`, `cti_period: -1`, and `no_epistemic_actions: false` are all off, so the actual learned-agent path is the final fallthrough in `_select_unknown_cluster_action`, which returns the agent's own action unmodified unless `epistemic_actions_available == 0`, in which case a `2` is remapped to `1` (block) since there is no G2 class left to buy.

For known-class traffic, the action space is collapsed to a binary choice in effect: if `automatic_cs_acceptance=True` (a config flag, default `false`), action is hard-coded to `0`; otherwise the agent is queried, but **only actions `0` (accept the classifier's verdict) or "anything else" (reject/no-confidence) are semantically distinguished** in the reward logic — see §2.4. So even though `act()` can return `0/1/2` here too, the code only branches on `action_signal.item() == 0` vs. not; there's no real CTI semantics for known traffic, just "trust the IM" vs. "discard its classification and penalize."


### 2.2 State space

`self.state_space_dim = hidden_size [+ hidden_size if use_node_feats] [+ hidden_size if use_packet_feats] + 6`. The trailing 6-dim block (called "proprioceptive" in the comments) is, in order, assembled by `assembly_state_vector(centroid, num_anom, num_known, zda_confidence, cs_classif_confidence, curr_budget)`:

```
[ centroid (hidden_size dims) ,
  num_anom, zda_confidence, num_known, cs_classif_confidence,
  epistemic_actions_available (0/1), current_budget ]
```

`zda_confidence` and `cs_classif_confidence` are passed into `assembly_state_vector` explicitly by the caller, so each call site supplies the confidence of the specific group/cluster the decision is actually about.

Both call sites take **one DM decision per identified group**, looped, with a real exteroceptive centroid every time:

- **Known-traffic step** (`act_on_known_traffic`): the predicted-known online samples of the tick are grouped by the IM's predicted class (`torch.unique(class_preds)`), one DM decision *per predicted-class group*. The centroid is the mean hidden vector of that group's members; `cs_classif_confidence` is that group's own classification confidence, computed by `_cs_confidence_for_slice` over just that group's logits/predictions (four `confidence_strategy` strategies — baseline/entropy/energy/margin — scoped to the group). The `zda_confidence` slot in this state is always `0.0`: these samples were never evaluated for anomalousness, so there is no meaningful anomaly-confidence to report, and the zero doubles as a regime indicator the agent can read.
- **Unknown-cluster step** (`act_on_unknown_clusters`): one DM decision *per identified cluster*, looped (`for idx, centroid in enumerate(centroids[~missing])`); `zda_confidence` is that cluster's own confidence, computed by `_zda_confidence_for_subset` over just the anomaly probabilities of that cluster's members (cluster membership is read off `clusters_oh`'s column for the cluster, matched against the online samples' anomaly probabilities in the same order). The `cs_classif_confidence` slot in this state is always `0.0`: these samples were never classified into a known class, so there is no meaningful classification-confidence to report, and the zero doubles as a regime indicator.

So the "state" the agent sees is a same-shaped vector that is always "here's a specific group/cluster's centroid + scalars", whether the tick is processing known or unknown traffic, and whether it's group 1, 2, or N within that tick. Slots 2 and 4 (`zda_confidence`, `cs_classif_confidence`) are mutually exclusive: exactly one of them is non-zero in any given state, telling the agent which regime — known-traffic class-inference or unknown-cluster — the rest of the vector describes. The exteroceptive part only falls back to a `-1`-filled placeholder for the very last group/cluster of a tick's loop, as the "next state" when there is no next group/cluster to chain to (see §2.3).

### 2.3 Transitions

There is no explicit `P(s'|s,a)` model — transitions are produced by direct simulation inside the loops, not sampled from a stored distribution:

- **Known-traffic transition**: `new_state[:-6] = ` the *next* predicted-class group's centroid if there's a next group in this tick's batch, else `-1*ones(...)`; `new_state[-1] = self.env.current_budget` (post-reward). So within one inference tick, the sequence of per-group decisions is chained as a genuine sequential MDP — group *i*'s "next state" is literally group *i+1*'s class-inference centroid — mirroring the unknown-cluster transition below. `num_anom`/`num_known` are **not** updated between groups in the same tick (frozen at the tick's batch-level values); `zda_confidence` stays `0.0` throughout (regime marker); `cs_classif_confidence` varies group-to-group, recomputed fresh for each group before `assembly_state_vector` is called.
- **Unknown-cluster transition**: `next_state[:-6] = centroids[~missing][idx+1]` if there's a next cluster in this tick's batch, else `-1*ones(...)`; `next_state[-2] = epistemic_actions_available` (post-purchase, can flip 0→1... actually 1→0 if last G2 was just bought); `next_state[-1] = current_budget` (post-reward). So within one inference tick, the sequence of per-cluster decisions is chained as a genuine sequential MDP — cluster *i*'s "next state" is literally cluster *i+1*'s centroid. `num_anom`/`num_known` are **not** updated between clusters in the same tick (frozen at the tick's batch-level values); `cs_classif_confidence` stays `0.0` throughout (regime marker); `zda_confidence` varies cluster-to-cluster, each cluster's `state_vec` assembled with its own freshly-computed cluster confidence.
- **Budget update is the core state-transition dynamic**: `self.env.current_budget += current_reward` (or `+= classification_reward`), accumulated *within* a tick across all per-cluster/per-group decisions before the episode-end check is even made.

The "transition" of the environment as a whole, across ticks, is driven by an exogenous, scripted process: real network traffic arriving from `process_input`/`process_input_from_record`, not by any agent-conditioned generative model of future flows. The DM's actions affect future *reward* (via budget and curriculum) but not which traffic arrives next — traffic arrival is action-independent. This is an important asymmetry vs. a textbook MDP: the "exogenous" part of the state (which classes/flows appear) is unaffected by the policy; only the "endogenous" part (budget, knowledge, replay buffers) is.

### 2.4 Reward

Reward is one rule, shared by every DM decision (known-traffic group or
unknown-traffic cluster alike) — `_decision_reward(accepted, group_rewards)`
in `tiger_brain_new.py`:

- **Accepted** (`action==0`, or `action==2` with `epistemic_is_blocking=False`):
  reward = `sum(group_rewards)`, the group's true per-flow rewards, signed
  positive for benign content and negative for malicious content. Trusting
  a benign verdict/cluster earns its reward; trusting a malicious one costs
  its reward — no separate correct/incorrect bookkeeping is needed, since
  the sign of the true reward already encodes that.
- **Blocked** (`action==1`, or any non-zero known-traffic action signal, or
  `action==2` with `epistemic_is_blocking=True`): reward =
  `-relu(group_rewards).sum()`, i.e. only the benign reward the group would
  have earned is forgone; there's no cost for the malicious content, since
  it's blocked.
- **Epistemic** (`action==2`): the accept/block reward above, minus
  `price_payed` (the actual CTI cost, `|class_reward * current_cti_price_factor|`,
  from `perform_epistemic_action`). Action `2` is only ever offered to the
  agent when `epistemic_actions_available == 1` (a real G2 class exists to
  buy) — `_select_unknown_cluster_action` remaps any `2` it returns to `1`
  (block) otherwise, so there is no separate "useless epistemic action"
  penalty: the choice simply isn't offered when it would be a no-op.
- Per-sample rewards come from `self.env.flow_rewards_dict`, set straight
  from the `rewards:` YAML block — fixed scalars per traffic class (e.g.,
  `mirai: -0.20`, `hue: 0.05` in the smaller config, or larger magnitudes in
  the dista override) — reward shaping is entirely a human-authored lookup
  table, not learned or derived from any cost model. This lookup table is
  the *only* tunable for reward magnitude; there is no separate penalty
  factor, "hard"/"easy" mode, or per-branch multiplier layered on top.
- For known-traffic groups, this is computed per predicted-class group
  (a single tick can credit several known-traffic rewards, one per distinct
  class the IM predicted that tick, W&B logs their per-group average). For
  unknown clusters, it's computed per identified cluster. In both cases the
  reward is added to `self.env.current_budget`.

**CTI pricing dynamics**: `current_cti_price_factor *= clamp(N(0.7, 0.4), 0.01, 0.99)` is applied stochastically *every* decision step if `price_decay=True` (`price_decay()`, `tiger_environment_new.py`, called at the top of both `act_on_known_traffic` and inside the unknown-cluster loop) — this is a strictly *decaying* multiplicative random walk (the sampled factor is clamped into `[0.01, 0.99]`, so price can only shrink over time, never recover), bounded below at `1%` of its previous value per single multiplicative step, not in absolute units. In `dista_tiger.yaml` (paper-relevant override) `price_decay` is explicitly `true`. In `default.yaml` it's `false`.

### 2.5 Episode termination

`has_episode_ended()`: episode ends when `current_budget < min_budget`, `current_budget > max_budget`, or `steps_done >= max_episode_steps`.
Upon termination, `reset_environment` is called, which calls `self.env.reset()` (restores `current_budget = init_budget`, restarts the curriculum back to `init_knowledge` — i.e. any purchased CTI knowledge from the previous episode is *forgotten*, `Knowns`/`G2s` reset to the YAML-configured lists) **and** `init_inference_neural_modules()`, which **re-instantiates the classifier/confidence-decoder/optimizer from scratch and reloads pretrained weights from disk**. This matches the system-description doc's claim that the IM is "reset to its pre-trained state at the start of every episode" — confirmed at code level, and it means the IM's within-episode learning (closed-set/AD/KR finetuning, and any new replay buffers from CTI purchases) is entirely discarded between episodes; only the DM's RL weights persist across episodes (the `mitigation_agent` is constructed once in `TigerBrain.__init__`, never reconstructed by `reset_environment`).

## 3. Per-tick orchestration (how the "game" actually unfolds in real time)

`online_inference` is the single tick-level entry point, run once per `process_input` call which is inside an unthrottled loop in `smart_check`.

1. IM forward pass in `eval()` mode (`with torch.no_grad()`), on a batch merged from (i) k-shot support samples drawn from per-class replay buffers and (ii) the fresh online flows (`prepare_online_batch`).
2. Anomaly detection splits the online samples into "known" vs. "predicted-unknown."
3. For known-predicted samples: those samples are grouped by the IM's predicted closed-set class, and **one DM decision per predicted-class group** (`act_on_known_traffic`), sequentially, with budget propagated between group-decisions within the same tick — looking at the correctness of just that group's members, not the whole tick's known sub-batch at once.
4. For unknown-predicted samples: kernel-regression/ground-truth clustering groups them into clusters (`collective_anomaly_detection`), then **one DM decision per cluster** (`act_on_unknown_clusters`), sequentially, with budget propagated between cluster-decisions within the same tick.
5. `self.mitigation_agent.replay(step)` is called once per tick regardless of how many transitions were just pushed (could be 0 if `num_known==0` and `num_anom==0`, or 1+ per known-traffic group plus 1+ per cluster).
6. *Only after* all DM decisions for the tick: `train_inf_module_single_batch()` runs one gradient step for the IM, sampling i.i.d. from replay buffers (decoupled from the exact online batch). This ordering — act first (frozen IM), train second — is what keeps the loop well-defined despite the IM changing under the DM's feet between ticks.

The DM step counter (`wb_tracker.step_counter`) increments once per individual action taken (not once per tick) when `agency=True` — so an "episode" in terms of `max_episode_steps` is counted in units of *individual block/accept/CTI decisions*, including potentially several per tick for multi-group and multi-cluster ticks, which is consistent with the per-decision `remember()` calls feeding the replay buffer.

## 4. Summary of simplifications/assumptions baked into the code (not paper claims, code facts)

1. **Confidence is per-group/per-cluster in the slot that matters, zero in the other slot**: `cs_classif_confidence` is recomputed per predicted-class group in known-traffic decisions and `zda_confidence` is recomputed per cluster in unknown-cluster decisions (each via its own `_cs_confidence_for_slice`/`_zda_confidence_for_subset` call, scoped to that group's/cluster's members); the *other* confidence slot (the one not pertinent to that decision's regime — `zda_confidence` for known-traffic states, `cs_classif_confidence` for unknown-cluster states) is always `0.0`, which also tells the agent which regime the state belongs to.
2. **State staleness across group/cluster decisions**: only the centroid, budget, and (for unknown clusters) the epistemic-flag scalar update between sequential decisions in the same tick; `num_anom`/`num_known` are frozen tick-level snapshots throughout the tick's whole decision sequence.
3. **Known-traffic decisions are per predicted-class group, not per-flow**: one accept/reject action governs each group of known-predicted samples that share the same IM-predicted class within the tick, with reward summed over that group's members; a tick with several distinct predicted classes credits several known-traffic decisions.
5. **IM is fully reset every episode** (weights reloaded from disk, replay buffers and curriculum reset); only the DM's network/memory persists across episodes.
6. **CTI price only decays, never rises**, and decays multiplicatively and stochastically every single decision step (not per-episode or per-purchase) when enabled.
7. **Greedy-CTI / periodic-CTI / no-epistemic-actions are exclusive override modes** that replace the learned policy's action for the epistemic dimension — useful as ablations, but mean "agent always decides" is only true in the default/else branch.
8. **Reward magnitudes are a fixed, hand-authored per-class lookup table** (`rewards:` YAML), and that table is the only tunable: there are no separate multiplicative penalty factors or "hard"/"easy" mode layered on top — accept and block share one rule (`_decision_reward`) everywhere.
9. **Traffic arrival is exogenous**: the DM's actions never influence which flows/classes appear next; they only affect the budget/reward and the curriculum (which classes are "known" vs. purchasable), not the environment's "physics."
10. **IM training batches are decoupled from the exact batch the DM just acted on** — IM gradient steps sample fresh i.i.d. batches from replay buffers rather than the literal online tick batch, which is what allows the act-then-train ordering to not create within-tick circular dependency, but also means the IM's loss is not a function of the DM's most recent decisions.

## 5. Offline replay: re-running the exact same DM/IM game from recorded data

`offline_replay.py` is a second way to exercise the entire mechanism described
in §1–§3, with no FastAPI/POX/uvicorn server and no live network traffic. It
reconstructs a real `TigerBrain` instance and feeds it samples that were
previously captured by `FlowDataRecorder` (`data_recorder.py`) during a live
run with `intrusion_detection.data_collection_mode: true`, replaying them
through the *same* `TigerBrain._process_batch` path that `process_input` uses
online. Concretely, `process_input_from_record` (`tiger_brain_new.py:1328`)
wraps one recorded tick's tensors into a `Batch`, restores the ground-truth
`class_labels` from the recorded label strings, and calls `_process_batch` —
none of the DM/IM coupling, MDP, reward, or episode-termination logic
described above is re-derived or approximated; it is the original online code
running on recorded inputs instead of live ones.

### 5.1 Where a collection run's data lives, and what's in it

A collection run is a directory `run_<timestamp>/` (under
`intrusion_detection.data_collection_dir`, e.g.
`/pox/pox/smartController/tiger_data_collection/`) containing:

- **`manifest.json`** — written once, at `FlowDataRecorder.__init__` time, via
  `TigerBrain._build_data_collection_manifest()` (`tiger_brain_new.py:212`).
  This is a full snapshot of the exact config the live run used to construct
  `TigerBrain`/`NewTigerEnvironment`: the `intrusion_detection`,
  `neural_modules`, `knowledge`, `rewards`, `health`, and `wandb` config
  blocks (the same ones documented in §0 and used throughout the rest of
  this file), plus `traffic_dict`, `container_ips`, `ips_containers`,
  `use_packet_feats`, `use_node_feats`, `flow_feat_dim`, `packet_feat_dim`,
  `hidden_size`, `device`, and `models` (the raw Python source text of the
  ASAP model classes that gets `exec()`'d, exactly as it does for a live run
  — see system-description §3/§5). It is JSON despite the suggestive name —
  there is no `manifest.yaml` anywhere in the recorded-run layout.
- **`shards_index.jsonl`** — one JSON line appended per flushed shard
  (shard filename, `num_samples`, `tick_min`/`tick_max`), for cheap
  inspection without loading the actual tensors.
- **`shard_NNNNNN.pt`** — `torch.save`d dicts of the buffered `tick`,
  `flow_features`, optional `packet_features`/`node_features`, and
  `element_classes` (ground-truth label strings only — `zda`/`test_zda`
  booleans are deliberately *not* recorded, since they depend on the
  curriculum state, i.e. which classes have been bought via CTI, at capture
  time; replay recomputes them from whatever curriculum the *replaying* run
  is configured with).

### 5.2 Every DM/IM hyperparameter comes from the run's `manifest.json`, not from the script

`offline_replay.py` takes a single positional argument — the `run_dir` — and
has **no flags for any RL/IM hyperparameter** (no learning rate, no
`init_epsilon_egreedy`, no `cti_price_factor`, no `Knowns`/`G1s`/`G2s` lists,
etc.). `build_kwargs()` (`offline_replay.py:148`) reconstructs the exact
kwargs dict `TigerBrain.__init__` expects entirely by copying the
`intrusion_detection`, `neural_modules`, `knowledge`, `rewards`, `health`, and
`wandb` blocks straight out of the loaded `manifest.json`:

```python
kwargs = {
    "intrusion_detection": dict(manifest.get("intrusion_detection", {})),
    "neural_modules": dict(manifest.get("neural_modules", {})),
    "knowledge": dict(manifest.get("knowledge", {})),
    "rewards": dict(manifest.get("rewards", {})),
    "health": dict(manifest.get("health", {})),
    "wandb": dict(manifest.get("wandb", {})),
    ...
    "models": manifest.get("models", ""),
}
```

So every quantity discussed in §2 — action space, reward magnitudes
(`rewards:` lookup table), CTI pricing/decay, budget bounds, epsilon decay,
PER/n-step settings, the agent type (`intrusion_detection.agent`), the
curriculum (`Knowns`/`G1s`/`G2s`), and even which ASAP model source
(`models`) gets `exec()`'d — is whatever that *specific collection run* was
configured with when it captured the data, not anything chosen by
`offline_replay.py` itself. Replaying a given `run_dir` therefore always
re-runs the live experiment's own DM/IM configuration, not the controller's
current `tiger/config/default.yaml` or any override file.

The script only ever *overrides* a handful of fields on top of the
manifest-derived kwargs, and always for a documented, non-hyperparameter
reason:
- `intrusion_detection.data_collection_mode` is always forced to `False`
  (the replay must not spin up a second `FlowDataRecorder` and re-record the
  data it is replaying).
- `--device`, `--pretrained-models-dir`, `--load-pretrained`, `--agency`,
  `--no-save` optionally override the corresponding single field (device,
  checkpoint dir, whether pretrained IM weights are loaded, whether the DM
  acts/learns during replay, whether checkpoints are saved).
- `wandb.wb_tracking` defaults to disabled (`--wandb` opts back in), since a
  replay is not a fresh tracked experiment by default.
- `--set path.to.key=value` is a generic escape hatch to override *any* of
  the six manifest-derived blocks by dotted path (e.g.
  `--set intrusion_detection.agent=DuelingDDQN`), applied last so it can
  override anything else, including the dedicated flags above — but absent
  explicit `--set` overrides, every DM/IM hyperparameter is exactly what the
  manifest says.

Before any of this, `check_feature_coverage()` (`offline_replay.py:234`)
cross-checks `manifest['use_packet_feats']`/`use_node_feats` (what the model
config wanted at capture time) against
`intrusion_detection.data_collection_use_packet_feats`/`_use_node_feats`
(what was actually captured), and aborts loudly rather than silently
replaying `None` tensors into a model that expects real ones.

### 5.3 How a run is replayed once `TigerBrain` is reconstructed

`shards_index.jsonl` entries are read in order, the first entry is skipped
(it covers an initial packet-repetition warm-up artifact from the collection
process itself, not real traffic), and each shard's `.pt` file is loaded and
split into contiguous same-`tick` groups by `iter_tick_groups()`
(`offline_replay.py:275`) — a FlowDataRecorder flush never splits a single
recorded tick across two shards, so per-shard grouping alone is enough to
reconstruct the original tick boundaries. Each tick group is handed to
`brain.process_input_from_record(...)` one at a time, in recorded order,
exactly reproducing the per-tick orchestration of §3 (IM forward pass,
known/unknown split, one DM decision per predicted-class group/cluster,
`replay(step)`, then `train_inf_module_single_batch()`) for that tick's
recorded samples.

`--repetitions` (default `20`) replays the *same* fixed set of shards that
many full passes, in the same order each time (no reshuffling) — the offline
analogue of training for multiple epochs over a fixed dataset, still driven
entirely by the manifest's hyperparameters. `--max-shards`/`--max-samples`
cap how much of the recorded run is consumed; neither changes any DM/IM
hyperparameter, only how much recorded data is fed through them.