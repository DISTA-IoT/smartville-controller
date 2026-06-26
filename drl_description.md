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

These three are mutually exclusive ablation modes, not part of the "default" DM behavior — confirmed against `tiger/config/default.yaml`, where `greedy_cti: False`, `cti_period: -1`, and `no_epistemic_actions: false` are all off, so the actual learned-agent path is the final fallthrough in `_select_unknown_cluster_action` (the old `else` branch), which returns the agent's own action unmodified.

For known-class traffic, the action space is collapsed to a binary choice in effect: if `automatic_cs_acceptance=True` (a config flag, default `false`), action is hard-coded to `0`; otherwise the agent is queried, but **only actions `0` (accept the classifier's verdict) or "anything else" (reject/no-confidence) are semantically distinguished** in the reward logic — see §2.4. So even though `act()` can return `0/1/2` here too, the code only branches on `action_signal.item() == 0` vs. not; there's no real CTI semantics for known traffic, just "trust the IM" vs. "discard its classification and penalize."


### 2.2 State space

`self.state_space_dim = hidden_size [+ hidden_size if use_node_feats] [+ hidden_size if use_packet_feats] + 6`. The trailing 6-dim block (called "proprioceptive" in the comments) is, in order, assembled by `assembly_state_vector`:

```
[ centroid (hidden_size dims) ,
  num_anom, zda_confidence, num_known, cs_classif_confidence,
  epistemic_actions_available (0/1), current_budget ]
```

Two distinct call sites populate this differently:
- **Known-traffic step** (`act_on_known_traffic`): centroid is a placeholder, `-1 * ones(hidden_size)` — i.e., the exteroceptive part carries *no* information for known traffic; the state is purely the 6 scalar proprioceptive features plus a dummy centroid.
- **Unknown-cluster step** (`act_on_unknown_clusters`): centroid is the actual mean hidden vector of one anomalous cluster, one DM decision *per identified cluster*, looped (`for idx, centroid in enumerate(centroids[~missing])`, `tiger_brain_new.py`).

So strictly, the "state" the agent sees is heterogeneous by construction: a same-shaped vector that means "no exteroceptive info, here are some scalars" for known-traffic ticks, and "here's a specific anomaly cluster's centroid + scalars" for unknown-cluster ticks — the network can presumably learn to distinguish the two regimes via the `-1`-filled placeholder, but there's no explicit one-hot "mode" flag.

`zda_confidence` and `cs_classif_confidence` are *batch-level* scalars computed once per tick by `evaluate_zda_confidence`/`perform_cs_inference` under one of four interchangeable strategies set by config `confidence_strategy`, then broadcast identically into every per-cluster state vector that tick. This is a simplification: the confidence summary is not per-cluster, it's a single global proxy reused across all clusters processed in the same tick.

### 2.3 Transitions

There is no explicit `P(s'|s,a)` model — transitions are produced by direct simulation inside the loops, not sampled from a stored distribution:

- **Known-traffic transition**: `new_state = state_vec.clone(); new_state[-1] = self.env.current_budget` (post-reward budget). Only the budget scalar changes; everything else (the dummy centroid, anomaly/known counts, confidences) is copied unchanged from the pre-action state. This is a strong simplification — the "next state" doesn't reflect any new observation, just the budget update.
- **Unknown-cluster transition**: `next_state[:-6] = centroids[~missing][idx+1]` if there's a next cluster in this tick's batch, else `-1*ones(...)`; `next_state[-2] = epistemic_actions_available` (post-purchase, can flip 0→1... actually 1→0 if last G2 was just bought); `next_state[-1] = current_budget` (post-reward). So within one inference tick, the sequence of per-cluster decisions is chained as a genuine sequential MDP — cluster *i*'s "next state" is literally cluster *i+1*'s centroid — but `num_anom`/`num_known`/confidences are **not** updated between clusters in the same tick (they stay fixed at the tick's batch-level values, only sliced from the *current* `state_vec` which already has them baked in before the loop starts).
- **Budget update is the core state-transition dynamic**: `self.env.current_budget += current_reward` (or `+= classification_reward`), accumulated *within* a tick across all per-cluster/per-known decisions before the episode-end check is even made.

The "transition" of the environment as a whole, across ticks, is driven by an exogenous, scripted process: real network traffic arriving from `process_input`/`process_input_from_record`, not by any agent-conditioned generative model of future flows. The DM's actions affect future *reward* (via budget and curriculum) but not which traffic arrives next — traffic arrival is action-independent. This is an important asymmetry vs. a textbook MDP: the "exogenous" part of the state (which classes/flows appear) is unaffected by the policy; only the "endogenous" part (budget, knowledge, replay buffers) is.

### 2.4 Reward

Reward is *not* a single scalar function `r(s,a)` — it's assembled from several independently-configured terms, computed and credited at two different sites:

**(a) Known-traffic reward**:
- If `action_signal == 0` ("trust the classifier"): reward = `sum(|reward_label| for correctly-classified known samples)` + `bad_classif_costs` for misclassified ones, where the penalty for being wrong is `-|reward_label|` if `wrong_inference_penalisation == 'easy'`, or `-|reward_label| * bad_classif_cost_factor` if `'hard'` (default config is `'hard'`, factor `40`).
- Else (any non-zero action signal here): flat penalty `-no_confidence_penalty` (default `0` in `default.yaml`, but configurable, e.g. `tiger.yaml`'s dev override) regardless of how many samples were in the batch — a coarse "reject the whole batch's verdict" penalty.
- Per-sample rewards come from `self.env.flow_rewards_dict`, set straight from the `rewards:` YAML block — fixed scalars per traffic class (e.g., `mirai: -0.20`, `hue: 0.05` in the smaller config, or larger magnitudes in the dista override) — i.e., reward shaping is entirely a human-authored lookup table, not learned or derived from any cost model.

**(b) Unknown-cluster reward**, computed per identified cluster:
- `rewards_if_acc` = sum of true class rewards for that cluster's members (signed: positive for benign, negative for malicious); `cost_if_acc = -relu(-rewards_if_acc)` (i.e. only the malicious-content cost matters if accepting); `benign_per_cluster = relu(rewards) summed per cluster` (the benign reward forgone/preserved).
- If accepted (`action==0`, or `action==2` and `epistemic_is_blocking=False`): `reward = f * cost_if_acc + benign_per_cluster`, where `f = bad_clustering_cost_factor` (default `8`) if `wrong_inference_penalisation=='hard'` else `1.0`.
- If blocked: `reward = -f * bad_classif_cost_factor * benign_per_cluster - uncertainty_blocking_penalty` (default penalty `1.5`) — i.e. blocking penalizes lost benign reward, scaled by *both* `bad_clustering_cost_factor` and `bad_classif_cost_factor` when in `'hard'` mode.
- If epistemic (`action==2`): in addition to the accept/block reward above (gated by `epistemic_is_blocking`), `current_reward -= price_payed`, where `price_payed` is either the actual CTI cost (`|class_reward * current_cti_price_factor|`, from `perform_epistemic_action`) or, if there were no more G2 classes left to buy (a "placeholder" purchase), the flat `useless_epistemic_penalty` (default `7`) — a hard-coded penalty for taking an epistemic action when no information was actually available to buy.
- This reward is then added to `self.env.current_budget`.

**(c) CTI pricing dynamics**: `current_cti_price_factor *= clamp(N(0.7, 0.4), 0.01, 0.99)` is applied stochastically *every* decision step if `price_decay=True` (`price_decay()`, `tiger_environment_new.py`, called at the top of both `act_on_known_traffic` and inside the unknown-cluster loop) — note this is a strictly *decaying* multiplicative random walk (the sampled factor is clamped into `[0.01, 0.99]`, so price can only shrink over time, never recover), bounded below at `1%` of its previous value per single multiplicative step, not in absolute units. In `dista_tiger.yaml` (paper-relevant override) `price_decay` is explicitly `true`. In `default.yaml` it's `false`.

### 2.5 Episode termination

`has_episode_ended()`: episode ends when `current_budget < min_budget`, `current_budget > max_budget`, or `steps_done >= max_episode_steps`.
Upon termination, `reset_environment` is called, which calls `self.env.reset()` (restores `current_budget = init_budget`, restarts the curriculum back to `init_knowledge` — i.e. any purchased CTI knowledge from the previous episode is *forgotten*, `Knowns`/`G2s` reset to the YAML-configured lists) **and** `init_inference_neural_modules()`, which **re-instantiates the classifier/confidence-decoder/optimizer from scratch and reloads pretrained weights from disk**. This matches the system-description doc's claim that the IM is "reset to its pre-trained state at the start of every episode" — confirmed at code level, and it means the IM's within-episode learning (closed-set/AD/KR finetuning, and any new replay buffers from CTI purchases) is entirely discarded between episodes; only the DM's RL weights persist across episodes (the `mitigation_agent` is constructed once in `TigerBrain.__init__`, never reconstructed by `reset_environment`).

## 3. Per-tick orchestration (how the "game" actually unfolds in real time)

`online_inference` is the single tick-level entry point, run once per `process_input` call which is inside an unthrottled loop in `smart_check`.

1. IM forward pass in `eval()` mode (`with torch.no_grad()`), on a batch merged from (i) k-shot support samples drawn from per-class replay buffers and (ii) the fresh online flows (`prepare_online_batch`).
2. Anomaly detection splits the online samples into "known" vs. "predicted-unknown."
3. For known-predicted samples: **one** DM decision for the whole sub-batch (`act_on_known_traffic`), looking at aggregate correctness across all known samples in the tick.
4. For unknown-predicted samples: kernel-regression/ground-truth clustering groups them into clusters (`collective_anomaly_detection`), then **one DM decision per cluster** (`act_on_unknown_clusters`), sequentially, with budget propagated between cluster-decisions within the same tick.
5. `self.mitigation_agent.replay(step)` is called once per tick regardless of how many transitions were just pushed (could be 0 if `num_known==0` and `num_anom==0`, 1, or 1+n_clusters).
6. *Only after* all DM decisions for the tick: `train_inf_module_single_batch()` runs one gradient step for the IM, sampling i.i.d. from replay buffers (decoupled from the exact online batch). This ordering — act first (frozen IM), train second — is what keeps the loop well-defined despite the IM changing under the DM's feet between ticks.

The DM step counter (`wb_tracker.step_counter`) increments once per individual action taken (not once per tick) when `agency=True` — so an "episode" in terms of `max_episode_steps` is counted in units of *individual block/accept/CTI decisions*, including potentially several per tick for multi-cluster ticks, which is consistent with the per-decision `remember()` calls feeding the replay buffer.

## 4. Summary of simplifications/assumptions baked into the code (not paper claims, code facts)

1. **Confidence is tick-global, not per-decision**: `cs_classif_confidence`/`zda_confidence` computed once per tick and broadcast unchanged into every per-cluster state that tick.
2. **State staleness across cluster decisions**: only the centroid and budget/epistemic-flag scalars update between sequential cluster decisions in the same tick; the anomaly/known counts and confidences are frozen tick-level snapshots.
3. **Known-traffic decisions are batch-level, not per-flow**: a single accept/reject action governs the entire known-traffic sub-batch of the tick, with reward summed over all its samples.
5. **IM is fully reset every episode** (weights reloaded from disk, replay buffers and curriculum reset); only the DM's network/memory persists across episodes.
6. **CTI price only decays, never rises**, and decays multiplicatively and stochastically every single decision step (not per-episode or per-purchase) when enabled.
7. **Greedy-CTI / periodic-CTI / no-epistemic-actions are exclusive override modes** that replace the learned policy's action for the epistemic dimension — useful as ablations, but mean "agent always decides" is only true in the default/else branch.
8. **Reward magnitudes are a fixed, hand-authored per-class lookup table** (`rewards:` YAML), with separate multiplicative penalty factors (`bad_classif_cost_factor`, `bad_clustering_cost_factor`, `uncertainty_blocking_penalty`, `useless_epistemic_penalty`) layered on asymmetrically between the accept-branch and block-branch of the unknown-cluster logic.
9. **Traffic arrival is exogenous**: the DM's actions never influence which flows/classes appear next; they only affect the budget/reward and the curriculum (which classes are "known" vs. purchasable), not the environment's "physics."
10. **IM training batches are decoupled from the exact batch the DM just acted on** — IM gradient steps sample fresh i.i.d. batches from replay buffers rather than the literal online tick batch, which is what allows the act-then-train ordering to not create within-tick circular dependency, but also means the IM's loss is not a function of the DM's most recent decisions.