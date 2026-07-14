# TIGER vs SIMBA — remaining differences (scrupulous audit)

Companion to `SIMBA_PARITY.md` (which maps what **was** aligned). This file is
the honest inventory of what still **differs** between the full TIGER stack
running under `tiger/config/overrides/dista_tiger.yaml` and the SIMBA
implementation (`simba/`) that produced the validated offline results. Every
item states what each side does, the expected impact, and — where one exists —
the lever that would close it. Compiled by re-reading both codebases side by
side after the parity work (including the `state: input` port of SIMBA's
raw-input-space DM representation, which is now **closed**).

Legend: 🔴 could plausibly change results · 🟡 second-order · ⚪ cosmetic/negligible.

---

## 1. Inference module

### 1.1 🔴 Encoder architecture
* **SIMBA** (`SimbaEncoder`): fixed parameter-free preprocessing
  (`log1p` on flowstats, `/255` on packet bytes) → GRU(4→64, 1 layer) over the
  flow window; MLP(64→64→64) over the **mean** packet bytes; concat → projection
  head Linear(128→64)+ReLU+Linear(64→64)+**LayerNorm** → a 64-d embedding.
* **TIGER** (`im_models/*` two-stream classifier): **BatchNorm1d** (learned,
  running stats) on each raw stream → [optional MLP encoder] → one GRU **per
  stream** → `ReLU` on the last GRU output (non-negative embedding!) → plain
  concat, **no projection head, no LayerNorm** → a `2·hidden_size`-d embedding
  (400-d at h200, 128-d with the h64 pretraining recipe).
* Impact: different embedding geometry and normalisation; the ReLU'd,
  un-LayerNormed embedding has drifting scale (this is why the DM state was
  moved off it — see §3.1). With `tiger_pretraining.yaml` (h64, 1 GRU layer, no
  dropout, `use_encoder: false`, `packets_per_sample: 1` so the packet GRU
  degenerates to a gated projection of the packet bytes) the *shape* is close,
  but BatchNorm-vs-log1p, the missing projection+LayerNorm, and the 128-vs-64
  width remain.
* Lever: none by config. Closing it fully means a new `im_models` variant with
  SIMBA's exact encoder — but then the classifier's `hiddens` width changes and
  `exteroceptive_proto_dim` bookkeeping must follow ('prototype' mode only;
  irrelevant under `state: input`).

### 1.2 🟡 Classifier similarity / loss geometry
* **SIMBA**: logits = **−cdist²** to prototypes, cross-entropy.
* **TIGER**: scores = **1/(cdist+1e-10)**, cross-entropy (after
  `use_huber_cs: false`). Same nearest-prototype argmax, different gradient
  field (hyperbolic vs quadratic in distance) and different logit scale
  entering CE.
* Lever: an `im_models` variant whose `MulticlassPrototypicalClassifier.forward`
  returns `-cdist**2`. NB: downstream consumers (relational summaries, `energy`
  confidence) assume `1/d` scores — under the parity profile those are off
  (`state: input`, `confidence_strategy: baseline`), so this variant would be
  safe to try.

### 1.3 🔴 Prototype estimation: 8-shot resampled per forward vs 64-sample cached
* **SIMBA**: prototypes recomputed by `calibrate()` every 10 ticks (and after
  every buy) from up to **64** buffer samples per class, then **held fixed**
  for inference between calibrations.
* **TIGER**: prototypes rebuilt on **every forward pass** from the k-shot
  support of a freshly sampled aux batch (**8** samples/class), both at
  inference and training. Higher-variance prototypes, resampled every tick.
* Impact: noisier per-tick class assignments and anomaly distances than SIMBA's.
* Lever: none by config; would require caching prototypes in the brain.

### 1.4 🔴 Inference known-set contains unbought-G2 / G1 prototype columns
* **SIMBA**: `infer()` scores queries **only against trainable prototypes**
  (Knowns + bought CTIs). An unbought G2 has no prototype anywhere.
* **TIGER**: `prepare_online_batch` samples the aux batch from **every**
  class buffer (INFERENCE mode), so every class the wire has ever shown —
  including G1s and unbought G2s — has a prototype column, and the closed-set
  argmax can classify a known-deemed sample *into an unbought G2's column*.
  The **AD decoder** is correctly restricted (`exclude_g1_from_ad_known_set:
  true` drops G1; G2 columns carry zda=1 so `get_known_classes_mask` drops
  them), but CS predictions and known-group formation in
  `act_on_known_traffic` are not.
* Impact: known-traffic groups can be keyed by classes SIMBA could never
  predict; group membership (and hence DM decisions/rewards) can differ.
* Lever: none by config today; would need masking non-trainable columns out of
  `perform_cs_inference`'s argmax.

### 1.5 🟡 AD threshold calibration: gradient-learned τ vs balanced-accuracy search
* **SIMBA**: after each calibration, τ = the candidate distance maximising
  balanced Known-vs-G1 accuracy (exact search); decision `d > τ` is hard.
* **TIGER** (`simba_ad`): τ (and a sharpness scale) are `nn.Parameter`s learned
  continuously by the AD/BCE loss on the same Known-vs-G1 supervision;
  decision `p > 0.5 ⇔ d > τ`. Also note the loss is `BCEWithLogitsLoss`
  applied to an already-sigmoided probability (a legacy quirk shared by every
  TIGER decoder): gradients are squashed twice, so τ moves more slowly than a
  clean BCE would move it.
* Impact: same decision rule, laggier calibration; τ can trail distribution
  shifts (e.g. right after an episode reset) where SIMBA's search snaps to the
  optimum every 10 ticks.

### 1.6 🟡 G1 novelty supervision: explicit margin repulsion vs BCE + kernel loss
* **SIMBA**: G1 queries are pushed at least `ad_repulsion_margin = 4.0` away
  from every Known prototype by an explicit hinge term (weight 1.0), *plus*
  G1 classes participate in the prototypical CE as their own classes.
* **TIGER**: G1s participate as their own classes ✔ (training batches carry
  them, zda=1), and repulsion happens implicitly through (a) the AD/BCE loss
  backpropagating into the encoder (`ad_loss_backprop_to_encoder: true`) and
  (b) the kernel-regression loss repelling different-class pairs. No fixed
  margin anywhere.
* Impact: same intent, different functional form; no guarantee of a minimum
  Known↔G1 separation.

### 1.7 ⚪ IM housekeeping deltas
* TIGER clips the IM gradient norm at 10 (`grad_clip_max_norm`); SIMBA doesn't
  clip. (Keep it: it exists to stop a known divergence mode.)
* Per-episode reset: SIMBA restores a snapshot (encoder + **optimizer state** +
  prototypes + τ + **buffer contents** + feature stats); TIGER reloads the
  on-disk pretrained weights and a **fresh Adam**, and its replay buffers are
  **not** rolled back — they keep accumulating across episodes (bounded by the
  512-capacity ring, so old samples age out anyway).
* IM training gate: SIMBA trains once ≥2 trainable classes have ≥16 samples;
  TIGER once **all** registered buffers exceed `batch_size` (16) and a Known
  class exists (`batch_processing_allowed`). Slightly later start, same idea.

---

## 2. Collective anomaly detection (clustering)

### 2.1 ✅ (closed) Clustering substrate
Was the biggest remaining architectural gap; now ported. `input_space_clustering:
true` (in `dista_tiger.yaml`) makes `collective_anomaly_detection` partition the
predicted-anomalous online samples by **single-linkage connected components in
the standardised raw-input space** — a verbatim port of SIMBA's `cluster()`
(proven byte-identical over 300 random trials) — under a radius calibrated on
the G1 pseudo zero-days exactly as SIMBA does (`_calibrate_cluster_radius`):

```
radius = 1.25 * max_G1(90th-pct within-class pairwise distance)
radius = min(radius, 0.6 * min between-G1-centroid distance)   [>= 2 G1s]
radius *= cluster_radius_factor
```

The G1 samples are drawn from their per-class replay buffers and placed in the
same standardised frame the online clustering uses (`_standardise_raw`,
read-only so calibration draws don't perturb the running stats). The learned
kernel-regression head is still trained and evaluated (`kr_metrics` keep
reporting) — only the partition the DM acts on changed. With `state: input`
the DM already sees the input-space centroid of each cluster, so representation
**and** partition are now both SIMBA's.

The single-linkage extraction is vectorised relative to SIMBA's Python
flood-fill (build the radius adjacency once with `torch.cdist`, extract
components with SciPy's C-level `connected_components`, renumber to
first-appearance order) — byte-identical partition, ~5–70× faster on 50–600
anomalies/tick.

**Composability with `use_neural_KR: false`**: `input_space_clustering` only
governs the *learned* substrate. With `use_neural_KR: false` the gold /
ground-truth partition (cluster by true label) still wins — the perfect-
clustering oracle ablation is unaffected — so the two knobs compose instead of
the input-space path shadowing the oracle.

Two honest residues, neither a substrate difference:

* 🟡 **Radius recalibration cadence**: SIMBA's *code* calibrates the radius
  **once** (`calibrate_clustering` is guarded by `cluster_radius is None`, then
  frozen via snapshot/restore) — the "recalibrated every 10 ticks" claim in the
  old audit was only ever true of SIMBA's prototypes/τ, not its radius. SIMBA
  can freeze after one shot because it fits `feat_mu/feat_sd` **offline** first,
  so that calibration already runs in the mature frame. TIGER has no offline
  pass, so it recalibrates **every tick** (cheap: a few G1s × 64 samples). An
  empirical sweep on the real trace (11 shards, G1s okiru/cc_heartbeat/
  generic_ddos, looped 20×) showed why this matters: the radius drifts **4.43 →
  3.86** over the first ~50 ticks as the running stats mature, then is stable to
  **<1% CoV**. Freezing the tick-0 value (or throttling to every-N-ticks) would
  lock in a ~15%-too-high radius and ~13% fewer clusters; recalibrating every
  tick tracks the maturation and converges to the value SIMBA's mature-frame
  one-shot would reach. No period knob is exposed — it would only be a footgun.
* 🟡 **Standardisation frame**: the same §3.1 residue as `state: input` — TIGER
  standardises by running Welford stats (never reset) rather than SIMBA's
  once-fit-and-frozen `feat_mu/feat_sd`. They converge within a few ticks; the
  radius calibrated in that frame inherits the same convergence (above).

* Lever (was): none by config. Now `input_space_clustering` /
  `cluster_radius_factor` / `cluster_calibration_samples`.

---

## 3. Decision module — state, transitions, rewards

### 3.1 ✅ (closed) Exteroceptive state
`state: input` now feeds the DM the group's/cluster's centroid in SIMBA's
standardised raw-input space ([latest flow row | window mean | mean packet
bytes], log1p / ÷255, per-feature standardised), un-LayerNormed. One residue:

* 🟡 **Standardisation fitting**: SIMBA fits `feat_mu/feat_sd` **once** on the
  pretraining shadow buffers and restores the same values every episode; live
  TIGER has no offline fitting pass, so it uses **running Welford stats over
  every online sample, never reset**. They converge within a few ticks and are
  then effectively frozen, but the first ticks of a run are standardised by
  immature stats, and a drastic traffic-mix change moves them slightly where
  SIMBA's would not move at all.

### 3.2 🔴 Proprioceptive block composition
Same *spirit* (bounded scalars + per-class knowledge flags), different channels:

| channel | SIMBA | TIGER |
|---|---|---|
| group size | ✔ `len/20` (clamped) | ✘ absent |
| regime flag | ✔ explicit `is_unknown` bit | implicit: which confidence slot is zero |
| cluster novelty | `dist_ratio` = clamp(mean d/τ,0,3)/3 | AD-probability confidence (asinh) in the zda slot |
| CS confidence | ✘ | ✔ (asinh) in the cs slot |
| batch anomaly/known counts | ✘ | ✔ log1p(num_anom), log1p(num_known) |
| CTI quote (price) | ✔ `quote/200` | ✘ absent (only the boolean below) |
| CTI on sale | ✔ `has_quote` | ✔ `epistemic_actions_available` |
| budget | linear `/1000` | floor-anchored tanh (`_proprio_budget_feature`) |
| knowledge owned | one flag **per class**, Knowns+bought =1 | acquired-CTI map, **bought only** =1 (initial Knowns are constant ⇒ same information) |
| knowns fraction | ✔ | acquired-CTI fraction (equivalent up to affine) |
| episode progress | ✔ ticks/total | ✘ absent |

* Impact: the two most meaningful absences on the TIGER side are the **quote
  (price) channel** — SIMBA's DM can literally read how expensive the buy on
  offer is; TIGER's must infer value-for-money through the acquired map and
  budget dynamics — and **group size** (volume of the decision). Episode
  progress matters less at fixed horizon.
* Lever: code change to `assembly_state_vector` (add price-of-majority-quote
  and member-count channels). Not done — every extra channel changes the DM
  state dim, and it was out of the "align, don't redesign" scope. Flagging it
  as the top candidate if live DRL still underbuys.

### 3.3 🔴 Transition chaining
* **SIMBA**: decisions chain **globally**: each decision's `next_state` is the
  *following* decision's state, **across tick boundaries** (`_pending`), with a
  zeros-state terminal flush at episode end.
* **TIGER**: decisions chain **within a tick only**, and the known-group loop
  and the unknown-cluster loop chain **separately**; the last decision of each
  loop gets a `-1`-filled exteroceptive placeholder as `next_state` (proprio
  tail kept), `done` = whether the episode has ended at that decision.
* Impact: TIGER's bootstrap sees a synthetic "no next group" state many times
  per episode where SIMBA sees the true successor decision; value propagation
  across ticks is broken at every tick boundary and only survives through the
  (shared) proprio tail. This is a structural MDP difference the config cannot
  express.
* Lever: porting SIMBA's `_pending` chaining into `act_on_known_traffic` /
  `act_on_unknown_clusters` (carry one pending transition on the brain,
  flush on episode reset). Contained change, not done yet.

### 3.4 🟡 Epistemic transition reward
* **SIMBA**: a buy's stored reward = pragmatic outcome of the (non-blocking)
  accept **plus** the price paid, in one number.
* **TIGER**: the transition stored for action 2 carries **only** the epistemic
  cost (−price −waste penalty); the pragmatic part of that same decision flows
  to the budget but not into the stored reward.
* Impact: TIGER's Q(buy) must recover the accept-side upside purely from the
  bootstrap; SIMBA's sees part of it immediately. Same fixed point in theory,
  different learning dynamics.

### 3.5 🟡 Action masking / exploration on known groups
* **SIMBA**: known groups call `act(allowed=[0,1])` — the buy action is masked
  *inside* the agent: random draws split 50/50 over accept/block, greedy takes
  `argmax(Q[[0,1]])`.
* **TIGER**: the agent always chooses over all 3 actions; a chosen 2 on a known
  group is remapped to block **after** the fact. So: random draws effectively
  give block 0.45+0.10 vs accept 0.45, and a greedy `argmax` that lands on
  Q(buy) becomes a block even when Q(accept) > Q(block).
* Impact: mild block-bias on known traffic, both during exploration and (rarer)
  exploitation.

### 3.6 🟡 "No G2 left" behaviour
* **SIMBA** (drl): action 2 always exists on unknown clusters; buying with no
  quote on sale charges `waste_buy_penalty` (25) — the `has_quote` flag is how
  the agent learns to stop.
* **TIGER**: once no unbought G2 remains (`epistemic_actions_available == 0`),
  a chosen 2 is remapped to block — the agent *cannot* waste-buy in that
  state. Wasted buys only occur while G2s remain but the cluster's majority
  label isn't purchasable (already-Known / G1 majority).
* Also wasted-buy pricing: SIMBA charges a flat 25; TIGER charges the target
  label's configured price (0 for Knowns/G1s under the parity `prices`) +
  `useless_epistemic_penalty` 25 — identical for Known/G1 targets, but a
  wasted re-buy of an **already-bought G2** costs its full price + 25 in TIGER
  vs 25 in SIMBA.

### 3.7 ⚪ Replay/update cadence details
* Per-decision replay ✔ (`dm_replay_per_decision`), but TIGER runs the tick's
  N replays back-to-back **after** all decisions rather than interleaved
  between them. With learn-start 1000 and a 100k buffer this is negligible.
* Hard target update is now counted in **gradient steps** inside `replay()`
  when `use_soft_update: false` — this **fixed a real bug** for the parity
  config: the legacy path checked `step_counter % update_target_freq == 0`
  once per tick, and since the counter advances ~20 decisions per tick it
  landed on an exact multiple only by chance, leaving the target net frozen
  for long random stretches. (The legacy per-tick check still also exists and
  can fire occasionally; an extra hard sync is harmless.)
* Uniform replay sampling: SIMBA samples **with** replacement, TIGER's non-PER
  path samples the batch **without** replacement. Negligible at 100k capacity.

### 3.8 🟡 No greedy-evaluation protocol live
SIMBA's headline numbers are **greedy eval episodes** (`training=False`: ε=0,
no DM updates, IM restored). Live TIGER runs are always training runs — the
W&B curves compare exploration-contaminated returns. When comparing live
results to `REAL_DATA_CALIBRATION.md`, compare against SIMBA's *training*
curves, or add an eval phase (freeze ε and skip `remember/replay`) — not
currently expressible by config.

---

## 4. Environment / economics

Fully aligned by config (`SIMBA_PARITY.md` §parameter map): budget 2500/0,
no bankruptcy termination, α=0.25, β=1.0, prices/rewards/curriculum, waste
penalty, non-blocking buys, static prices, no impurity shaping, no AD oracle.
Remaining nuances:

* 🟡 **Episode horizon semantics**: SIMBA ends after `max_episode_ticks` (51 —
  one full trace pass, ~20 decisions/tick); TIGER after `max_episode_steps =
  1000` **decisions**. Equivalent in expectation, but TIGER's decisions/tick
  varies with the clustering (§2.1), so episode *wall-time* and per-class
  appearance counts per episode will drift from SIMBA's fixed 51-tick pass.
  Also, a TIGER tick that crosses the horizon finishes its remaining
  group/cluster decisions (each flagged done) before the reset lands.
* ⚪ CTI purchase mechanics (majority-label targeting, instant delivery,
  G2→Known promotion, per-class buffers pre-filled before purchase, buy burst
  10) — aligned.

---

## 5. Data path

Same by construction (the offline traces were recorded by TIGER's own
`data_collection` pipeline), per the project owner: not audited further. One
reminder: SIMBA's offline runs **skip the warm-up shard** and replay 51 fixed
ticks repeatedly; live traffic is nonstationary in ways the trace never was.

---

## 6. Priority list (if live results still diverge)

1. ✅ **Clustering substrate** (§2.1) — DONE: SIMBA's input-space single-linkage
   + G1-calibrated radius ported into `collective_anomaly_detection`, gated by
   `input_space_clustering` (on in `dista_tiger.yaml`).
2. **Cross-tick transition chaining** (§3.3) — port SIMBA's `_pending`.
3. **Proprio channels** (§3.2) — add the quote/price and group-size channels.
4. **Prototype caching** (§1.3) — 64-sample prototypes recomputed every 10
   ticks instead of 8-shot per-forward resampling.
5. **−d² logits variant** (§1.2) — cheap `im_models` experiment.
6. **Greedy eval phase** (§3.8) — to make live numbers comparable to the
   offline eval numbers at all.
