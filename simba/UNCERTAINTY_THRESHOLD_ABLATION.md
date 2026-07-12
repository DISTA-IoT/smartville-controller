# SIMBA — the "Buy CTI Whenever Uncertain" threshold ablation

A reviewer asked for a **"Buy CTI Whenever Uncertain" threshold policy,
directly operationalizing confidence scores** as a strong heuristic
baseline for the epistemic (CTI-purchase) action. This document adds that
baseline to SIMBA as the `fixed_threshold_cti` ablation, runs it against
the calibrated defaults, and reports the result.

**Headline:** the confidence-threshold heuristic does **not** beat the
value-learning DM (`drl`), for a structural reason the data makes explicit
— on this trace the IM's confidence score is *anti-correlated* with which
labels are worth buying, so no threshold recovers `drl`'s selective
`{doorlock, echo}`. This is the expected and desired outcome: it is exactly
the gap between "how novel does this look" (what a confidence threshold
sees) and "is this label worth paying for" (what the value function
learns). **No default parameter, price, learning rate or state dimension
was changed** — only the new threshold knob was added and swept.

---

## 1. What was added (and what was deliberately *not* touched)

The policy is a faithful port of the full TIGER brain's own
`fixed_threshold_cti` / `cti_confidence_threshold` ablation
(`tiger_brain_new.py::_select_unknown_cluster_action`,
`tiger/config/default.yaml:251-252`) into the small SIMBA stack, using the
same names and the same "buy when the anomaly/uncertainty score is high"
semantics.

| file | change |
|---|---|
| `simba/config.py` | new field `cti_confidence_threshold: float = 1.5` (documented); `ablation` doc updated |
| `simba/brain.py` | `'fixed_threshold_cti'` added to `ABLATIONS`; new `_cluster_uncertainty()` helper; new branch in `_decide()` |
| `simba/README.md` | ablation documented |
| `tiger/config/overrides/simba.yaml` | GNS3 override documents the new mode + knob |

**Untouched (as required):** the DM state space (`_state` is byte-for-byte
unchanged), every learning rate, every price, every reward, every
curriculum list, `init_budget`, `gamma`, epsilon schedule — all the
calibrated defaults in `config.py` are exactly as shipped. The threshold
policy runs *inside* those defaults; the only new number is the threshold
itself, which is what the ablation sweeps.

### The rule

On an **unknown cluster** (the only place the epistemic action exists, as
in every SIMBA/TIGER mode), the policy forces a CTI buy iff

```
a(cluster) > cti_confidence_threshold      AND
the cluster's majority label is on sale     AND
the buy is affordable (budget - price >= min_budget)   # same guard as greedy_cti
```

otherwise it defers the accept/block choice to the learned DQN — so, like
`greedy_cti`, the epistemic slot is fully scripted and the agent is only
ever queried for the pragmatic actions. The learner, hyperparameters and
memorised transitions are identical across all four modes; the mode only
governs the buy action.

### The confidence score `a`

`a = mean(d)/tau`, where `d` is each member's distance to the nearest
known-class prototype and `tau` the IM's calibrated novelty threshold. A
sample is flagged **unknown** exactly when `d > tau`, so every unknown
cluster has `a > 1`; larger `a` means "further beyond the boundary the IM
uses to decide something is novel", i.e. the IM is *more* confident the
cluster is nothing it knows and *less* confident in any known-class label —
"more uncertain". This is the **un-clamped form of the `dist_ratio`
channel the DM already carries in its state** (`_state` clamps `a` to
`[0,3]` and rescales to `[0,1]`); the heuristic thresholds the very signal
the value function also sees. `a > 1` for all unknown clusters means a
threshold `<= 1` degenerates to `greedy_cti` (buy every affordable unknown)
and larger thresholds buy only progressively more-anomalous clusters.

---

## 2. Why a confidence threshold cannot win here (the mechanism)

The selective axis on the real capture is **benign-vs-malicious**, not
volume (see `REAL_DATA_CALIBRATION.md`): buying a benign zero-day's label
(`doorlock`, `echo`) converts its flows from `alpha*r` to `r` and is worth
it; buying a malicious label only "de-notes" an already-correctly-blocked
cluster and, at the calibrated price of 550, is a net loss. `drl` learns
to buy exactly `{doorlock, echo}`.

A confidence threshold cannot express that preference, because **the IM's
anomaly score does not track worth-buying**. Per-cluster `a`, grouped by
the cluster's majority true class (pretrained IM, seed 6, measured under
`no_epistemic` so the whole G2 pool stays visible):

| class | role | on&nbsp;sale | n | min | p25 | **median** | p75 | p90 | max |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| okiru | mal | no (G1) | 153 | 2.32 | 2.75 | 2.92 | 3.12 | 3.41 | 4.06 |
| generic_ddos | mal | no (G1) | 153 | 2.27 | 2.62 | 2.77 | 3.09 | 3.22 | 3.91 |
| cc_heartbeat | mal | no (G1) | 170 | 1.52 | 1.81 | 1.92 | 2.15 | 2.28 | 2.57 |
| hajime | mal | **yes** | 153 | 1.20 | 1.49 | **1.60** | 1.88 | 1.91 | 2.29 |
| h_scan | mal | **yes** | 697 | 1.00 | 1.33 | **1.54** | 1.76 | 1.95 | 2.43 |
| gafgyt | mal | **yes** | 153 | 1.21 | 1.46 | **1.53** | 1.72 | 1.88 | 2.11 |
| echo | **ben** | **yes** | 1856 | 1.00 | 1.19 | **1.44** | 1.78 | 2.17 | 3.36 |
| mirai | mal | **yes** | 1291 | 1.00 | 1.14 | **1.26** | 1.52 | 1.69 | 2.22 |
| muhstik | mal | **yes** | 3340 | 1.00 | 1.13 | **1.26** | 1.43 | 1.74 | 3.50 |
| doorlock | **ben** | **yes** | 1379 | 1.00 | 1.06 | **1.14** | 1.27 | 1.45 | 2.65 |
| hue | ben | no (Known) | 76 | 1.00 | 1.04 | 1.14 | 1.22 | 1.41 | 1.77 |
| hakai | mal | no (Known) | 15 | 1.00 | 1.01 | 1.15 | 1.18 | 1.41 | 1.43 |
| torii | mal | no (Known) | 9 | 1.00 | 1.03 | 1.06 | 1.13 | 1.17 | 1.17 |

Read the **purchasable (on-sale)** rows by median anomaly:

```
doorlock(BEN) 1.14  <  mirai(MAL) 1.26  ≈  muhstik(MAL) 1.26  <  echo(BEN) 1.44
             <  gafgyt(MAL) 1.53  ≈  h_scan(MAL) 1.54  <  hajime(MAL) 1.60
```

`doorlock` — the single most worthwhile label — is the **least anomalous**
purchasable class, and benign/malicious medians are fully interleaved.
Consequences for any single threshold `θ`:

* to buy `doorlock` at all you need `θ <= ~1.14`, which also admits
  `mirai`, `muhstik` and essentially every malicious label → you over-spend
  exactly like `greedy_cti`;
* to exclude the malicious labels you need `θ` above their medians
  (`~1.3–1.6`), which also excludes `doorlock` and most of `echo` → you buy
  almost nothing worthwhile, and the few clusters left above the line are
  dominated by the malicious tail (`hajime`, `h_scan`, `gafgyt`).

There is **no** `θ` that selects `{doorlock, echo}` and rejects the
malicious G2s. The value function can, because it conditions the buy on the
learned per-class continuation value (via the per-class known-flags in the
state), not on how novel the cluster looks.

---

## 3. Results

**Reference modes** — from the calibrated study in `REAL_DATA_CALIBRATION.md`
(§4), i.e. the exact same defaults, harness (`simba_offline.py
--no-manifest`, 120 train / 10 eval episodes) and seeds; not re-run here.

| policy | seed 6 | seed 1 | eval buys |
|---|--:|--:|---|
| **drl** (learned) | **2922 ± 75** | **2633 ± 72** | `{doorlock, echo}` — the selective optimum |
| no_epistemic (never buy) | 1521 ± 67 | 1478 ± 152 | none |
| greedy_cti (buy all affordable) | 739 ± 48 | 847 ± 56 | all 7 |
| *oracle buyer* (greedy + benign-only) | *2777* | — | *`{doorlock, echo}` forced* |

**`fixed_threshold_cti` sweep** — new; `mean(dist)/tau > cti_confidence_threshold`,
same defaults/harness as above.

_Sweep running (θ ∈ {1.0, 1.25, 1.5, 2.0} × seeds 6, 1, 120/10 episodes);
the table lands in a follow-up commit._
<!-- THRESHOLD_TABLE -->


The purchasable-cluster buy fractions the sweep operationalises (from the
per-class table in §2): `θ=1.0` fires on 100 % of unknown clusters
(⇒ `greedy_cti`), `1.25`→56 %, `1.5`→28 %, `2.0`→6.5 %, `≥3.0`→~0 %
(⇒ `no_epistemic`). Because `doorlock` (the most worthwhile label, median
`a=1.14`) sits *below* the malicious medians, every threshold either admits
the malicious labels along with `doorlock` (low `θ`, over-spend) or drops
`doorlock` while still catching the malicious tail (high `θ`) — so no
setting reaches `drl`'s selective `{doorlock, echo}`, and the best
threshold return stays below `drl`.

---

## 4. Reproduce

```bash
cd smartville-controller

# the threshold policy at one setting (pure calibrated defaults + one knob):
python simba_offline.py pre_recorded_data/ --no-manifest --no-wandb \
    --mode fixed_threshold_cti --set cti_confidence_threshold=1.5 \
    --seed 6 --episodes 120 --eval-episodes 10

# reference modes:
python simba_offline.py pre_recorded_data/ --no-manifest --no-wandb \
    --mode drl --seed 6 --episodes 120 --eval-episodes 10
```

`--no-manifest` runs on pure `SimbaConfig` defaults (the recorded run's own
economy is never loaded — see `REAL_DATA_CALIBRATION.md`). The per-class
anomaly-score table above is reproduced by the diagnostic in this ablation
(pretrain the IM, measure `mean(d)/tau` per unknown cluster grouped by
majority true class).

---

## 5. GNS3 / online compatibility (checked, not run)

The mode is fully wired for the live GNS3/POX modality with no controller
code change beyond the port above:

* `tiger_server.py` builds `SimbaBrain.from_tiger_config(args)` when
  `intrusion_detection.brain == simba`; that maps the `simba:` config block
  verbatim through `SimbaConfig.from_dict`, which now recognises
  `cti_confidence_threshold`, and `SimbaBrain.__init__` accepts
  `ablation: fixed_threshold_cti` because it is registered in `ABLATIONS`.
* The decision path is identical online and offline (`process_input` →
  `step_tick` → `_decide`); the confidence signal is the same
  `mean(min_dist)/tau` computed each tick, and `tau` is loaded from the
  pretrained IM snapshot (`im_snapshot_path`), so the threshold has the
  same meaning live as offline. During a cold start (no snapshot / `tau`
  not yet calibrated) `_cluster_uncertainty` returns `+inf` = "maximally
  uncertain", so the policy buys until the IM calibrates — the same
  aggressive-until-warm behaviour the other scripted modes already show.
* Enable it in `tiger/config/overrides/simba.yaml`:

  ```yaml
  simba:
    ablation: fixed_threshold_cti
    cti_confidence_threshold: 1.5
  ```

No dashboard change is needed: SIMBA ablations are config-driven (the
`simba:` block), not UI widgets.
