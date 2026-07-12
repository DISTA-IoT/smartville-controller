# SIMBA — the "Buy CTI Whenever Uncertain" threshold ablation

A reviewer asked for a **"Buy CTI Whenever Uncertain" threshold policy,
directly operationalizing confidence scores** as a strong heuristic
baseline for the epistemic (CTI-purchase) action. This document adds that
baseline to SIMBA as the `fixed_threshold_cti` ablation, runs it against
the calibrated defaults, and reports the result.

**Headline:** across the swept thresholds the heuristic never beats the
value-learning DM (`drl`, mean **2778**) — its best single run is **1748**
(θ=2.0, seed 6) and its best *mean* is **1005** (θ=2.0), below even
`no_epistemic` (never-buy, 1500). The reason the data makes explicit: on
this trace the IM's confidence score is *not aligned* with which labels are
worth buying — `doorlock`, the single most worthwhile label, is the **least
anomalous** purchasable class — so **no threshold recovers `drl`'s selective
`{doorlock, echo}`**; the policy never buys `doorlock` at any θ>1.0. This is
the expected, desired outcome: exactly the gap between "how novel does this
look" (all a confidence threshold sees) and "is this label worth paying for"
(what the value function learns). **No default parameter, price, learning
rate or state dimension was changed** — only the new threshold knob was
added and swept.

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
the cluster's majority true class (the seed-6 pretrained IM, `tau=2.888`,
measured under `no_epistemic` so the whole G2 pool stays visible; same
single-threaded pretrain the runs below use):

| class | role | on&nbsp;sale | n | p25 | **median** | p75 | p90 |
|---|---|---|--:|--:|--:|--:|--:|
| torii | mal | no (Known) | 29 | 1.03 | 1.07 | 1.13 | 1.17 |
| hakai | mal | no (Known) | 26 | 1.04 | 1.08 | 1.13 | 1.49 |
| **doorlock** | **ben** | **yes** | 403 | 1.04 | **1.09** | 1.18 | 1.27 |
| hue | ben | no (Known) | 16 | 1.04 | 1.10 | 1.19 | 1.45 |
| **echo** | **ben** | **yes** | 635 | 1.08 | **1.20** | 1.41 | 1.72 |
| muhstik | mal | **yes** | 2324 | 1.10 | **1.21** | 1.38 | 1.59 |
| mirai | mal | **yes** | 1017 | 1.11 | **1.22** | 1.37 | 1.49 |
| h_scan | mal | **yes** | 621 | 1.20 | **1.31** | 1.43 | 1.58 |
| hajime | mal | **yes** | 147 | 1.22 | **1.36** | 1.52 | 1.61 |
| gafgyt | mal | **yes** | 148 | 1.14 | **1.44** | 1.60 | 1.85 |
| cc_heartbeat | mal | no (G1) | 172 | 1.64 | 1.83 | 2.12 | 2.26 |
| generic_ddos | mal | no (G1) | 153 | 1.62 | 1.93 | 2.20 | 2.36 |
| okiru | mal | no (G1) | 153 | 1.96 | 2.30 | 2.50 | 2.80 |

Read the **purchasable (on-sale)** rows by median anomaly:

```
doorlock(BEN) 1.09  <  echo(BEN) 1.20  ≈  muhstik(MAL) 1.21  ≈  mirai(MAL) 1.22
             <  h_scan(MAL) 1.31  <  hajime(MAL) 1.36  <  gafgyt(MAL) 1.44
```

`doorlock` — the single most worthwhile label — is the **least anomalous**
purchasable class, and `echo` (the other worthwhile buy, median `1.20`) is
sandwiched *between* `doorlock` and the two highest-volume malicious
zero-days `muhstik` (`1.21`) and `mirai` (`1.22`) — statistically
indistinguishable from them. Consequences for any single threshold `θ`:

* to buy `doorlock` at all you need `θ ≲ 1.1`, which also admits every
  malicious label → you over-spend exactly like `greedy_cti`;
* `echo` cannot be separated from `muhstik`/`mirai`: any `θ` low enough to
  buy `echo` also buys those two (which, being the highest-volume classes,
  dominate the spend), and any `θ` high enough to reject them also rejects
  `echo` — and always rejects `doorlock`.

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

| `cti_confidence_threshold` | seed 6 | seed 1 | **mean** | what it buys |
|--:|--:|--:|--:|---|
| 1.0 | 267 ± 562 | 331 ± 503 | **299** | all 7 labels — degenerates to `greedy_cti` |
| 1.25 | 455 ± 359 | 503 ± 456 | **479** | `echo` + 4–5 malicious labels; **never `doorlock`** |
| 1.5 | 925 ± 255 | 142 ± 211 | **534** | s6: `echo`+`mirai`+`muhstik`; s1: `mirai`+`muhstik` only; **never `doorlock`** |
| 2.0 | 1748 ± 283 | 263 ± 100 | **1005** | s6: mostly `echo`; s1: `mirai`+`muhstik`; **never `doorlock`** |

The threshold fires on 100 % of unknown clusters at `θ=1.0` (⇒ `greedy_cti`),
then 44 % / 15 % / 1 % at `θ = 1.25 / 1.5 / 2.0`, and ~0 % by `θ=3.0`
(⇒ `no_epistemic`). Three things stand out:

1. **No setting beats `drl`.** The best single run (seed 6, `θ=2.0`) reaches
   1748 — still ~900 below `drl`'s worse seed (2633) and ~1030 below its
   mean (2778). Every mean is below every reference except `greedy_cti`.
2. **It never buys `doorlock`**, the most worthwhile label, at any `θ>1.0`.
   `doorlock` is the least anomalous purchasable class (§2), so the moment
   the threshold rises enough to skip *any* malicious label it has already
   skipped `doorlock`. The best the policy can do is recover *`echo`* alone
   — half of `drl`'s `{doorlock, echo}` — which is exactly the ~1000-point
   gap to `drl`.
3. **It is brittle across seeds.** At `θ ∈ {1.5, 2.0}` seed 6 happens to
   catch `echo`'s upper tail (return climbs to 1748) while seed 1's IM puts
   the high-volume `mirai`/`muhstik` tails above the line and `echo` below,
   so it spends 550-a-label on malicious CTI and craters to 142–263. A
   *fixed* anomaly threshold selects benign on one IM and malicious on
   another — because it keys on how novel a cluster looks, which the IM's
   idiosyncratic geometry controls, not on whether the label pays off. On
   average the sweep peak (`θ=2.0`, 1005) sits *below even `no_epistemic`*
   (1500): confidence-gated buying, without learning *what* to buy, is worse
   than never buying.

This is the intended result: the heuristic operationalizes the confidence
score faithfully and still loses, because "how novel is this cluster" and
"is this label worth paying for" are different questions — the first is all
a threshold can see, the second is what the value function learns.

---

## 4. Reproduce

```bash
cd smartville-controller

# the whole sweep in this document (theta x seed):
for th in 1.0 1.25 1.5 2.0; do for s in 6 1; do
  OMP_NUM_THREADS=1 python simba_offline.py pre_recorded_data/ \
      --no-manifest --no-wandb --mode fixed_threshold_cti \
      --set cti_confidence_threshold=$th --seed $s \
      --episodes 120 --eval-episodes 10 \
      --run-name fixed_threshold_cti_t${th}_seed${s}
done; done

# reference modes are NOT re-run here -- their numbers are the calibrated
# study's (REAL_DATA_CALIBRATION.md), same defaults/harness/seeds.
```

`--no-manifest` runs on pure `SimbaConfig` defaults (the recorded run's own
economy is never loaded — see `REAL_DATA_CALIBRATION.md`). `OMP_NUM_THREADS=1`
pins the pretrain so `tau` (and thus the exact `a` values) reproduce; the
*qualitative* result — `doorlock` least-anomalous, no threshold selective —
is thread- and seed-robust. The per-class anomaly-score table in §2 is a
diagnostic over the same pretrained IM (measure `mean(d)/tau` per unknown
cluster, grouped by majority true class, under `no_epistemic`).

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
