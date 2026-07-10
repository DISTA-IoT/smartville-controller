# Relational State — Design Notes

**Where:** `tiger_brain_new.py`, class `TigerBrain`.
Config knob: `intrusion_detection.state`, one of `'prototype'` (default),
`'relational'`, `'mixed'`.

This document describes the relational exteroceptive state block **as it
currently exists in code** — not a proposal, the shipped implementation.

## 0. The three state spaces

The DM's exteroceptive block can be built in one of three ways, selected by
the `state` knob:

- **`prototype`** (default, legacy): the raw hidden-space centroid of the
  group/cluster (§1).
- **`relational`**: the fixed-size relational summary of the group's/
  cluster's similarity to the known-class prototypes described in the rest
  of this document (§2 onward).
- **`mixed`**: the concatenation `[centroid | relational summary]`, in that
  fixed order, so the DM sees the absolute embedding and the class-relative
  summary at once. Its width is `prototype`'s width `+ RELATIONAL_STATE_DIM`.

Both `relational` and `mixed` build (and, in the known-traffic regime,
write to) the relational-summary machinery below; `prototype` uses none of
it. `_exteroceptive_block(centroid, score_slice)` is the single place that
assembles the correct block per mode, keeping the concatenation order in
sync with `exteroceptive_dim` (§5).

The legacy boolean `relational_state: true` is still accepted (mapped to
`'relational'`) when `state` is absent; internally `self.relational_state`
remains a derived boolean, `True` for both `'relational'` and `'mixed'`.

---

## 1. What problem it solves

The Decision Module's (DM's) state vector has two parts: a fixed-size
**proprioceptive** block (budget, counts, confidences) and an
**exteroceptive** block that describes the group of flows or the anomaly
cluster the DM is currently deciding on.

The legacy exteroceptive encoding was the group's/cluster's **raw
hidden-space centroid** — a vector whose width scales with `hidden_size`
(and with the number of active feature streams: flow [+ node] [+ packet]).
That ties the DM's input size to representation/config choices, and gives it
absolute coordinates that drift as the encoder keeps training online.

`state='relational'` replaces the centroid with a **fixed-size summary of
the group's/cluster's similarity to the known-class prototypes** — i.e. a
function of the prototypical classifier's logits, never of absolute hidden
coordinates. This is invariant to how many known classes currently exist (K
grows as CTI is bought) and stays consistent with the prototypical /
relational-bottleneck inductive bias already used by the perception layer.
`state='mixed'` keeps the centroid **and** appends this summary, trading the
two invariances above for strictly more information.

`TigerBrain.RELATIONAL_STATE_DIM` fixes the width of this block; the DM's
`exteroceptive_dim` (and therefore its net input size) is derived from it
automatically in `init_agents` whenever the mode includes the summary
(`relational` or `mixed`).

---

## 2. Current vector: 9 dimensions

Built by `TigerBrain._relational_summary(score_slice)`, where `score_slice`
is `[n_members, K]` — the group's or cluster's members' rows of prototypical
similarity logits against the K known-class prototypes.

```
[ s_max, r_max, s_mean, s_min, s_std, margin, r_runnerup, entropy, energy ]
```

Let `s = score_slice.mean(dim=0)` — the `[K]` mean similarity to each known
prototype. Then:

| Stat | Definition | Meaning |
|---|---|---|
| `s_max` | `s.max()` | similarity to the **nearest** known prototype |
| `s_mean` | `s.mean()` | average similarity across all known prototypes |
| `s_min` | `s.min()` | similarity to the **farthest** known prototype |
| `s_std` | `s.std(unbiased=False)` (`0` if `K == 1`) | spread of similarity across prototypes |
| `margin` | `top2[0] - top2[1]` (`= s_max` if `K == 1`) | top-2 ambiguity: how much the nearest prototype beats the runner-up |
| `entropy` | `-(softmax(s) * log(softmax(s))).sum()` | how spread-out the assignment distribution is |
| `energy` | `logsumexp(s)` | overall magnitude of match to the known set |
| `r_max` | running accept-reward of the class at `argmax(s)` | *"has the class this most resembles paid off?"* |
| `r_runnerup` | running accept-reward of the class at the 2nd-highest `s` | *"has the runner-up class (the one the margin is against) paid off?"* |

`r_max` and `r_runnerup` are **not** the maximum/minimum reward value —
they are the reward *of the class that owns* `s_max` / the runner-up
similarity respectively. They are deliberately interleaved next to the
similarity stat they're tied to, rather than appended at the end, and are
indexed by **relational rank** (nearest / runner-up), never by class
identity — putting a specific class's reward at a fixed vector position
would break the relational bottleneck (the state would stop being invariant
to which classes are currently known).

`s_min`'s counterpart reward was deliberately **not** added: the farthest
prototype is an arbitrary, low-relevance class, so its reward carries
essentially no decision-relevant signal.

---

## 3. The reward feature: what it is and how it's computed

### 3.1 What "reward" means here

This is **not** the platonic per-class value from `tiger/config`'s
`rewards` block (`flow_rewards_dict` in `tiger_environment_new.py`). It is
the **empirical, possibly-noisy group reward the DM actually received** when
it chose to accept a group the IM had just classified as a given class —
i.e. exactly the `classification_reward` computed by `_decision_reward` in
`act_on_known_traffic`, before the (optional) classification-accuracy bonus
is added. Because groups are built from the IM's *predicted* class, a
reward sample can be "spurious" whenever the IM misclassifies — that noise
is intentionally part of the signal the DM sees, not filtered out.

Blocking a group always earns a reward of exactly `0` (see
`_decision_reward`: block ⇒ `return 0`), so **only accepted groups are
recorded**. If blocked-group zeros were folded in too, a class the policy
has learned to block would have its estimate dragged toward zero regardless
of how good or bad accepting it actually is — masking the exact information
the DM needs when reconsidering that class.

### 3.2 Bookkeeping (per predicted class, per episode)

Three helper methods on `TigerBrain`:

- **`_reset_relational_reward_tracker()`** — (re)initializes:
  - `_class_reward_sum: dict[class_idx -> float]`
  - `_class_reward_count: dict[class_idx -> int]`
  - `_reward_abs_total`, `_reward_abs_n`, `_reward_abs_scale` (running mean
    of `|reward|` across all recorded groups, any class)

  Called once in `__init__` and again at the top of every
  `reset_environment()` call (i.e. every episode). **The averages are not
  persisted across episodes.** This mirrors the inference module's own
  per-episode reset and the "no absolute prototypes" design stance: DM
  network *weights* persist across episodes (that's how it learns), but the
  *per-episode reward statistics* must not, or the agent would effectively
  be memorizing a fixed curriculum instead of learning to size up
  never-seen-before classes from scratch each episode.

- **`_update_relational_reward(class_idx, group_reward)`** — called from
  `act_on_known_traffic`, immediately after `_decision_reward` computes the
  group's pragmatic reward, **only when `accepted_group` is `True`**:

  ```python
  if self.relational_state and accepted_group:
      self._update_relational_reward(unique_classes[idx], classification_reward)
  ```

  It accumulates `sum`/`count` for that class index (running mean) and
  folds `|group_reward|` into the global running `_reward_abs_scale`.

  The key, `unique_classes[idx]`, is the IM's **predicted** class for that
  group — the same index space as the columns of the prototypical logits
  (`interest_logits_slice`) that `_relational_summary` reads `s` from, so
  the value written here and the value read back by `argmax(s)` /
  runner-up lookup always refer to the same class.

- **`_class_reward_feature(class_idx, device, dtype)`** — the read side,
  called from inside `_relational_summary`:

  ```python
  count = self._class_reward_count.get(idx, 0)
  if count == 0:
      return 0.0   # neutral: no evidence yet this episode
  mean_r = self._class_reward_sum[idx] / count
  scale = self._reward_abs_scale if self._reward_abs_scale > 1e-8 else 1.0
  return tanh(mean_r / scale)
  ```

  - **Cold class** (not accepted yet this episode, `count == 0`): returns
    `0.0`. This is a deliberate coincidence with the value of a block
    (`_decision_reward` returns `0` on block) — "no evidence this is worth
    accepting" reads the same as "blocking earns nothing", which is a
    reasonable prior for an untested class.
  - **Scale normalization**: the relational summary is deliberately built to
    be self-scaled, because the DM nets do **not** LayerNorm it. Each channel
    is bounded/scaled by construction — the similarity stats live in log-score
    space at O(tens) (see §3.2 of the code comment / the log-score fix), the
    entropy in `[0, log K]`, and the two reward channels are squashed through
    `tanh` into `(-1, 1)`. The reward feature in particular is divided by the
    running mean of `|reward|` seen so far this episode, then `tanh`'d, so a
    reward in budget-units sits on the same rough footing as the similarity
    logits without a config knob. See §7 for exactly which part of the
    exteroceptive block **is** LayerNorm'd (the prototype centroid) and why the
    relational summary is left alone.

### 3.3 Read side inside `_relational_summary`

```python
if k > 1:
    top2 = torch.topk(s, 2)
    margin = top2.values[0] - top2.values[1]
    k_max, k_run = top2.indices[0], top2.indices[1]
else:
    margin = s_max
    k_max = torch.argmax(s)
    k_run = k_max   # single known class: nearest and runner-up collapse

r_max = self._class_reward_feature(k_max, s.device, s.dtype)
r_runnerup = self._class_reward_feature(k_run, s.device, s.dtype)
```

`k_max`/`k_run` come from the exact same `torch.topk(s, 2)` call that
produces `margin`, so `r_max` is guaranteed to refer to the class `s_max`
scores, and `r_runnerup` to the class `margin` is measured against — no
separate re-derivation, no risk of the two going out of sync.

---

## 4. Call sites

`_relational_summary` (and therefore this reward machinery) is reached via
`_exteroceptive_block` in two places, whenever `self.relational_state` is
`True` (mode `relational` or `mixed`):

1. **`act_on_known_traffic`** (known-traffic regime) — builds
   `group_exteroceptive` per predicted-class group from
   `interest_logits_slice[mask]`, and (as described above) is also the
   place that *writes* to the reward tracker after each group's DM action.
2. **`act_on_unknown_clusters`** (unknown/zero-day regime) — builds
   `cluster_exteroceptive` per identified anomaly cluster from
   `anomalous_logits[...]`. This call site is **read-only**: an anomalous
   cluster's relational summary can reflect the accept-reward history of
   the known class it happens to resemble, but nothing here writes back to
   the tracker (there is no "predicted known class" ground truth for a
   genuinely unknown cluster to attribute a reward to).

---

## 5. Dimension bookkeeping

`RELATIONAL_STATE_DIM = 9` must stay equal to the number of elements
`_relational_summary` stacks. `init_agents` builds `exteroceptive_dim`
additively, in the fixed order `[prototype | relational]`:

- the prototype part (`hidden_size` [`+ node`] [`+ packet`]) is added when
  the mode includes the centroid (`prototype` or `mixed`);
- `RELATIONAL_STATE_DIM` is added when the mode includes the summary
  (`relational` or `mixed`).

`state_space_dim` follows from that, so every DM net's input width adapts
automatically — to any of the three modes, and to a change in either the
stream count or `RELATIONAL_STATE_DIM`. `_exteroceptive_block` assembles
each group's/cluster's vector in the **same** order, so the layout the nets
are sized for always matches the vectors they are fed.

---

## 6. Known limitations / things this design does *not* yet do

- **No confidence weighting.** A reward sample is folded in with full
  weight regardless of how confidently the IM classified that group as the
  class in question. A low-confidence (likely-wrong) group can therefore
  contaminate a class's running average as much as a high-confidence one.
  A natural extension is a parallel running average of
  `cs_classif_confidence` per class, to let the DM (or the feature itself)
  discount reward estimates built on shaky classifications. Not
  implemented yet.
- **Reward magnitude vs. mean.** The recorded quantity is the *group's*
  total reward (`classification_reward`, a sum over the group's members),
  not a per-sample mean — so it scales with group size. The `tanh` +
  running-`|reward|`-scale normalization absorbs most of this, but it has
  not been empirically checked for bias from group-size variance.
- **Cold-start ambiguity.** `0.0` for an unaccepted class is deliberately
  neutral, but it is indistinguishable from "this class truly earns nothing
  when accepted" until the confidence-weighting extension above lets the
  DM tell the two apart.

---

## 7. Exteroceptive normalization (per `state` mode)

The DM decision nets (`PolicyNet`, `ValueNet`, `NEFENet`, `DQN`,
`DuelingDQN`) normalise the **prototype (raw-centroid) part** of the
exteroceptive block and leave the relational summary alone. This is one
module, `neural_modules.ExteroceptiveNorm` (built by `_make_extero_norm`),
applied where `proprio_norm` is applied. Its width comes from
`exteroceptive_proto_dim`, injected by `init_agents` (the leading-columns
width of the centroid; `0` in pure `relational` mode). Behaviour per mode:

| `state` | prototype part | relational part |
|---|---|---|
| `prototype` | `LayerNorm` the whole exteroceptive block | — (none) |
| `relational` | — (none) | pass through, **no** LayerNorm |
| `mixed` | `LayerNorm` the leading centroid slice | pass through, **no** LayerNorm |

**Why.** The centroid is raw absolute hidden coordinates whose scale drifts
as the encoder keeps training online — exactly what a per-sample
`LayerNorm` (zero-mean/unit-variance + learnable affine) tames before it
reaches the value net. The relational summary is already bounded/scaled
per-feature by construction (log-score stats at O(tens), entropy in
`[0, log K]`, `tanh` reward channels in `(-1, 1)`), **and** it carries
meaningful structural zeros (a cold class's reward, low entropy) that a
`LayerNorm` would smear away — so in `relational` and `mixed` the summary is
fed to the net untouched. `ExteroceptiveNorm` reduces to a no-op when
`proto_dim == 0` and normalises the whole block when the block is entirely
prototype, so the same code path serves all three modes.

This is orthogonal to `proprio_feature_scaling`, which governs the
proprioceptive *tail* (`proprio_norm`) independently.
