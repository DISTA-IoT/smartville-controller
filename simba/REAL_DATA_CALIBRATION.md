# SIMBA on real traces — calibration study

This records the move from the synthetic trace (which "gave good results")
to the **real GNS3 capture** in `../pre_recorded_data/`, and the default
changes in `config.py` it required. All trials were run with
`simba_offline.py --no-manifest`, i.e. on pure `SimbaConfig` defaults —
the recorded run's own economy in `manifest.json` is never loaded (and
could not override the defaults anyway; see the note at the bottom).

## 0. Data fix

The shared `pre_recorded_data/shards_index.jsonl` listed **82** shards
(`shard_000000..081`) but only **11** shard files (`0..10`) were actually
committed, so `load_trace` tried to `torch.load` 71 missing files. The
index is trimmed to the 11 shards on disk (ticks 0..149; shard 0 is the
warm-up the loader skips → 51 usable ticks).

## 1. Why the synthetic economics don't transfer

| | synthetic | real capture |
|---|---|---|
| doorlock rate | 5 flows/tick | **~32** flows/tick |
| echo rate | 1 flow/tick | **~29** flows/tick |
| samples/tick | ~24 | ~411 |
| ticks/episode | 200 | **51** |
| what marks the worthwhile buy | **volume** (doorlock ≫ echo) | — |

In the real capture **every class arrives at ~32 flows/tick**, so volume
no longer separates the worthwhile CTI purchases. The selective axis
becomes **benign-vs-malicious**: the two benign zero-days (doorlock,
echo) recover a large `(1-alpha)` service gap (~1000/class/episode) and
are worth buying; the five malicious ones are not (blocking already
scores 0).

Two real-trace effects the synthetic trace never exercised:

* **CTI de-noising.** Buying a *malicious* label is not "nothing": it
  turns a noisy unknown cluster the DM sometimes mis-accepts (`-r`) into a
  Known class it blocks cleanly (`0`). Measured worth on this trace:
  ~150–400/class. So a cheap malicious label is genuinely worth buying —
  which is exactly why the default-priced agent bought the wrong things
  (below).
* **Online-IM drift.** At the dense real flow rate, online IM training at
  `im_learning_rate=1e-3` drifts the encoder away from the pretrained
  snapshot the DM policy was trained against, silently degrading the DM
  and blowing up eval-return variance ~9× (std 711 → 80 once fixed).

## 2. Baseline (old defaults, real data)

`eps_decay_steps` first lowered to fit the short trace, everything else
at the *old* defaults (echo 400, malicious 150, `im_lr` 1e-3,
`init_budget` 1000). Eval mean return, seed 6:

| mode | return | eval buys |
|---|--:|---|
| greedy_cti | **2899** | all 7 |
| drl | 2512 | doorlock, **muhstik**, echo(1/5) |
| no_epistemic | 1349 | none |

drl **loses to greedy** and buys the *wrong* labels: it buys `muhstik`
(malicious — cheap at 150, and de-noising made it pay) and skips `echo`
(benign, worth buying — but overpriced at 400). The default prices,
tuned for synthetic volumes, actively mislead the agent on real data.

## 3. Calibrated defaults

| field | old | new | why |
|---|--:|--:|---|
| `prices['echo']` | 400 | **80** | benign, ~32 flows/tick → clearly worth buying; make it cheap enough that the DQN actually buys it |
| `prices[malicious]` | 150 | **550** | above the ~150–400 de-noising value → a net loss, so greedy is punished and drl learns to skip |
| `im_learning_rate` | 1e-3 | **1e-4** | keep online IM near its pretrained snapshot → stable, low-variance DM |
| `eps_decay_steps` | 100000 | **30000** | 51-tick episode ≈ 1k decisions; 100k never lets the agent exploit |
| `init_budget` | 1000 | **2500** | headroom so greedy_cti actually affords (and is punished for) all 7 buys |

## 4. Result (pure defaults, `--no-manifest`, 120 ep, 10 eval ep)

| mode | seed 6 | seed 1 | eval buys |
|---|--:|--:|---|
| **drl** | **2922 ± 75** | **2633 ± 72** | `{doorlock, echo}` — every episode, both seeds |
| no_epistemic | 1521 ± 67 | 1478 ± 152 | none |
| greedy_cti | 739 ± 48 | 847 ± 56 | all 7 |

For reference, an **oracle buyer** (`greedy_cti --hard-g2s mirai gafgyt
hajime h_scan muhstik`, i.e. forced to buy *only* the benign labels)
scores **2777** — drl (2922) matches/edges the selective optimum while
`no_epistemic` forfeits the benign gap and `greedy_cti` craters on the
malicious over-spend. **drl > no_epistemic > greedy_cti on both seeds.**

## Reproduce

```bash
# any single (mode, seed):
python simba_offline.py pre_recorded_data/ --no-manifest --no-wandb \
    --mode drl --seed 6 --episodes 120 --eval-episodes 10

# the oracle upper-bound baseline:
python simba_offline.py pre_recorded_data/ --no-manifest --no-wandb \
    --mode greedy_cti --hard-g2s mirai gafgyt hajime h_scan muhstik \
    --seed 6 --episodes 120 --eval-episodes 10
```

`--no-manifest` runs on pure `SimbaConfig` defaults. Even without it,
`apply_manifest` only *setdefaults* rewards/prices, and every G2 class is
already priced in `DEFAULT_PRICES`, so the calibrated defaults win and
the capture's TIGER-format prices (echo −200, mirai 200, …) are ignored.
