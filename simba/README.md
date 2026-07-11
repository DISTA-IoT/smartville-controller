# SIMBA — Simple Intelligence Module for Basic Ablations

A from-scratch, deliberately small re-implementation of the TIGER
controller brain, built to answer ONE question cleanly:

> can a value-learning agent discover **which** CTI labels are worth
> buying — beating both a `no_epistemic` agent (never buys) and a
> `greedy_cti` agent (buys everything it can afford)?

Everything that is not needed to answer that question was removed: no
Rainbow tweaks, no Kafka/health monitoring, no node features, no Hydra.
Traffic only — flowstats windows + raw packet bytes.

```
simba/
  config.py        one dataclass; the whole game economy is documented here
  data.py          recorder-format trace loading (manifest/shards/ticks)
  synth.py         synthetic trace generator (same on-disk format)
  inference.py     IM: GRU+MLP encoder, prototypical classifier,
                   G1-supervised anomaly threshold, input-space clustering
  environment.py   budget, curriculum knowledge, CTI marketplace, rewards
  agent.py         DM: one plain DQN (eps-greedy, uniform replay, 1-step TD)
  brain.py         SimbaBrain: ties IM+DM+env; offline & POX modalities
simba_offline.py       train+eval one (mode, seed) pair on a trace
simba_experiments.py   parallel launcher over modes x seeds x GPUs (wandb: SIMBA)
```

## The game

Each tick the IM splits incoming samples into groups (one per predicted
Known class, one per cluster of unknown-looking traffic). Per group the
DM picks one of the canonical three actions:

| action | benign flow                    | malicious flow |
|--------|--------------------------------|----------------|
| 0 accept | `+r` if class Known, `+alpha*r` if unknown | `-r` |
| 1 block  | `-beta*r`                     | `0` |
| 2 buy    | like accept, **minus the CTI price** (or a waste penalty if no CTI is on sale for that cluster) | like accept |

The buy action only exists on *unknown clusters* (as in TIGER); known
groups are always accept/block. Buying on a cluster whose majority class
has no CTI on sale (a G1 pseudo zero-day) charges `waste_buy_penalty` —
the `has_quote` state flag lets the agent learn to avoid it.

`alpha < 1` is the crux: unlabeled traffic can be accepted, but only a
fraction of its value is realised (sandboxing / no SLA). Therefore:

* buying a **benign** zero-day's label converts its future flows from
  `alpha*r` to `r` — worth it iff the price is below the remaining
  volume gap;
* buying a **malicious** zero-day's label converts "blocked unknown"
  (0) into "blocked known" (0) — *never* worth it;
* `no_epistemic` forfeits the benign gap forever; `greedy_cti` wastes
  the price of every malicious/overpriced label. Only a learner that
  reads the cluster (state = stationary input-space centroid + budget /
  quote / confidence block) can take just the profitable buys.

**Discounting visibility.** For a buy to be visible to a γ-discounted
value learner the price must satisfy

```
price  <  (1 - alpha) * r * rate * γ^d / (1 - γ^d)
```

with `rate` = the class's flows/tick and `d` = DM decisions/tick. If you
swap in your own traces and the DQN "converges to zero epistemic
actions", check this inequality before blaming the agent — with the
default `gamma=0.998` and the default synthetic rates, doorlock
(rate 5, price 40) is clearly worth buying and echo (rate 1, price 400)
clearly is not.

Two more anti-"zero-epistemic-collapse" defaults, learned the hard way:

* **bankruptcy does not terminate episodes** (`bankrupt_terminates:
  false`; it stays a reported metric). With termination on, random
  exploratory buys bankrupt every episode within a few ticks, so the
  agent never lives long enough to observe a good buy's payoff — and
  never past exploration;
* **exploration is buy-averse** (`explore_buy_weight: 0.1`): a uniform
  random draw over three actions pays a CTI price every third decision,
  drowning the buy action's value signal in exploration damage.

## Inference module (kept honest, kept small)

* **encoder** GRU over the `[10,4]` flowstats window + MLP over mean raw
  packet bytes → LayerNorm'd `hidden_size` embedding;
* **classifier** prototypical — nearest class prototype, prototypes from
  the per-class buffers of *trainable* classes (Knowns + bought CTIs);
* **anomaly detection** supervised by the **G1 pseudo zero-days**, like
  ASAP: G1 classes join the prototypical episodes as their own classes
  and are repelled a margin away from Known prototypes; the unknown
  threshold `tau` is calibrated as the best Known-vs-G1 separation;
* **clustering** of unknown traffic happens in the **standardised
  input space** (not the drifting embedding), radius calibrated on G1
  within/between-class distances. The DM state centroid uses the same
  stationary representation, so cluster identities stay legible to the
  DQN across IM training.

Ground-truth labels feed per-class *shadow* buffers (the offline trace
and the GNS3 traffic_dict both provide them), but only Knowns + bought
classes are ever trained on — G1 labels are used solely as novelty
supervision, G2 labels only after purchase.

## Ablations

Same DQN, same hyperparameters, same transitions memorised — the modes
only restrict/force the epistemic action at decision time:

* `drl` — all three actions available;
* `no_epistemic` — action 2 masked out;
* `greedy_cti` — action 2 forced on every unknown cluster whose CTI is
  on sale and affordable (`budget - price >= min_budget`, i.e. bankrupt
  buys are guarded); pragmatic actions still learned.

Orthogonal to the modes, `--hard-g2s CLASS...` (config `hard_g2s`)
removes classes from the CTI market entirely — layer it on `greedy_cti`
with every not-worth-buying class listed to get the "oracle buyer"
upper-bound baseline, mirroring TIGER's `hard_g2s`.

## Offline usage

```bash
# synthetic smoke run, no wandb
python simba_offline.py ./simba_synth --synthetic --mode drl --seed 6 --no-wandb

# full comparison on pre-recorded traces, wandb project SIMBA
python simba_experiments.py pre_recorded_data/ \
    --gpus 0 2 3 4 5 6 \
    --ablation-modes drl no_epistemic greedy_cti \
    --seeds 6 1 \
    --episodes 300 --wandb-project SIMBA

# any config field can be overridden per-run (forwarded after --)
python simba_experiments.py pre_recorded_data/ --seeds 6 -- \
    --set gamma=0.997 --set unknown_accept_discount=0.25 \
    --prices-json '{"doorlock": 60, "echo": 400}'
```

`DATA` is either a single `run_*/` capture dir or a directory of them
(newest wins). Real captures skip the first (warm-up) shard
automatically; synthetic ones don't. Per-run outputs land in
`runs_simba/<mode>_seed<seed>/` (`results.json`, `dm_final.pt`,
`im_pretrained.pt`).

## GNS3 / POX usage

`tiger_server.py` instantiates SIMBA instead of the TIGER brain when the
config carries:

```yaml
intrusion_detection:
  brain: simba
simba:                 # optional SimbaConfig overrides
  gamma: 0.998
  im_snapshot_path: /pox/pox/smartController/simba_models/im_pretrained.pt
```

`SimbaBrain.from_tiger_config` maps the usual payload keys (knowledge,
rewards, prices, feature dims, device) and `SimbaBrain.process_input(flows)`
plugs into the existing `smart_check` loop unchanged. Without a
pretrained IM snapshot the module cold-starts online: everything is
unknown until the Known buffers fill and the first calibration runs.

## Reading the results

Per-episode wandb series: `train/return`, `train/final_budget`,
`train/n_buys`, `buys/<class>`, `train/cs_acc`, `train/ad_recall`,
`train/epsilon`, `train/dm_loss`, `train/im_loss`. Eval summary:
`eval_mean_return`, `eval_mean_buys`, `eval_buys/<class>`.

The intended end state: `drl` buys the cheap high-volume benign
zero-days early, never the malicious/overpriced ones — beating
`no_epistemic` by the recovered `(1-alpha)` gap and `greedy_cti` by the
avoided waste.
