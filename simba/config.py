"""SIMBA configuration.

One flat dataclass, no Hydra. Everything the offline runner and the POX
integration need lives here. `SimbaConfig.from_dict` accepts a plain dict
(e.g. parsed from the JSON payload the tiger dashboard POSTs to the
controller) and ignores unknown keys, so the same config can travel over
HTTP like the rest of the system's configuration.

-----------------------------------------------------------------------
The economics (why DRL is necessary)
-----------------------------------------------------------------------
Every decision step the DM looks at ONE traffic group (either a group of
flows the IM classified as a Known class, or one cluster of
unknown-looking flows) and picks one of the canonical three actions:

    0 = accept   1 = block   2 = buy CTI (epistemic; flows also accepted)

Per-flow reward of the group, by TRUE class role:

    accept  benign     +r          if its class is Known to the IM
                       +alpha * r  if its class is unknown  (alpha < 1)
    accept  malicious  -r                          (full damage, always)
    block   benign     -beta * r                   (lost service)
    block   malicious   0                          (no damage, no gain)
    buy                same as accept for the flows, MINUS the CTI price
                       (or `waste_buy_penalty` if no CTI exists for the
                       cluster: G1 pseudo zero-days / already-Known)

`alpha` is the crux: unverified traffic can be let through, but only a
fraction of its service value is realised (sandboxing / rate limiting /
no SLA). Buying the label of a *benign* zero-day converts every one of
its future flows from `alpha*r` to `r`. Buying the label of a
*malicious* zero-day converts "blocked unknown" (0) into "blocked known"
(0): it buys nothing. Hence:

  * no_epistemic  loses (1-alpha)*r*volume on every benign zero-day,
                  forever;
  * greedy_cti    recovers that, but also pays the price of every
                  malicious / low-volume / overpriced label — pure loss;
  * DQN           must learn to buy ONLY the labels whose price is below
                  the discounted recoverable value: some, not all.

For a buy to be visible to a discounted value learner, the price must
satisfy   price < (1-alpha)*r*rate * gamma^d / (1 - gamma^d)
where `rate` is the class's flows/tick and `d` the number of DM
decisions per tick. Keep that inequality in mind when adapting prices to
a new trace (this is exactly the trap where a correctly-implemented DQN
"converges to zero epistemic actions": the label IS worth its price
undiscounted, but not under gamma).

-----------------------------------------------------------------------
Real traces (pre_recorded_data/) — what the defaults are calibrated to
-----------------------------------------------------------------------
The synthetic trace separated the worthwhile buys by VOLUME (doorlock
5 flows/tick vs echo 1). The real GNS3 capture does not: every class
arrives at ~32 flows/tick, so volume is uninformative and the selective
axis becomes benign-vs-malicious. Two real-trace effects the synthetic
economics did not model, and how the defaults answer them:

  * malicious buys are NOT quite "nothing". Buying a malicious label
    turns a noisy unknown cluster (which the DM sometimes mis-accepts for
    -r) into a Known class it blocks cleanly (0) — a "de-noising" gain
    worth ~150-400/class here. So malicious prices are set (550) safely
    ABOVE that, keeping them a net loss for a greedy buyer;
  * the online IM must stay near its pretrained snapshot. At the dense
    real flow rate a large `im_learning_rate` drifts the encoder away
    from the representation the DM policy was trained against, degrading
    and destabilising the DM at eval — hence the reduced default.

With these defaults, `--no-manifest` offline runs on pre_recorded_data/
rank drl > no_epistemic > greedy_cti (drl buys exactly {doorlock, echo}
and skips all malicious). NB: `apply_manifest` deliberately does NOT
override these — a recorded run's own rewards/prices are adopted only as
setdefaults, so the calibrated defaults here win unless a trial passes
explicit overrides.
"""

from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional


# Calibrated on the pre_recorded_data/ real capture (see the "real traces"
# note in this module's docstring). Unlike the synthetic trace — where the
# doorlock:echo VOLUME ratio (5:1) did the separating — every class in the
# real capture arrives at ~32 flows/tick, so volume no longer distinguishes
# the worthwhile buys. The selective axis is therefore benign-vs-malicious,
# and the prices are set so that:
#   * benign zero-days (doorlock, echo) recover a large (1-alpha) service
#     gap (~1000/class over an episode) and are priced well below it -> a
#     value-learning DQN buys both, promptly;
#   * malicious zero-days only offer classifier "de-noising" value (buying a
#     label turns a noisy unknown cluster the DM might mis-accept into a
#     known class it blocks cleanly). On the real trace that is worth only
#     ~150-400/class, so pricing them at 550 makes buying them a net loss --
#     `greedy_cti` (buys everything affordable) is punished, `drl` skips them.
DEFAULT_PRICES = {
    'doorlock': 40.0,
    'echo': 80.0,
    'mirai': 550.0,
    'gafgyt': 550.0,
    'hajime': 550.0,
    'h_scan': 550.0,
    'muhstik': 550.0,
}


@dataclass
class SimbaConfig:
    # ---------------------------------------------------------------- dims
    flow_feat_dim: int = 4
    flows_per_sample: int = 10
    packet_feat_dim: int = 64
    packets_per_sample: int = 1
    use_packet_feats: bool = True
    hidden_size: int = 64
    device: str = 'cpu'
    seed: int = 777

    # ------------------------------------------------- inference module (IM)
    # Online (in-episode) IM step size. Kept an order of magnitude below the
    # pretraining rate: on the dense real trace (~410 flows/tick) a 1e-3
    # online rate lets the encoder DRIFT away from the pretrained snapshot the
    # DM policy was trained against, which silently degrades and destabilises
    # the DM at eval (return variance blew up ~9x). 1e-4 keeps the IM stable
    # while still assimilating purchased labels via the buy_train_burst.
    im_learning_rate: float = 1e-4
    im_train_steps_per_tick: int = 1
    im_support_size: int = 8       # support samples per class per episode step
    im_query_size: int = 8         # query samples per class per training step
    buffer_capacity: int = 512     # per-class shadow buffer capacity
    min_samples_per_class: int = 16  # a class participates in training/prototypes above this
    proto_samples: int = 64        # samples used to (re)compute each prototype
    ad_quantile: float = 0.95      # fallback tau calibration quantile (no G1 data yet)
    ad_margin: float = 1.5         # fallback tau multiplier: unknown iff dist > tau
    # The AD is *supervised by the G1 pseudo zero-days*, like ASAP: G1
    # classes join the prototypical episodes as their own classes (so
    # distinct unknowns stay distinct for clustering) and are additionally
    # pushed at least `ad_repulsion_margin` away from every Known
    # prototype; tau is then calibrated as the best Known-vs-G1 threshold.
    ad_repulsion_weight: float = 1.0
    ad_repulsion_margin: float = 4.0
    cluster_radius_factor: float = 1.0  # unknown clustering radius scale
    calibrate_every_ticks: int = 10     # tau/prototype recalibration cadence
    # Extra IM gradient steps right after a CTI purchase (the "retrain on
    # delivered labels" burst): without it the freshly bought class sits
    # in the classifier untrained, transiently degrading accuracy — a
    # hidden, unintended cost of every buy.
    buy_train_burst: int = 10
    pretrain_steps: int = 1000
    pretrain_learning_rate: float = 1e-3

    # ------------------------------------------------------ environment
    # Enough headroom that a greedy_cti buyer can actually afford every CTI
    # label it wants (7 buys, up to ~2.9k on the real prices) and so genuinely
    # demonstrates the over-spend, rather than being silently rescued by the
    # affordability guard.
    init_budget: float = 2500.0
    min_budget: float = 0.0        # bankruptcy line (see bankrupt_terminates)
    # If True, an episode ends the moment the budget drops below
    # min_budget. Default False: the episode runs its full length and
    # "bankrupt" is only a reported metric. Termination looks natural but
    # poisons learning: under epsilon-greedy exploration random CTI buys
    # bankrupt the episode within a few ticks, so the agent never lives
    # long enough to observe a good buy's payoff (nor to decay epsilon) —
    # and Q(buy) collapses to "toxic" everywhere. Same rationale as
    # TIGER's disable_budget_bankrupt_termination.
    bankrupt_terminates: bool = False
    max_episode_ticks: int = 0     # 0 = one full pass over the trace
    unknown_accept_discount: float = 0.25   # alpha
    block_benign_scale: float = 1.0         # beta
    waste_buy_penalty: float = 25.0  # buying for a cluster with no CTI on sale
    default_reward: float = 1.0    # |per-flow reward| when a class has no entry in `rewards`
    default_price: float = 150.0   # CTI price when a class has no entry in `prices`
    # Blocklist of G2 classes whose CTI is never on sale (TIGER's
    # `hard_g2s`): layered on any ablation. With it, greedy_cti becomes
    # the "oracle buyer" baseline — put every not-worth-buying class here
    # and greedy buys only the worthwhile ones.
    hard_g2s: List[str] = field(default_factory=list)
    rewards: Dict[str, float] = field(default_factory=dict)  # per-class per-flow reward override
    prices: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRICES))

    # -------------------------------------------------- decision module (DM)
    ablation: str = 'drl'  # 'drl'|'no_epistemic'|'greedy_cti'|'fixed_threshold_cti'
    # fixed_threshold_cti ablation only -- the reviewer's "buy CTI whenever
    # uncertain" threshold policy (mirrors TIGER's intrusion_detection.
    # fixed_threshold_cti / cti_confidence_threshold). On an unknown cluster the
    # policy FORCES a CTI buy whenever the IM's anomaly score for that cluster --
    # a = mean(nearest-prototype distance)/tau, i.e. the un-clamped form of the
    # `dist_ratio` confidence channel the DM already sees in its state -- exceeds
    # this value (and the label is on sale and affordable), otherwise it defers
    # the accept/block choice to the learned DQN. Higher a == the IM is more
    # confident the cluster is nothing it knows == less confident in any
    # known-class assignment == "more uncertain". Because a > 1 for every unknown
    # cluster (each member is beyond the novelty threshold tau), a threshold <= 1
    # buys every unknown cluster (degenerating to greedy_cti) while larger
    # thresholds buy only the progressively more-anomalous clusters. It never
    # chooses WHICH label is worth buying (only how novel the cluster looks), so
    # unlike the value-learning DM it cannot separate the worthwhile benign
    # zero-days from the not-worth-buying malicious ones -- see
    # UNCERTAINTY_THRESHOLD_ABLATION.md.
    cti_confidence_threshold: float = 1.5
    gamma: float = 0.998
    dm_learning_rate: float = 5e-4
    dm_hidden: int = 128
    replay_capacity: int = 100_000
    replay_batch_size: int = 64
    target_update_freq: int = 500  # hard target-net update, in gradient steps
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    # Linear decay horizon, in decision steps. The real trace is only ~51
    # ticks (~1k decisions) per episode, vs the 200-tick synthetic one, so
    # 100k steps would keep the agent exploring for ~100 episodes before it
    # ever exploits. 30k floors epsilon in ~30 episodes, leaving the rest of a
    # normal (>=100 episode) run to exploit.
    eps_decay_steps: int = 30_000
    # Probability mass the RANDOM (exploratory) draw puts on the buy
    # action; the rest is split evenly between accept/block. A uniform
    # 1/3 makes exploration itself ruinously expensive (every third
    # random decision pays a CTI price), drowning the buy action's true
    # value in exploration damage.
    explore_buy_weight: float = 0.1
    reward_scale: float = 0.02     # rewards are scaled by this before TD learning
    learn_start: int = 1_000       # decision steps before DM gradient updates begin

    # ------------------------------------------------------ state scaling
    budget_scale: float = 1000.0
    price_scale: float = 200.0
    group_size_scale: float = 20.0

    # -------------------------------------------------------- curriculum
    knowns: List[str] = field(default_factory=lambda: ['hue', 'hakai', 'torii'])
    g1s: List[str] = field(default_factory=lambda: ['okiru', 'cc_heartbeat', 'generic_ddos'])
    g2s: List[str] = field(default_factory=lambda: [
        'doorlock', 'echo', 'mirai', 'gafgyt', 'hajime', 'h_scan', 'muhstik'])
    benign_classes: List[str] = field(default_factory=lambda: ['echo', 'doorlock', 'hue'])
    malicious_classes: List[str] = field(default_factory=lambda: [
        'hakai', 'mirai', 'hajime', 'gafgyt', 'muhstik', 'h_scan',
        'okiru', 'generic_ddos', 'torii', 'cc_heartbeat'])

    # ---------------------------------------------------------- logging
    log_every_ticks: int = 25      # wandb `running/*` flush cadence, in ticks

    # ----------------------------------------------------------- persistence
    im_snapshot_path: str = ''     # optional pretrained IM weights (GNS3 mode)
    dm_checkpoint_path: str = ''   # optional pretrained DM weights

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> 'SimbaConfig':
        d = d or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def apply_manifest(self, manifest: dict) -> None:
        """Adopt curriculum/roles/dims recorded in a trace's manifest.json.

        Explicit rewards/prices in the manifest are used as defaults but
        anything already set in this config wins.
        """
        know = manifest.get('knowledge', {})
        if know:
            self.knowns = list(know.get('Knowns', self.knowns))
            self.g1s = list(know.get('G1s', self.g1s))
            self.g2s = list(know.get('G2s', self.g2s))
            benign = know.get('benign_patterns', know.get('bening_patterns'))
            if benign:
                self.benign_classes = list(benign)
            attack = know.get('attack_patterns')
            if attack:
                self.malicious_classes = list(attack)
        if 'flow_feat_dim' in manifest:
            self.flow_feat_dim = int(manifest['flow_feat_dim'])
        if 'packet_feat_dim' in manifest:
            self.packet_feat_dim = int(manifest['packet_feat_dim'])
        if 'use_packet_feats' in manifest:
            self.use_packet_feats = bool(manifest['use_packet_feats'])
        det = manifest.get('intrusion_detection', {})
        if 'flows_per_sample' in det:
            self.flows_per_sample = int(det['flows_per_sample'])
        if 'packets_per_sample' in det:
            self.packets_per_sample = int(det['packets_per_sample'])

        rewards = _as_flat_dict(manifest.get('rewards'))
        for k, v in rewards.items():
            self.rewards.setdefault(k, float(v))
        prices = _as_flat_dict(manifest.get('prices'))
        for k, v in prices.items():
            self.prices.setdefault(k, float(v))

    # ------------------------------------------------------------ helpers
    def all_classes(self) -> List[str]:
        return list(self.knowns) + list(self.g1s) + list(self.g2s)

    def is_benign(self, label: str) -> bool:
        return label in self.benign_classes

    def reward_of(self, label: str) -> float:
        """Signed per-flow reward: +|r| benign, -|r| malicious."""
        r = abs(float(self.rewards.get(label, self.default_reward)))
        return r if self.is_benign(label) else -r

    def price_of(self, label: str) -> float:
        return float(self.prices.get(label, self.default_price))


def _as_flat_dict(obj) -> Dict[str, float]:
    """Rewards/prices may arrive as a dict or as a YAML-style list of
    single-key dicts; normalise to a flat dict."""
    if not obj:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    flat = {}
    for entry in obj:
        if isinstance(entry, dict):
            flat.update(entry)
    return flat
