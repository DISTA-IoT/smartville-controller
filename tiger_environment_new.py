import random
from collections import Counter

from smartController.cti_delivery import CTIDeliveryModel


class NewTigerEnvironment:

    def __init__(self, kwargs):
        """Initialize the attributes of the Car class."""
        self.init_budget = float(kwargs.intrusion_detection.tiger_init_budget)
        # Per-flow reward: the budget delta an accepted/blocked *flow* earns or
        # costs (benign accepted / malicious blocked add, the opposite subtract).
        self.flow_rewards_dict = {key: float(value) for key, value in kwargs.rewards.items()}
        # Per-class price of an epistemic action (buying that class's CTI/label),
        # read directly from the config `prices` dict. DECOUPLED from
        # flow_rewards_dict: the CTI price used to be |reward| * cti_price_factor,
        # which tied the label-buying cost to the flow economics; it is now an
        # independent, flat price (see _cti_price / perform_epistemic_action).
        # One entry per class -- a wasted buy or a dynamic_knowledge reshuffle can
        # target any class -- so .get() below falls back to 0 for anything absent.
        self.cti_prices_dict = {key: float(value) for key, value in kwargs.get('prices', {}).items()}
        self.min_budget = float(kwargs.intrusion_detection.min_budget)
        self.max_budget = float(kwargs.intrusion_detection.max_budget)
        self.current_budget = self.init_budget
        self.traffic_dict = kwargs.traffic_dict # No need to deepcopy if not modified
        self.init_knowledge = {k: list(v) if isinstance(v, list) else v for k, v in kwargs.knowledge.items()}
        self.logger = kwargs.logger
        self.max_episode_steps = int(kwargs.intrusion_detection.max_episode_steps)
        # Per-episode CTI-price decay MULTIPLIER, starting at 1.0 (full price).
        # The legacy intrusion_detection.cti_price_factor no longer sizes the
        # epistemic price -- that now comes straight from cti_prices_dict -- so
        # this is purely the factor the optional price_decay feature shrinks over
        # an episode. It stays 1.0 the whole episode unless price_decay is on,
        # in which case the flat price is scaled down stochastically (price_decay).
        self.current_cti_price_factor = 1.0
        self.seed = int(kwargs.intrusion_detection.seed)
        # When True, episodes no longer terminate on hitting max_budget, so an
        # episode's cumulative reward (sum_episode_rewards) is no longer pinned
        # to the +max_budget termination threshold and can actually separate a
        # better agent from a worse one. Episodes then end only on bankruptcy
        # (< min_budget) or the step horizon (>= max_episode_steps), i.e. a
        # fixed-horizon return. Defaults to False (legacy win-termination).
        self.disable_budget_win_termination = bool(
            kwargs.intrusion_detection.get('disable_budget_win_termination', False))
        # When True, episodes no longer terminate on bankruptcy (< min_budget),
        # so a run can keep exploring past a deep negative budget. Episodes then
        # end only on the step horizon (and, unless disabled, the win threshold).
        # Default False (legacy bankrupt termination).
        self.disable_budget_bankrupt_termination = bool(
            kwargs.intrusion_detection.get('disable_budget_bankrupt_termination', False))
        # Hard epistemic action: when True, a single epistemic action does the
        # work of both former levels in one shot -- buying a G2's label also
        # immediately buys its AD oracle (see _deliver_label /
        # online_anomaly_detection), so from delivery on its samples are flagged
        # Known for free without consulting the IM's neural anomaly detector.
        # Charged the same single price as an ordinary label buy. Default False
        # reproduces the plain behaviour (a label buy grants no oracle).
        self.hard_epistemic_action = bool(
            kwargs.intrusion_detection.get('hard_epistemic_action', False))
        # Stable, full set of class names across the episode: the initial
        # Knowns/G1s/G2s partition. Buying a G2 only moves it from G2s to
        # Knowns, so this union is invariant and gives every per-class wandb
        # series a fixed key for the whole run. Also invariant under
        # dynamic_knowledge, which only reshuffles the same pool of classes
        # across Knowns/G1s/G2s -- it never adds or removes a class.
        self.all_class_labels = (
            list(self.init_knowledge.get('Knowns', []))
            + list(self.init_knowledge.get('G1s', []))
            + list(self.init_knowledge.get('G2s', []))
        )
        # When True, every episode reset draws a fresh random Knowns/G1s/G2s
        # partition of self.all_class_labels instead of replaying the fixed
        # partition read from config/manifest, keeping the original group
        # sizes (e.g. 3 Knowns / 3 G1s / 7 G2s) fixed. See
        # _randomize_knowledge_partition. Default False reproduces the
        # legacy fixed-curriculum behaviour.
        self.dynamic_knowledge = bool(
            kwargs.intrusion_detection.get('dynamic_knowledge', False))
        # Dedicated RNG for the per-episode partition draw, seeded once from
        # the run's seed and advanced every episode. Kept separate from the
        # global `random` module -- which reset_intelligence reseeds to a
        # fixed self.seed every episode for price_decay's reproducibility --
        # so drawing a partition here neither depends on nor perturbs that
        # determinism.
        self._knowledge_rng = random.Random(self.seed)
        # Imperfect CTI delivery (RC 3.4): degrades/partially-captures/delays a
        # purchase's delivery. A strict no-op in the clean regime (all knobs at
        # default), so existing runs are unaffected. See cti_delivery.py.
        self.cti_delivery = CTIDeliveryModel(kwargs.intrusion_detection)


    def _randomize_knowledge_partition(self):
        """
        Draws a fresh random Knowns/G1s/G2s split of self.all_class_labels,
        keeping each group's size equal to the initial config's (e.g. 3
        Knowns / 3 G1s / 7 G2s -- whatever init_knowledge specified). Only
        called when dynamic_knowledge is enabled.
        """
        pool = list(self.all_class_labels)
        self._knowledge_rng.shuffle(pool)
        n_known = len(self.init_knowledge.get('Knowns', []))
        n_g1 = len(self.init_knowledge.get('G1s', []))
        return {
            'bening_patterns': list(self.init_knowledge.get('bening_patterns', [])),
            'attack_patterns': list(self.init_knowledge.get('attack_patterns', [])),
            'Knowns': pool[:n_known],
            'G1s': pool[n_known:n_known + n_g1],
            'G2s': pool[n_known + n_g1:],
        }


    def reset_intelligence(self):

        if self.dynamic_knowledge:
            self.current_knowledge = self._randomize_knowledge_partition()
        else:
            self.current_knowledge = {k: list(v) if isinstance(v, list) else v for k, v in self.init_knowledge.items()}
        self.current_knowledge['updated_labels'] = []
        self.update_cti_options()
        self.current_cti_price_factor = 1.0
        random.seed(self.seed)
        return {'current_knowledge': self.current_knowledge,
                'updated_label': None,
                'new_label': None,
                'reset' : True}


    def update_cti_options(self, n_options=1):
        """
        This method updates the CTI agent state vector, i.e., according to available labels to buy.
        """
        
        # set a list of n_options available cti options:
        self.current_cti_options = {}
        # A G2 whose CTI has been paid for but not yet delivered (delivery delay)
        # is no longer offered for purchase -- it must not be paid for twice --
        # even though it is still a G2 (unlearnable) until delivery. getattr
        # guards the first call, which runs before __init__ sets cti_delivery.
        pending = getattr(self, 'cti_delivery', None)
        pending = pending.pending_labels() if pending is not None else set()
        g2s = [g for g in self.current_knowledge['G2s'] if g not in pending]
        num_g2s = len(g2s)

        for idx in range(n_options):

            # do we still have so many unknowns?
            if idx < num_g2s:
                label = g2s[idx]
                self.current_cti_options[label] = self._cti_price(label)
            else:
                # if we do not have unknowns anymore, then lets put a placeholder in the state space (with high cost).
                self.current_cti_options[f'placeholder_{idx}'] = 100

        # An epistemic buy is available while an unbought G2 remains. Under
        # hard_epistemic_action a single buy grants both the label and its AD
        # oracle, so there is no separate second-level buy to keep offering.
        self.epistemic_actions_available = 1 if num_g2s > 0 else 0


    def acquired_cti_fraction(self):
        """
        Fraction of this episode's zero-day (G2) pool whose CTI has been bought
        AND delivered so far, in [0, 1]. 0 at episode start, rising toward 1 as
        G2s are purchased and promoted to Knowns. Counts delivered acquisitions
        only (acquired_g2_stats[...]['bought'], set in _deliver_label): a paid-
        but-not-yet-delivered G2 (delivery delay) is still a G2 and is not yet
        counted, matching the moment its intelligence actually becomes usable.

        This is the DM state's explicit, monotone memory of how much CTI it has
        acquired this episode. Without it the value function cannot represent
        "I bought class X, so my future return went up" -- the purchase's payoff
        is earned later, on the known-traffic path, by transitions whose state
        carries no trace of the buy -- so the delayed return is not Markov-
        attributable to the epistemic action that caused it. The denominator is
        the fixed per-episode G2 count (acquired_g2_stats is populated with the
        full initial G2 set in reset() and its entries persist after delivery),
        so the fraction is stable and comparable across ticks and episodes.

        Guarded with getattr so a call before the first reset() (which creates
        acquired_g2_stats) reads as 0.0 -- nothing acquired yet -- rather than
        raising, mirroring the cti_delivery guard in update_cti_options.
        """
        stats_by_label = getattr(self, 'acquired_g2_stats', None)
        if not stats_by_label:
            return 0.0
        delivered = sum(1 for stats in stats_by_label.values() if stats['bought'])
        return delivered / len(stats_by_label)


    def has_episode_ended(self):
        if self.steps_done >= self.max_episode_steps:
            return True
        if not self.disable_budget_bankrupt_termination \
                and self.current_budget < self.min_budget:
            return True
        if not self.disable_budget_win_termination \
                and self.current_budget > self.max_budget:
            return True
        return False


    def reset(self):
        self.logger.info('TIGER ENV: restarting episode!')
        self.episode_rewards = []
        self.episode_budgets = []
        self.epistemic_actions = 0
        self.steps_done = 0
        # Per-episode CTI-delivery bookkeeping: total real acquisitions and how
        # many of them were delivered imperfectly (any noise/partial/delay),
        # reported at episode end as the "bad-buy rate". Cleared here and the
        # delivery model's pending/active state is wiped before reset_intelligence
        # (which reads cti_delivery.pending_labels via update_cti_options).
        self.total_cti_buys = 0
        self.corrupt_cti_buys = 0
        self.cti_delivery.reset(self.seed)
        self.restart_budget()
        self.reset_intelligence()
        # Fixed-key per-G2 stats, pre-populated every episode so wandb sees
        # the same 7 series (reappearances/<label>, post-buyin reward, and
        # CTI price paid -- reported as separate wandb series) whether or not
        # that label gets bought this episode.
        # cti_shots / cti_misses split every post-buyin reappearance by the
        # IM's anomaly-detection verdict on that sample (deemed Known vs. still
        # deemed anomalous), independent of what the DM decides afterwards.
        # Per-episode scalars, reported once at episode end alongside
        # reappearances, and reappearances == cti_shots + cti_misses.
        self.acquired_g2_stats = {
            label: {'bought': False, 'price_paid': 0.0,
                    'reappearances': 0, 'reward_since_purchase': 0.0,
                    'cti_shots': 0, 'cti_misses': 0,
                    'oracle_ad': False, 'oracle_price_paid': 0.0,
                    'oracled_reappearances': 0}
            for label in self.current_knowledge['G2s']
        }
        # Labels whose AD oracle is active. Under hard_epistemic_action it is
        # granted together with the label buy (in _deliver_label): their online
        # samples are forced Known without consulting the IM for the rest of the
        # episode. Reset every episode, like the knowledge base.
        self.ad_oracle_labels = set()
        # Fixed-key per-G2 series for the pre-purchase (unsupervised) regime:
        # net reward/cost accrued this episode from *accepting* a G2 cluster
        # in act_on_unknown_clusters, before that label has been bought.
        # Reset every episode so it only ever reflects the current episode's
        # unsupervised behaviour.
        self.unsupervised_costs = {label: 0.0 for label in self.current_knowledge['G2s']}
        # Per-class episode appearance tally: how many times each class is
        # seen on the wire this episode, counted from the true labels of the
        # online traffic regardless of the DM's knowledge state -- so a G2
        # keeps accruing appearances both before and after it is bought.
        self.appearances = {label: 0 for label in self.all_class_labels}
        # Per-class, per-episode pragmatic-decision tallies, counted in
        # *samples* (group/cluster members) rather than in group/cluster
        # decisions, so their grand total is on the same footing as appearances
        # (both count wire samples, not DM decisions). Known-regime tallies are
        # keyed by the group's most-similar predicted class; unknown-regime
        # tallies by the cluster's majority true label. Epistemic buys (action
        # 2) are not pragmatic and are never recorded here, so the four tallies
        # sum to slightly under total appearances by exactly the bought-cluster
        # samples. Reported at episode end alongside appearances.
        self.known_acceptances = {label: 0 for label in self.all_class_labels}
        self.known_blocks = {label: 0 for label in self.all_class_labels}
        self.unknown_acceptances = {label: 0 for label in self.all_class_labels}
        self.unknown_blocks = {label: 0 for label in self.all_class_labels}
        # Per-class net value for the non-G2 classes (Knowns and G1s),
        # accumulated from episode init: the raw per-sample reward of their
        # accepted known-traffic samples. G2 net_values are tracked separately
        # (post-buyin) in acquired_g2_stats, so they are excluded here to
        # avoid double counting; together the two cover every class once.
        self.net_values = {
            label: 0.0 for label in self.all_class_labels
            if label not in self.current_knowledge.get('G2s', [])
        }
        # Count of action-2 purchases whose targeted (majority-vote) label
        # wasn't actually a purchasable G2 -- e.g. the cluster was a mixed/
        # spurious one whose majority label is an already-Known class or a
        # G1 true zero-day with no CTI option. These still cost full price
        # (see perform_epistemic_action) but acquire nothing.
        self.wasted_epistemic_actions = 0

    def record_cti_reappearances(self, true_label_names, predicted_zda_flags):
        """
        Called once per online tick with the true label and the IM's anomaly-
        detection verdict (predicted-zda True/False) of *every* online sample,
        regardless of how the DM later acts on it. For each already-bought G2,
        tallies how the IM now handles its post-purchase traffic on the wire:

          - cti_shots  : samples correctly deemed KNOWN (not predicted-zda) --
                         the freshly-installed prototype is catching them.
          - cti_misses : samples still deemed ANOMALY (predicted-zda) despite
                         the class having been bought -- the prototype isn't.

        reappearances is their sum: every post-buyin reencounter on the wire,
        so reappearances == cti_shots + cti_misses by construction. Counting is
        by TRUE label over the whole online batch (both the known- and anomaly-
        predicted splits), so a miss -- which flows to the unknown-cluster path
        downstream, not act_on_known_traffic -- is still counted here. All three
        are per-episode scalars, reset in reset() and reported at episode end.
        """
        for name, is_zda in zip(true_label_names, predicted_zda_flags):
            stats = self.acquired_g2_stats.get(name)
            if stats is None or not stats['bought']:
                continue
            stats['reappearances'] += 1
            # Post-oracle reencounters, a subset of reappearances.
            if name in self.ad_oracle_labels:
                stats['oracled_reappearances'] += 1
            if is_zda:
                stats['cti_misses'] += 1
            else:
                stats['cti_shots'] += 1

    def record_reappearances(self, true_label_names, per_sample_rewards, accepted):
        """
        Called once per known-traffic group from act_on_known_traffic,
        whether the DM accepted or blocked it, with the true label and
        reward of just that group's members. Handles only the *reward* side of
        the CTI-ROI tracker now (the reappearance/shot/miss counts live in
        record_cti_reappearances, counted per tick by true label so misses are
        included too). reward_since_purchase (which *is* net_values for a bought
        G2) only accumulates on accepted groups, since a blocked group never
        earns or costs the raw per-flow reward. The CTI purchase price is
        tracked separately in price_paid and is never netted against this
        reward, so net_values for a benign G2 floors at zero rather than going
        negative.

        The non-G2 classes (Knowns and G1s) instead feed the episode-wide
        net_values tally here: their accepted known-traffic reward accrues
        from episode init (Knowns are known from the start), mirroring the
        bought-G2 reward_since_purchase rule -- raw per-sample reward, accepted
        groups only -- so net_values reads uniformly across every class.
        """
        for name, reward in zip(true_label_names, per_sample_rewards):
            stats = self.acquired_g2_stats.get(name)
            if stats is not None and stats['bought']:
                if accepted:
                    stats['reward_since_purchase'] += reward
            elif accepted and name in self.net_values:
                self.net_values[name] += reward

    def record_classification_stats(self, true_label_names, pred_label_names):
        """
        Called once per online tick with the true and IM-predicted class name
        of every known-predicted sample (i.e. the same population
        act_on_known_traffic decides over). For every class observed in this
        tick -- Knowns, G1s and G2s alike, bought or not -- computes this
        tick's one-vs-all recall and precision from its reencounters in this
        tick only, and returns them in a flat dict keyed by metric name -- no
        per-episode accumulation, every tick is its own data point. A label is
        omitted from the returned dict for a given metric when this tick has
        no occurrences to compute it from: zero true instances of the label
        (no recall sample) or zero predictions of it (no precision sample) --
        skipped rather than counted as 0 or 1.
        """
        tick_metrics = {}
        true_counts = Counter(true_label_names)
        pred_counts = Counter(pred_label_names)
        # True positives per class: samples whose true and predicted names
        # agree. One-vs-all recall = tp / true_count, precision = tp / pred_count.
        tp_counts = Counter(
            true_name
            for true_name, pred_name in zip(true_label_names, pred_label_names)
            if true_name == pred_name)
        for label in set(true_counts) | set(pred_counts):
            tp = tp_counts.get(label, 0)
            if true_counts.get(label, 0) > 0:
                tick_metrics[f'classification_recall/{label}'] = tp / true_counts[label]
            if pred_counts.get(label, 0) > 0:
                tick_metrics[f'classification_precision/{label}'] = tp / pred_counts[label]
        return tick_metrics

    def record_appearances(self, true_label_names):
        """
        Called once per online tick with the true class name of every online
        sample, tallying how many times each class is seen on the wire this
        episode. Counts every class -- Knowns, G1s and G2s alike -- regardless
        of the DM's knowledge state, so a G2 keeps accruing appearances both
        before and after it is bought. A pure observation count: the DM's
        accept/block decisions never enter it.
        """
        for name in true_label_names:
            if name in self.appearances:
                self.appearances[name] += 1

    def record_pragmatic_decision(self, regime, label, accepted, count):
        """
        Accumulate a single pragmatic (accept/block) DM decision into the
        per-episode, per-class sample tallies. `count` is the number of
        samples the decision covered (group/cluster members), so the tallies
        aggregate wire samples exactly like appearances -- one group-level
        decision over N members adds N, not 1. `regime` is 'known' (keyed by
        the group's most-similar predicted class) or 'unknown' (keyed by the
        cluster's majority true label); `accepted` selects the accept vs block
        tally. Epistemic buys are not pragmatic and must not be passed here.
        """
        target = {
            ('known', True): self.known_acceptances,
            ('known', False): self.known_blocks,
            ('unknown', True): self.unknown_acceptances,
            ('unknown', False): self.unknown_blocks,
        }.get((regime, bool(accepted)))
        if target is not None and label in target:
            target[label] += count

    def record_unsupervised_pass(self, true_label_names, per_sample_rewards):
        """
        Called from act_on_unknown_clusters for the members of a cluster the
        DM just accepted (let pass), before any CTI was bought for that
        label. `per_sample_rewards` already has the same accept-side scaling
        applied as `_decision_reward` -- positive (benign) part scaled by
        `unknown_accept_reward_scale`, negative (malicious) part scaled by
        `unknown_malicious_accept_penalty_scale` -- so this tally matches the
        actual budget impact of accepting that traffic, not the raw label
        reward. Accumulates into this episode's unsupervised_costs tally.
        Skips labels already bought this episode, since those are tracked by
        record_reappearances/acquired_g2_stats instead.
        """
        for name, reward in zip(true_label_names, per_sample_rewards):
            if name not in self.unsupervised_costs:
                continue
            if self.acquired_g2_stats[name]['bought']:
                continue
            self.unsupervised_costs[name] += reward

    def _cti_price(self, label):
        """
        Flat price of an epistemic action buying CTI for `label`, read from the
        config `prices` dict (self.cti_prices_dict) -- DECOUPLED from the flow
        rewards (self.flow_rewards_dict), which now only price accepted/blocked
        flows. Scaled by current_cti_price_factor, the per-episode decay
        multiplier (1.0 unless price_decay is enabled), so the optional
        stochastic price-decay feature still applies on top of the flat price.
        A label with no configured price (a state-space option placeholder, or a
        wasted buy whose target class is absent from `prices`) costs 0.
        """
        return self.cti_prices_dict.get(label, 0.0) * self.current_cti_price_factor

    def price_decay(self):
        self.current_cti_price_factor *= max(0.01, min(0.99, random.gauss(0.7,0.4)))


    def perform_epistemic_action(self, target_label=None, current_step=0,
                                 purity=None, confidence=None):
        """
        Buys CTI for `target_label` -- the majority true label among the
        observed cluster's members, computed by the caller -- turning it
        into a Known class, *if* it is currently a purchasable G2. Only
        called when `epistemic_actions_available == 1` (the DM's action-2
        is masked to a block otherwise), so there's always *some* real G2
        to buy, but not necessarily the one this particular cluster's
        majority vote points at.

        If `target_label` is None, falls back to the sole entry in
        `current_cti_options` (the previous, cluster-agnostic behavior) --
        kept only as a defensive default, since the one call site always
        supplies a majority-vote label now.

        If `target_label` is not a purchasable G2 (the cluster turned out
        to be a mixed/spurious one whose majority label is an already-Known
        class, or a G1 true zero-day for which no CTI is ever offered), the
        purchase is wasted: full price is still paid -- computed from the
        cluster's actual majority label, not from whatever G2 happened to be
        on offer -- but no class moves between G1s/G2s/Knowns, since there
        was nothing real to acquire. This is intentional: it's the cost
        structure a state-conditioned policy can learn to avoid (by reading
        the cluster's centroid/confidence) and a blind scripted policy
        (greedy/periodic CTI) cannot.

        Under hard_epistemic_action a G2's AD oracle is granted together with its
        (level-1) label buy in _deliver_label, so there is no separate buy here --
        a repeat buy on an already-Known class is wasted like any other.

        `current_step`, `purity` and `confidence` parameterise imperfect CTI
        delivery (RC 3.4). `current_step` (the DM step of this buy) times any
        delivery delay; `purity` (the fraction of the cluster's members that are
        the majority class) and `confidence` (the cluster's anomaly-confidence)
        are the quality signals the delivery model couples corruption severity
        to. They only take effect for a real (non-wasted) acquisition, and only
        when the CTI-delivery knobs are set; otherwise the
        purchase is delivered clean and instantly, as before.
        """
        g2s = self.current_knowledge['G2s']
        # G2s whose CTI has been paid for but not yet delivered (delivery delay).
        # They are still in current_knowledge['G2s'] -- an unlearnable, still-
        # costly zero-day until delivery -- so the `not in g2s` guard below would
        # let them be re-purchased. A re-buy would append a second pending entry
        # for the same label, and _deliver_label would then run G2s.remove(label)
        # twice, the second raising ValueError. Treat a still-pending label as
        # non-acquirable (a wasted buy), honouring "it must not be paid for twice".
        pending = self.cti_delivery.pending_labels()

        if target_label is None:
            target_label = list(self.current_cti_options.keys())[0]

        if target_label not in g2s or target_label in pending:
            # The target is not a purchasable G2 (an already-Known class -- e.g. a
            # G2 already bought -- a G2 already paid for but still awaiting delayed
            # delivery, or a G1 for which no CTI is ever offered), so the buy is
            # wasted: full price is paid but nothing is acquired. Under
            # hard_epistemic_action a G2's AD oracle is already granted at its
            # (level-1) label buy in _deliver_label, so a repeat buy on it is
            # wasted like any other -- there is no separate second-level action.
            self.wasted_epistemic_actions += 1
            price_payed = self._cti_price(target_label)
            return {'updated_label': None,
                    'current_knowledge': self.current_knowledge,
                    'price_payed': price_payed,
                    'wasted': True}

        acquired_cti = target_label

        # Payment is immediate. Record the price and count the acquisition now,
        # at the moment of purchase, regardless of any delivery delay.
        self.epistemic_actions += 1
        price_payed = self._cti_price(acquired_cti)
        stats = self.acquired_g2_stats[acquired_cti]
        stats['price_paid'] = price_payed
        self.total_cti_buys += 1

        # Route the purchase through the CTI delivery model (RC 3.4). It computes
        # this buy's effective corruption (label-noise / partial-capture, both
        # optionally coupled to the cluster's purity/confidence) and its delivery
        # delay. All no-ops in the clean regime, giving the legacy instant path.
        delay, noise, capture = self.cti_delivery.on_purchase(
            acquired_cti, current_step, purity=purity, confidence=confidence)
        if noise > 0.0 or capture < 1.0 or delay > 0:
            self.corrupt_cti_buys += 1

        if delay > 0:
            # Deferred delivery: the class stays a G2 -- still an unlearnable,
            # still-costly zero-day (its buffer stays training-skipped and its
            # samples keep counting as ground-truth anomalies) -- until
            # deliver_pending_cti() promotes it `delay` steps from now. It is
            # removed from the buyable set via cti_delivery.pending_labels() in
            # update_cti_options so it cannot be paid for twice.
            self.update_cti_options()
            return {'updated_label': None,
                    'current_knowledge': self.current_knowledge,
                    'price_payed': price_payed,
                    'wasted': False,
                    'scheduled': True}

        # Instant delivery (legacy behaviour): promote the class in this tick.
        self._deliver_label(acquired_cti)
        return {'updated_label': acquired_cti,
                'current_knowledge': self.current_knowledge,
                'price_payed': price_payed,
                'wasted': False}

    def _deliver_label(self, label):
        """
        Promote a purchased G2 to a Known class -- the moment its CTI is actually
        delivered (instantly, or after a delay via deliver_pending_cti). From
        here its replay buffer becomes training-eligible (no longer skipped as a
        G2 in sample_from_replay_buffers) and its samples stop counting as
        ground-truth zero-days. Any label-noise / partial-capture registered for
        this label at purchase time (in cti_delivery) now begins to bite on its
        training data. Marks `bought` (which gates all post-buyin ROI tracking).
        Under hard_epistemic_action the same buy also grants this class's AD
        oracle here, in one shot -- no extra charge, it rides on the label buy.
        """
        self.current_knowledge['G2s'].remove(label)
        self.current_knowledge['Knowns'].append(label)
        self.current_knowledge['updated_labels'].append(label)
        self.acquired_g2_stats[label]['bought'] = True
        # Hard epistemic action: buying the label also buys its AD oracle. From
        # now on this class's online samples are forced Known without consulting
        # the IM's neural anomaly detector (see online_anomaly_detection).
        if self.hard_epistemic_action:
            self.ad_oracle_labels.add(label)
            self.acquired_g2_stats[label]['oracle_ad'] = True
        self.update_cti_options()

    def deliver_pending_cti(self, current_step):
        """
        Promote every purchased G2 whose (delayed) CTI is due by `current_step`
        to a Known class, and return the list of newly-delivered labels so the
        brain can register them with the label encoder and open their replay
        buffers. Called once per tick from online_inference; a no-op returning []
        in the instant-delivery regime.
        """
        ready = self.cti_delivery.pop_ready(current_step)
        for label in ready:
            self._deliver_label(label)
        return ready


    def restart_budget(self):

        # now we can restart the budget
        self.current_budget = self.init_budget


