import random
from collections import Counter


class NewTigerEnvironment:

    def __init__(self, kwargs):
        """Initialize the attributes of the Car class."""
        self.init_budget = float(kwargs.intrusion_detection.tiger_init_budget)
        self.flow_rewards_dict = {key: float(value) for key, value in kwargs.rewards.items()}
        self.min_budget = float(kwargs.intrusion_detection.min_budget)
        self.max_budget = float(kwargs.intrusion_detection.max_budget)
        self.current_budget = self.init_budget
        self.traffic_dict = kwargs.traffic_dict # No need to deepcopy if not modified
        self.init_knowledge = {k: list(v) if isinstance(v, list) else v for k, v in kwargs.knowledge.items()}
        self.logger = kwargs.logger
        self.max_episode_steps = int(kwargs.intrusion_detection.max_episode_steps)
        self.init_cti_price_factor = float(kwargs.intrusion_detection.cti_price_factor)
        self.current_cti_price_factor = self.init_cti_price_factor
        self.seed = int(kwargs.intrusion_detection.seed)
        # When True, episodes no longer terminate on hitting max_budget, so an
        # episode's cumulative reward (sum_episode_rewards) is no longer pinned
        # to the +max_budget termination threshold and can actually separate a
        # better agent from a worse one. Episodes then end only on bankruptcy
        # (< min_budget) or the step horizon (>= max_episode_steps), i.e. a
        # fixed-horizon return. Defaults to False (legacy win-termination).
        self.disable_budget_win_termination = bool(
            kwargs.intrusion_detection.get('disable_budget_win_termination', False))
        # Stable, full set of class names across the episode: the initial
        # Knowns/G1s/G2s partition. Buying a G2 only moves it from G2s to
        # Knowns, so this union is invariant and gives every per-class wandb
        # series a fixed key for the whole run.
        self.all_class_labels = (
            list(self.init_knowledge.get('Knowns', []))
            + list(self.init_knowledge.get('G1s', []))
            + list(self.init_knowledge.get('G2s', []))
        )


    def reset_intelligence(self):
        
        self.current_knowledge = {k: list(v) if isinstance(v, list) else v for k, v in self.init_knowledge.items()}
        self.current_knowledge['updated_labels'] = []
        self.update_cti_options()
        self.current_cti_price_factor = self.init_cti_price_factor
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
        g2s = self.current_knowledge['G2s']
        num_g2s = len(g2s)

        for idx in range(n_options):

            # do we still have so many unknowns?
            if idx < num_g2s:
                label = g2s[idx]
                self.current_cti_options[label] = abs(self.flow_rewards_dict[label] * self.current_cti_price_factor)
                self.epistemic_actions_available = 1
            else:
                # if we do not have unknowns anymore, then lets put a placeholder in the state space (with high cost).
                self.current_cti_options[f'placeholder_{idx}'] = 100 
                self.epistemic_actions_available = 0


    def has_episode_ended(self):
        if self.current_budget < self.min_budget \
                or self.steps_done >= self.max_episode_steps:
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
                    'cti_shots': 0, 'cti_misses': 0}
            for label in self.init_knowledge['G2s']
        }
        # Fixed-key per-G2 series for the pre-purchase (unsupervised) regime:
        # net reward/cost accrued this episode from *accepting* a G2 cluster
        # in act_on_unknown_clusters, before that label has been bought.
        # Reset every episode so it only ever reflects the current episode's
        # unsupervised behaviour.
        self.unsupervised_costs = {label: 0.0 for label in self.init_knowledge['G2s']}
        # Per-class episode appearance tally: how many times each class is
        # seen on the wire this episode, counted from the true labels of the
        # online traffic regardless of the DM's knowledge state -- so a G2
        # keeps accruing appearances both before and after it is bought.
        self.appearances = {label: 0 for label in self.all_class_labels}
        # Per-class net value for the non-G2 classes (Knowns and G1s),
        # accumulated from episode init: the raw per-sample reward of their
        # accepted known-traffic samples. G2 net_values are tracked separately
        # (post-buyin) in acquired_g2_stats, so they are excluded here to
        # avoid double counting; together the two cover every class once.
        self.net_values = {
            label: 0.0 for label in self.all_class_labels
            if label not in self.init_knowledge.get('G2s', [])
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

    def price_decay(self):
        self.current_cti_price_factor *= max(0.01, min(0.99, random.gauss(0.7,0.4)))


    def perform_epistemic_action(self, target_label=None):
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
        """
        g2s = self.current_knowledge['G2s']

        if target_label is None:
            target_label = list(self.current_cti_options.keys())[0]

        if target_label not in g2s:
            self.wasted_epistemic_actions += 1
            price_payed = abs(self.flow_rewards_dict.get(target_label, 0.0) * self.current_cti_price_factor)
            return {'updated_label': None,
                    'current_knowledge': self.current_knowledge,
                    'price_payed': price_payed,
                    'wasted': True}

        acquired_cti = target_label

        self.epistemic_actions += 1
        self.current_knowledge['G2s'].remove(acquired_cti)
        self.current_knowledge['Knowns'].append(acquired_cti)
        self.current_knowledge['updated_labels'].append(acquired_cti)
        price_payed = abs(self.flow_rewards_dict[acquired_cti] * self.current_cti_price_factor)
        self.update_cti_options()

        stats = self.acquired_g2_stats[acquired_cti]
        stats['bought'] = True
        stats['price_paid'] = price_payed

        return {'updated_label': acquired_cti,
                'current_knowledge': self.current_knowledge,
                'price_payed': price_payed,
                'wasted': False}


    def restart_budget(self):

        # now we can restart the budget
        self.current_budget = self.init_budget


