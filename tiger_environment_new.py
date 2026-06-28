import random


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
                or self.steps_done >= self.max_episode_steps\
                or self.current_budget > self.max_budget:
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
        # the same 7 series (reappearances/<label>, net gain net of price)
        # whether or not that label gets bought this episode.
        self.acquired_g2_stats = {
            label: {'bought': False, 'price_paid': 0.0,
                    'reappearances': 0, 'reward_since_purchase': 0.0}
            for label in self.init_knowledge['G2s']
        }

    def record_reappearances(self, true_label_names, per_sample_rewards):
        """
        Called once per online tick with the true label and reward of every
        online sample. Only accumulates for G2 labels already bought this
        episode -- pre-purchase occurrences are accounted for by the
        unknown-cluster reward path, not this CTI-ROI tracker.
        """
        for name, reward in zip(true_label_names, per_sample_rewards):
            stats = self.acquired_g2_stats.get(name)
            if stats is not None and stats['bought']:
                stats['reappearances'] += 1
                stats['reward_since_purchase'] += reward

    def price_decay(self):
        self.current_cti_price_factor *= max(0.01, min(0.99, random.gauss(0.7,0.4)))


    def perform_epistemic_action(self, current_action=0):
        """
        Buys CTI for the G2 class at `current_action`'s slot in
        `current_cti_options`, turning it into a known class. Only called
        when `epistemic_actions_available == 1` (the DM's action-2 is
        masked to a block otherwise), so there's always a real label to buy.
        """
        acquired_cti = list(self.current_cti_options.keys())[current_action]

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
                'price_payed': price_payed}


    def restart_budget(self):

        # now we can restart the budget
        self.current_budget = self.init_budget


