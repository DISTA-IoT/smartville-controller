import random

G2 = 'G2'
NEW = 'NEW'


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
        self.useless_epistemic_penalty = int(kwargs.intrusion_detection.useless_epistemic_penalty)
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


    def get_cti_prices(self):
        try:
            return {
                # the cti price is n times the cost or revenue of the corresponding flow
                unknown: abs(self.flow_rewards_dict[unknown] * self.init_cti_price_factor)
                for unknown in self.init_knowledge['G2s']
            }
        except KeyError as e:
            raise RuntimeError(f'Error during CTI processing... {e}')


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
                or self.steps_done >= self.max_episode_steps == 0\
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

    def price_decay(self):
        self.current_cti_price_factor *= max(0.01, min(0.99, random.gauss(0.7,0.4)))


    def perform_epistemic_action(self, current_action=0):
        """
        This method changes the curriculum by turning an attack that
        was a type2 ZDA in a known attack.
        TODO check legacy with corresponding tiger_enrivonment action.
        """
               
        price_payed = 0
        # get the label corresponding to the attack we want to purchase info about
        # Optimization: Avoid list(keys()) if possible, but for small n_options it's fine.
        # However, we can use a more direct way if current_action is always 0.
        keys = list(self.current_cti_options.keys())
        if current_action < len(keys):
            acquired_cti = keys[current_action]
        else:
            acquired_cti = 'placeholder'

        self.epistemic_actions += 1
        
        # if the action corresponds to a placeholder, it means we did not buy anything.
        if 'placeholder' not in acquired_cti:
            self.current_knowledge['G2s'].remove(acquired_cti)
            self.current_knowledge['Knowns'].append(acquired_cti)
            self.current_knowledge['updated_labels'].append(acquired_cti)
            price_payed = abs(self.flow_rewards_dict[acquired_cti] * self.current_cti_price_factor)
            self.update_cti_options()
        else:
            acquired_cti = None
            price_payed = self.useless_epistemic_penalty
            
        return {'updated_label': acquired_cti,
                'current_knowledge': self.current_knowledge,
                'price_payed': price_payed}


    def restart_budget(self):

        # now we can restart the budget
        self.current_budget = self.init_budget


