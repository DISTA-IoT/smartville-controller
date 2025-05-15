import time
import requests
import threading

G2 = 'G2'
NEW = 'NEW'


class NewTigerEnvironment:

    def __init__(self, kwargs):
        """Initialize the attributes of the Car class."""
        self.init_budget = float(kwargs['tiger_init_budget'] if 'tiger_init_budget' in kwargs else 1)
        self.flow_rewards_dict = kwargs['rewards'].copy()
        self.min_budget = kwargs['min_budget']
        self.max_budget = kwargs['max_budget'] 
        self.current_budget = self.init_budget
        self.traffic_dict = kwargs['traffic_dict'].copy()
        self.init_knowledge = kwargs['knowledge'].copy()
        self.logger = kwargs['logger']
        self.max_episode_steps = kwargs['max_episode_steps'] 
        self.cti_price_factor = float(kwargs['cti_price_factor'] if 'cti_price_factor' in kwargs else 20)

    def reset_intelligence(self):
        
        self.current_knowledge = self.init_knowledge.copy()
        self.current_knowledge['updated_labels'] = []
        self.update_cti_options()
        
        return {'current_knowledge': self.current_knowledge,
                'updated_label': None,
                'new_label': None,
                'reset' : True}


    def update_cti_options(self, n_options=1):
        """
        This method updates the CTI agent state vector, i.e., according to available labels to buy.
        """
                
        self.cti_prices = {}

        for unknown in self.current_knowledge['G2s']:
            try:
                # the cti price is n times the cost or revenue of the corresponding flow 
                self.cti_prices[unknown] = abs(self.flow_rewards_dict[unknown] * self.cti_price_factor)  
            except Exception as e:
                self.logger.error(f'Something went wrong duting CTI processing... {e}')
        # set a list of n_options available cti options:   
        self.current_cti_options = {}  

        for idx in range(n_options):

            # do we still have so many unknowns? 
            if idx < len(self.current_knowledge['G2s']):
                self.current_cti_options[self.current_knowledge['G2s'][idx]] = self.cti_prices[self.current_knowledge['G2s'][idx]]
            else:
                # if we do not have unknowns anymore, then lets put a placeholder in the state space (with high cost).  
                self.current_cti_options[f'placeholder_{idx}'] = 100 


    def get_intelligence_options(self):
        """
        returns n attack-specific technical CTI prices 
        """
        return list(self.current_cti_options.values())


    def has_episode_ended(self, current_steps):
        if self.current_budget < self.min_budget \
                or current_steps % self.max_episode_steps == 0\
                or self.current_budget > self.max_budget:
            return True
        return False


    def has_intelligence_episode_ended(self):
        # TODO do we need to frame these episodes to converge?
        return False


    def reset(self):
        self.logger.info('TIGER ENV: restarting episode!')
        self.restart_budget()
        self.reset_intelligence()


    def perform_epistemic_action(self, current_action=0):
        """
        This method changes the curriculum by turning an attack that
        was a type2 ZDA in a known attack.
        TODO check legacy with corresponding tiger_enrivonment action.
        """
               
        price_payed = 0
        # get the label corresponding to the attack we want to purchase info about 
        acquired_cti = list(self.current_cti_options.keys())[current_action]

        # if the action corresponds to a placeholder, it means we did not buy anything. 
        if 'placeholder' not in acquired_cti:

            self.current_knowledge['G2s'].remove(acquired_cti)
            self.current_knowledge['Knowns'].append(acquired_cti)
            self.current_knowledge['updated_labels'].append(acquired_cti)
            price_payed = self.cti_prices[acquired_cti]
            self.update_cti_options()
        else:
            acquired_cti = None
            # TODO shall we penalize here??
            # price_payed = big money??
            pass
    
        return {'updated_label': acquired_cti,
                'current_knowledge': self.current_knowledge,
                'acquired_pattern': acquired_cti,
                'price_payed': price_payed}


    def restart_budget(self):

        # now we can restart the budget
        self.current_budget = self.init_budget


