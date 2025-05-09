import time
import requests
import threading

G2 = 'G2'
NEW = 'NEW'


class NewTigerEnvironment:

    def __init__(self, kwargs):
        """Initialize the attributes of the Car class."""
        self.init_budget = float(kwargs['tiger_init_budget'] if 'tiger_init_budget' in kwargs else 1)
        self.init_flow_rewards_dict = kwargs['flow_rewards_dict'].copy()
        self.flow_rewards_dict = self.init_flow_rewards_dict.copy()
        self.min_budget = kwargs['min_budget']
        self.max_budget = kwargs['max_budget'] 
        self.current_budget = self.init_budget
        self.init_ZDA_DICT = kwargs['ZDA_DICT']
        self.init_TEST_ZDA_DICT = kwargs['TEST_ZDA_DICT']
        self.init_TRAINING_LABELS_DICT = kwargs['TRAINING_LABELS_DICT'].copy()
        self.max_episode_steps = kwargs['max_episode_steps'] 
        self.cti_price_factor = float(kwargs['cti_price_factor'] if 'cti_price_factor' in kwargs else 20)

    def reset_intelligence(self):
        
        self.current_ZDA_DICT = self.init_ZDA_DICT
        self.current_TEST_ZDA_DICT = self.init_TEST_ZDA_DICT
        self.current_TRAINING_LABELS_DICT = self.init_TRAINING_LABELS_DICT.copy()
        self.flow_rewards_dict = self.init_flow_rewards_dict.copy()
        self.update_cti_options()
        
        
        return {'NEW_ZDA_DICT': self.current_ZDA_DICT,
                'NEW_TEST_ZDA_DICT': self.current_TEST_ZDA_DICT,
                'NEW_TRAINING_LABELS_DICT': self.current_TRAINING_LABELS_DICT,
                'updated_label': None,
                'new_label': None,
                'reset' : True}


    def update_cti_options(self, n_options=1):
        """
        This method updates the CTI agent state vector, i.e., according to available labels to buy.
        """
        
        # current unknowns according to the dict 
        self.unknowns = [label for label in self.current_TRAINING_LABELS_DICT.values() if G2 in label]
        self.cti_prices = {}

        for unknown in self.unknowns:
            try:
                # the cti price is n times the cost or revenue of the corresponding flow 
                self.cti_prices[unknown] = abs(self.flow_rewards_dict[unknown] * self.cti_price_factor)  
            except:
                print('something went wrong...')
        # set a list of n_options available cti options:   
        self.current_cti_options = {}  

        for idx in range(n_options):

            # do we still have so many unknowns? 
            if idx < len(self.unknowns):
                self.current_cti_options[self.unknowns[idx]] = self.cti_prices[self.unknowns[idx]]
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
        print('TIGER ENV: restarting episode!')
        self.restart_budget()
        self.reset_intelligence()


    def perform_epistemic_action(self, current_action=0):
        """
        This method changes the curriculum by turning an attack that
        was a type2 ZDA in a known attack.
        """
        
        new_label = None
        updated_label = None
        changed_ip = None      
        price_payed = 0

        # get the label corresponding to the attack we want to purchase info about 
        acquired_cti = list(self.current_cti_options.keys())[current_action]

        # if the action corresponds to a placeholder, it means we did not buy anything. 
        if 'placeholder' not in acquired_cti:

            # turn the corresponding ip address as not a Zda nor a test Zda and 
            # TAKE OUT THE G2 SUBSTRING FROM THE LABEL 
            for ip, label in self.current_TRAINING_LABELS_DICT.items():

                if label == acquired_cti:
                    
                    # Take out the indicators of ZDA from the curriculum dicts.  
                    self.current_ZDA_DICT[ip] = False
                    self.current_TEST_ZDA_DICT[ip]  = False 
                    new_label = str(label).replace(G2, NEW)
                    self.current_TRAINING_LABELS_DICT[ip] = new_label
                    # This will be used for changing the encoder values in the brain class  
                    updated_label = label
                    changed_ip = ip
                    price_payed = self.cti_prices[label]

                    
                    # update also the rewards dictionary:
                    reward = self.flow_rewards_dict[label]    
                    del self.flow_rewards_dict[label]
                    self.flow_rewards_dict[new_label] = reward

                # update our options vector:
                self.update_cti_options()

        
        return {'NEW_ZDA_DICT': self.current_ZDA_DICT,
                'NEW_TEST_ZDA_DICT': self.current_TEST_ZDA_DICT,
                'NEW_TRAINING_LABELS_DICT': self.current_TRAINING_LABELS_DICT,
                'updated_label': updated_label,
                'new_label': new_label,
                'changed_ip':changed_ip,
                'price_payed': price_payed}


    def restart_budget(self):

        # now we can restart the budget
        self.current_budget = self.init_budget


