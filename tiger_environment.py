import time
import requests


class TigerEnvironment:

    def __init__(self, kwargs):
        """Initialize the attributes of the Car class."""
        host_ip_addr = kwargs['host_ip_addr'] 
        self.container_manager_ep = f'http://{host_ip_addr}:7777/'
        self.init_budget = (kwargs['tiger_init_budget'] if 'tiger_init_budget' in kwargs else 1)
        self.flow_rewards_dict = self.get_flow_rewards()
        self.samples_to_acquire = {key: 0 for key in self.flow_rewards_dict}
        self.min_budget = kwargs['min_budget']
        self.max_budget = kwargs['max_budget']
        self.restart_traffic()

    def has_episode_ended(self):
        if self.current_budget > self.max_budget:
            return True
        else:
            return self.current_budget < self.min_budget
    

    def get_flow_rewards(self):
        try:
            response = requests.get(self.container_manager_ep+'flow_rewards')

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response into a Python dictionary
                data = response.json()
                # Print the received dictionary
                print("Flow rewards received from container maganer server:")
                return data
            else:
                print(f"Error: Received status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            assert 1 == 0


    def reset(self):
        print('TIGER ENV: restarting episode!')
        # self.restart_traffic()
        self.restart_budget()


    def restart_budget(self):
        self.current_budget = self.init_budget


    def restart_traffic(self):
        print(f'stopping traffic...')
        response = requests.get(self.container_manager_ep+'stop_traffic')
        time.sleep(3)
        print(f'launching traffic...')
        response = requests.get(self.container_manager_ep+'launch_traffic')
        if response.status_code == 200:
            data = response.text
            print(data)
        else:
            print(f"Error: Received status code {response.status_code}")