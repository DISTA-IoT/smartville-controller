import time
import requests

class TigerEnvironment:

    def __init__(self, kwargs):
        """Initialize the attributes of the Car class."""
        host_ip_addr = kwargs['host_ip_addr'] 
        self.container_manager_ep = f'http://{host_ip_addr}:7777/'
        self.init_budget = (kwargs['tiger_init_budget'] if 'tiger_init_budget' in kwargs else 100)
        self.init_prices = self.get_init_TCI_prices()
        self.samples_to_acquire = {key: 0 for key in self.init_prices}


    def get_init_TCI_prices(self):
        try:
            response = requests.get(self.container_manager_ep+'init_prices')

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response into a Python dictionary
                data = response.json()
                # Print the received dictionary
                print("Received prices from container maganer server:", data)
                return data
            else:
                print(f"Error: Received status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            assert 1 == 0


    def reset(self):
        self.restart_traffic()
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