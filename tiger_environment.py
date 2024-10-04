import time
import requests

class TigerEnvironment:


    def __init__(self, kwargs):
        """Initialize the attributes of the Car class."""
        host_ip_addr = kwargs['host_ip_addr'] 
        self.container_manager_ep = f'http://{host_ip_addr}:7777/'
    
    
    def reset(self):
        self.restart_traffic()


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