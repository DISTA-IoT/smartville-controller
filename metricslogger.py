# This file is part of the "Smartville" project.
# Copyright (c) 2024 University of Insubria
# Licensed under the Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# For the full text of the license, visit:
# https://www.apache.org/licenses/LICENSE-2.0

# Smartville is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License 2.0 for more details.

# You should have received a copy of the Apache License 2.0
# along with Smartville. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

# Additional licensing information for third-party dependencies
# used in this file can be found in the accompanying `NOTICE` file.
from prometheus_client import start_http_server, Gauge
from prometheus_api_client import PrometheusConnect
from smartController.consumer_thread import ConsumerThread
from smartController.dashgenerator import DashGenerator
from smartController.graphgenerator import GraphGenerator
from grafana_api.grafana_face import GrafanaFace
from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient
from collections import deque
import time
import socket
import threading

RAM = 'RAM'
CPU = 'CPU'
INBOUND = 'INBOUND'
OUTBOUND = 'OUTBOUND'
RTT = 'RTT'




class MetricsLogger: 

    def __init__(
            self, 
            kwargs):
        self.kwargs = kwargs
        self.kafka_endpoint = kwargs['monitor_ip']+":"+str(kwargs['kafka']['port'])
        self.topics = None
        self.topic_list = []
        self.threads = []
        self.working_threads_count = 0
        self.sortcount = 0
        self.kafka_admin_client = None
        self.max_conn_retries = kwargs['health']['max_conn_retries'] 
        self.metrics_dict = {}
        self.node_features_time_window = kwargs['health']['node_features_time_window']
        self.grafana_connection = GrafanaFace(
                auth=(kwargs['grafana']['user'], kwargs['grafana']['password']), 
                host=kwargs['monitor_ip']+':'+str(kwargs['grafana']['port']))
        self.logger = kwargs['logger']
        if self.init():
            self.logger.info("MetricsLogger initialized")
        else:
            self.logger.error("MetricsLogger initialization failed")

    def init(self):

        if self.init_kafka_connection():     
            self.init_prometheus_server()
            try:
                self.dash_generator = DashGenerator(self.grafana_connection, self.logger, self.max_conn_retries)
            except Exception as e:
                self.logger.error(f"Error during dashboard generation: {e}")
                return False
            try:
                self.graph_generator = GraphGenerator(
                    grafana_connection=self.grafana_connection,
                    prometheus_connection=self.prometheus_connection)
            except Exception as e:
                self.logger.error(f"Error during graph generation: {e}")
                return False
            
            self.consumer_thread_manager = threading.Thread(
                target=self.start_consuming, 
                args=())
            
            try:
                self.consumer_thread_manager.start()
                return True
            except KeyboardInterrupt:
                for thread in self.threads:
                    if (thread.is_alive()):
                        thread.stop_threads()
                        working_threads_count += 1
                self.logger.info(f" Closed {working_threads_count} threads")
                return False

        else: 
            return False
        

    def server_exist(self):

        if ':' not in self.kafka_endpoint:
            self.logger.error("Error: the string must have the format host:port")
            return False
        split_values = self.kafka_endpoint.split(':')

        if len(split_values) != 2 :
            self.logger.error("Error: the string must have the format host:port")
            return False
        host, port = self.kafka_endpoint.split(':')

        if not port.isdigit():
            self.logger.error(f"Error: the port {port} is invalid. It must be a number")
            return False
        try:
            # Attempt to create a socket connection to the Kafka broker
            with socket.create_connection((host, port), timeout=2):
                self.logger.info(f"Server {host}:{port} RAGGIUNTO.")
                return True
        except (socket.error, socket.timeout) as e:
            self.logger.error(f"Server {host}:{port} non raggiungibile")
            return False


    def init_kafka_connection(self):
        retries = 0
        while retries < self.max_conn_retries: 
            if self.server_exist():
                try:
                    conf = {'bootstrap.servers': self.kafka_endpoint}
                    self.kafka_admin_client = AdminClient(conf)
                    self.topics = self.kafka_admin_client.list_topics(timeout=5)
                    return True
                except KafkaException as e:
                    self.logger.error(f"Kafka connection error {e}")
                    self.kafka_admin_client = None
                    return False
            else:
                self.logger.error(f"Could not find Kafka server at {self.kafka_endpoint}")
                retries += 1
        return False
    

    def init_prometheus_server(self):
        start_http_server(port=self.kwargs['prometheus']['clientport'], addr=self.kwargs['prometheus']['clienthost'])

        # Definizione metriche inserite su Prometheus
        self.cpu_metric = Gauge('CPU_percentage', 'CPU percentage metric', ['label_name'])
        self.ram_metric = Gauge('RAM_GB', 'RAM metric', ['label_name'])
        self.ping_metric = Gauge('Latency ms', 'Network delay metric', ['label_name'])
        self.incoming_traffic_metric = Gauge('Inbound (KB)', 'Inbound traffic metric', ['label_name'])
        self.outcoming_traffic_metric = Gauge('Outbound (KB)', 'Outbound traffic metric', ['label_name'])
        
        # prometheus_connection will permit the graph generator 
        # organize graphs...  
        self.prometheus_connection = PrometheusConnect(self.kwargs['grafana']['datasource_url'])
        

    def start_consuming(self):

        while True:
            updated_topic_list = []
            curr_topics_dict = self.kafka_admin_client.list_topics().topics

            # Inserimento topics in una lista di topics aggiornata
            for topic_name in curr_topics_dict.keys():
                if topic_name != '__consumer_offsets':
                    updated_topic_list.append(topic_name)

            # Creazione di una lista contenente i nuovi topics inseriti
            to_add_topic_list = list(set(updated_topic_list) - set(self.topic_list))

            # La lista di topics aggiornata prende il posto della lista di topics vecchia
            self.topic_list = updated_topic_list

            time.sleep(5)

            # Per ciascun topic nuovo, viene avviato un thread dedicato alla lettura delle metriche
            for topic_name in to_add_topic_list:

                self.graph_generator.generate_all_graphs(topic_name)

                
                self.metrics_dict[topic_name] = {
                    CPU: deque(maxlen=self.node_features_time_window), 
                    RTT: deque(maxlen=self.node_features_time_window), 
                    INBOUND: deque(maxlen=self.node_features_time_window), 
                    OUTBOUND: deque(maxlen=self.node_features_time_window),
                    RAM: deque(maxlen=self.node_features_time_window) 
                    }
                

                
                thread = ConsumerThread(
                    self.kafka_endpoint, 
                    topic_name,
                    curr_topics_dict[topic_name],
                    self.cpu_metric,
                    self.ram_metric,
                    self.ping_metric,
                    self.incoming_traffic_metric,
                    self.outcoming_traffic_metric,
                    self.metrics_dict,
                    self.kwargs)

                self.threads.append(thread)
                thread.start()
                self.logger.info(f"Consumer Thread for topic {topic_name} commencing")

            if (self.sortcount>=12):     # Ogni minuto (5 secs * 12)
                self.logger.info(f"Organizing dashboard priorities...")
                self.graph_generator.sort_all_graphs()
                self.sortcount = 0

            self.sortcount +=1