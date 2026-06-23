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
from smartController.consumer_thread import ConsumerThread
from grafana_api.grafana_face import GrafanaFace
from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient
from collections import deque
import time
import socket
import threading



class MetricsLogger: 

    def __init__(
            self, 
            kwargs,
            wb_tracker=None):
        self.kwargs = kwargs
        self.kafka_endpoint = kwargs['monitor_ip']+":"+str(kwargs['kafka']['port'])
        self.topics = None
        self.topic_list = []
        self.consumer_threads = []
        self.working_threads_count = 0
        self.kafka_admin_client = None
        self.max_conn_retries = kwargs['health']['max_conn_retries'] 
        self.metrics_to_monitor = kwargs['health']['probe_metrics']
        self.metrics_dict = {}
        self.node_features_time_window = kwargs['health']['node_features_time_window']
        self.grafana_connection = GrafanaFace(
                auth=(kwargs['grafana']['user'], kwargs['grafana']['password']), 
                host=kwargs['monitor_ip']+':'+str(kwargs['grafana']['port']))
        self.logger = kwargs['logger']
        self.consumer_thread_manager = None
        self.wb_tracker = wb_tracker
        # Defining metric Gauges in Prometheus
        self.CPU_metric = None
        self.RAM_metric = None
        self.RTT_metric = None
        self.INBOUND_metric = None
        self.OUTBOUND_metric = None

    def init(self):

        if self.init_kafka_connection():
            self.consumer_thread_manager = threading.Thread(
                target=self.start_consuming, 
                args=())
            
            try:
                self.active = True
                self.consumer_thread_manager.start()
                return True
            except KeyboardInterrupt:
                for thread in self.consumer_threads:
                    if (thread.is_alive()):
                        thread.stop_threads()
                        working_threads_count += 1
                self.logger.info(f" Closed {working_threads_count} threads")

        

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
                self.logger.info(f"Server {host}:{port} REACHED.")
                return True
        except (socket.error, socket.timeout) as e:
            self.logger.error(f"Server {host}:{port} UNREACHABLE: {e}")
            return False


    def init_kafka_connection(self):
        retries = 0
        while retries < self.max_conn_retries: 
            if self.server_exist():
                try:
                    conf = {'bootstrap.servers': self.kafka_endpoint}
                    self.kafka_admin_client = AdminClient(conf)
                    self.topics = self.kafka_admin_client.list_topics(timeout=5).topics

                    # let's delete all topics, we need to start from zero!
                    for topic in self.topics:
                        self.logger.info(f"Deleting topic {topic}")
                        returned_futmap = self.kafka_admin_client.delete_topics([topic])
                        self.logger.info(f"Deleted topic {returned_futmap}")

                    return True
                except KafkaException as e:
                    self.logger.error(f"Kafka connection error {e}")
                    self.kafka_admin_client = None
                    return False
            else:
                self.logger.error(f"Could not find Kafka server at {self.kafka_endpoint}")
                retries += 1
        return False
    

    def shutdown(self):
        self.active = False
        
        # Signal all threads to stop first (non-blocking)
        for consumer_thread in self.consumer_threads:
            consumer_thread.stop()

        # Now join them all — they're all winding down in parallel
        for consumer_thread in self.consumer_threads:
            if consumer_thread.is_alive():
                consumer_thread.join(timeout=1.0)
        
        if self.consumer_thread_manager:
            self.consumer_thread_manager.join(timeout=2.0)
            self.logger.info("Consumer thread stopped")
        self.logger.info("MetricsLogger gracefully shutdown")


    def start_consuming(self):

        while self.active:

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

                self.metrics_dict[topic_name] = {}
                for metric in self.metrics_to_monitor:
                    self.metrics_dict[topic_name][metric] = deque([-1] * self.node_features_time_window, maxlen=self.node_features_time_window)
                

                
                thread = ConsumerThread(
                    self.kafka_endpoint, 
                    topic_name,
                    curr_topics_dict[topic_name],
                    self.CPU_metric,
                    self.RAM_metric,
                    self.RTT_metric,
                    self.INBOUND_metric,
                    self.OUTBOUND_metric,
                    self.metrics_dict,
                    self.wb_tracker,
                    self.kwargs)

                self.consumer_threads.append(thread)
                thread.start()
                self.logger.info(f"Consumer Thread for topic {topic_name} commencing")