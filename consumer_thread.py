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
from confluent_kafka import Consumer, KafkaException
from confluent_kafka.admin import AdminClient
import threading
import math
import string
import random
import json

RAM = 'RAM'
CPU = 'CPU'
INBOUND = 'INBOUND'
OUTBOUND = 'OUTBOUND'
RTT = 'RTT'


class ConsumerThread(threading.Thread):


    def __init__(
            self, 
            bootstrap_servers,
            topic_name, 
            topic_object,
            cpu_metric, 
            ram_metric, 
            rtt_metric, 
            inbound_metric,
            outbound_metric,
            controller_metrics_dict,
            kwargs
            ):
        
        threading.Thread.__init__(self)

        self.lock = threading.Lock()
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.topic_object = topic_object
        self.cpu_metric = cpu_metric
        self.ram_metric = ram_metric
        self.rtt_metric = rtt_metric
        self.inbound_metric = inbound_metric
        self.outbound_metric = outbound_metric
        self.exit_signal = threading.Event()
        self.controller_metrics_dict = controller_metrics_dict
        self.kwargs = kwargs
        self.logger = kwargs['logger']

    # Definizione metodi di aggiornamento delle metriche nelle rispettive variabili
    def update_cpu_metric(self, value, label_value):
        self.cpu_metric.labels(label_name=label_value).set(value)
        with self.lock:
            if value == b'nan' or math.isnan(value):
                value = -1.0
            self.controller_metrics_dict[self.topic_name][CPU].append(value)


    def update_ram_metric(self, value, label_value):
        self.ram_metric.labels(label_name=label_value).set(value)
        with self.lock:
            if value == b'nan' or math.isnan(value):
                value = -1.0
            self.controller_metrics_dict[self.topic_name][RAM].append(value)

    def update_rtt_metric(self, value, label_value):
        self.rtt_metric.labels(label_name=label_value).set(value)
        with self.lock:
            if value == b'nan' or math.isnan(value):
                value = -1.0
            self.controller_metrics_dict[self.topic_name][RTT].append(value)

    def update_incoming_traffic_metric(self, value, label_value):
        self.inbound_metric.labels(label_name=label_value).set(value)
        with self.lock:
            if value == b'nan' or math.isnan(value):
                value = -1.0
            self.controller_metrics_dict[self.topic_name][INBOUND].append(value)

    def update_outcoming_traffic_metric(self, value, label_value):
        self.outbound_metric.labels(label_name=label_value).set(value)
        with self.lock:
            if value == b'nan' or math.isnan(value):
                value = -1.0
            self.controller_metrics_dict[self.topic_name][OUTBOUND].append(value)

    def stop_threads(self):
        print("Stopping thread...")
        self.exit_signal.set()
    

    def deserialize_message(self, msg):
        """
        Deserialize the JSON-serialized data received from the Kafka Consumer.

        Args:
            msg (Message): The Kafka message object.

        Returns:
            dict or None: The deserialized Python dictionary if successful, otherwise None.
        """
        try:
            # Decode the message and deserialize it into a Python dictionary
            message_value = json.loads(msg.value().decode('utf-8'))
            self.logger.debug(f"Deserialized message: {message_value}")
            return message_value
        except json.JSONDecodeError as e:
            self.logger.error(f"Error deserializing message: {e}")
            return None
        
    
    def process_message(self, message):

        self.received_messages += 1

        if (message[self.topic_name+"_"+CPU]):
            self.logger.debug(f'CPU probe received from {self.topic_name}: {message[self.topic_name+"_"+CPU]}')
            self.update_cpu_metric(float(message[self.topic_name+"_"+CPU]), self.topic_name)

        if (message[self.topic_name+"_"+RAM]):
            self.logger.debug(f'RAM probe received from {self.topic_name}: {message[self.topic_name+"_"+RAM]}')
            self.update_ram_metric(float(message[self.topic_name+"_"+RAM]), self.topic_name)

        if (message[self.topic_name+"_"+RTT]):
            self.logger.debug(f'RTT probe received from {self.topic_name}: {message[self.topic_name+"_"+RTT]}')
            self.update_rtt_metric(float(message[self.topic_name+"_"+RTT]), self.topic_name)

        if (message[self.topic_name+"_"+INBOUND]):
            self.logger.debug(f'IN_TRAFFIC probe received from {self.topic_name}: {message[self.topic_name+"_"+INBOUND]}')
            self.update_incoming_traffic_metric(float(message[self.topic_name+"_"+INBOUND]), self.topic_name)

        if (message[self.topic_name+"_"+OUTBOUND]):
            self.logger.debug(f'OUT_TRAFFIC probe received from {self.topic_name}: {message[self.topic_name+"_"+OUTBOUND]}')
            self.update_outcoming_traffic_metric(float(message[self.topic_name+"_"+OUTBOUND]), self.topic_name)


    def run(self):

        def generate_random_string(length=10):
            letters = string.ascii_letters + string.digits
            return ''.join(random.choice(letters) for i in range(length))
    
        consumer_conf = {'bootstrap.servers': self.bootstrap_servers, 
                         'group.id': generate_random_string(7),  # Consumer group ID for message offset tracking
                         'auto.offset.reset': 'earliest'  # Start reading from the earliest message if no offset is present
                        }
        consumer = Consumer(consumer_conf)

        conf = {'bootstrap.servers': self.bootstrap_servers}
        admin_client = AdminClient(conf)
        
        self.received_messages = 0       

        consumer.subscribe([self.topic_name])

        try:

            while not self.exit_signal.is_set():

                poll_temptative = 0
                message_received = False
                while not message_received and self.kwargs['health']['max_failed_polls'] > poll_temptative:
                    poll_temptative += 1
                    try:
                        # well wait a message max for poll_timeout_seconds secs...
                        msg = consumer.poll(timeout=self.kwargs['health']['poll_timeout_seconds'])
                        if msg is not None:
                            message_received = True
                    except KafkaException as e:
                        self.logger.error(f"Kafka consuming error {e}")

                # we put nans if there's no message after three secs
                if msg is None :
                    self.logger.warning(f'No message received for {self.topic_name} after {self.kwargs["health"]["max_failed_polls"]} polls')
                    self.logger.warning(f'Will now halt the consumer thread for {self.topic_name} and delete the topic')
                    break

                self.logger.debug(f'Got message: {msg.value().decode("utf-8")} from partition {msg.partition()}')

                if msg.error():
                    if msg.error().code() == KafkaException._PARTITION_EOF:
                        self.logger.warning(f'End of partition reached for {msg.topic()}')
                        continue
                    else:
                        self.logger.error(f'Consumer Error: {msg.error()}')
                        break

                # Each metric will be sent to prometheus
                deserialized_data = self.deserialize_message(msg)
                if deserialized_data:
                    self.process_message(deserialized_data)

        # Notice we delete kafka topic at the end of the consuming process    
        finally:
            admin_client.delete_topics([self.topic_name])
            self.logger.info(f"Deleted topic {self.topic_name}")
            self.logger.info(f"Consuming thread for topic {self.topic_name}: stopped!")


    