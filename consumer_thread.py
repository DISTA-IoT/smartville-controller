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
from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient
import threading
import math
import string
import random
import json

RAM = 'RAM'
CPU = 'CPU'

external_http_rtt = 'external_http_rtt'
icmp_min_rtt_ms = 'icmp_min_rtt_ms'
icmp_max_rtt_ms = 'icmp_max_rtt_ms'
icmp_avg_rtt_ms = 'icmp_avg_rtt_ms'
icmp_loss_percent = 'icmp_loss_percent'
http_min_rtt_ms = 'http_min_rtt_ms'
http_max_rtt_ms = 'http_max_rtt_ms'
http_avg_rtt_ms = 'http_avg_rtt_ms'
inbound_MBps = 'inbound_MBps'
inbound_packets_per_second = 'inbound_packets_per_second'
outbound_MBps = 'outbound_MBps'
outbound_packets_per_second = 'outbound_packets_per_second'



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
            self.controller_metrics_dict[self.topic_name][external_http_rtt].append(value)

    def update_incoming_traffic_metric(self, value, label_value):
        self.inbound_metric.labels(label_name=label_value).set(value)
        with self.lock:
            if value == b'nan' or math.isnan(value):
                value = -1.0
            self.controller_metrics_dict[self.topic_name][inbound_MBps].append(value)

    def update_outcoming_traffic_metric(self, value, label_value):
        self.outbound_metric.labels(label_name=label_value).set(value)
        with self.lock:
            if value == b'nan' or math.isnan(value):
                value = -1.0
            self.controller_metrics_dict[self.topic_name][outbound_MBps].append(value)


    def update_generic_metric(self, value, label_value, metric_name):
        self.controller_metrics_dict[self.topic_name][metric_name].append(value)

    

    def stop(self):
        self.logger.info(f"Stopping consumer thread for topic: {self.topic_name}")
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

        if CPU in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+CPU in message.keys():
                self.logger.debug(f'CPU probe received from {self.topic_name}: {message[self.topic_name+"_"+CPU]}')
                self.update_cpu_metric(float(message[self.topic_name+"_"+CPU]), self.topic_name)
            else:
                self.logger.warning(f'Configuration says {CPU} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')
            

        if RAM in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+RAM in message.keys():
                self.logger.debug(f'RAM probe received from {self.topic_name}: {message[self.topic_name+"_"+RAM]}')
                self.update_ram_metric(float(message[self.topic_name+"_"+RAM]), self.topic_name)
            else:
                self.logger.warning(f'Configuration says {RAM} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')
                

        if external_http_rtt in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+external_http_rtt in message.keys():
                self.logger.debug(f'{external_http_rtt} probe received from {self.topic_name}: {message[self.topic_name+"_"+external_http_rtt]}')
                self.update_rtt_metric(float(message[self.topic_name+"_"+external_http_rtt]), self.topic_name)
            else:
                self.logger.warning(f'Configuration says {external_http_rtt} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')


        if icmp_min_rtt_ms in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+icmp_min_rtt_ms in message.keys():
                self.logger.debug(f'{icmp_min_rtt_ms} probe received from {self.topic_name}: {message[self.topic_name+"_"+icmp_min_rtt_ms]}')
                self.update_generic_metric(float(message[self.topic_name+"_"+icmp_min_rtt_ms]), self.topic_name, icmp_min_rtt_ms)
            else:
                self.logger.warning(f'Configuration says {icmp_min_rtt_ms} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')
                

        if icmp_max_rtt_ms in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+icmp_max_rtt_ms in message.keys():
                self.logger.debug(f'{icmp_max_rtt_ms} probe received from {self.topic_name}: {message[self.topic_name+"_"+icmp_max_rtt_ms]}')
                self.update_generic_metric(float(message[self.topic_name+"_"+icmp_max_rtt_ms]), self.topic_name, icmp_max_rtt_ms)
            else:
                self.logger.warning(f'Configuration says {icmp_max_rtt_ms} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')
                

        if icmp_avg_rtt_ms in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+icmp_avg_rtt_ms in message.keys():
                self.logger.debug(f'{icmp_avg_rtt_ms} probe received from {self.topic_name}: {message[self.topic_name+"_"+icmp_avg_rtt_ms]}')
                self.update_generic_metric(float(message[self.topic_name+"_"+icmp_avg_rtt_ms]), self.topic_name, icmp_avg_rtt_ms)
            else:
                self.logger.warning(f'Configuration says {icmp_avg_rtt_ms} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')
                

        if icmp_loss_percent in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+icmp_loss_percent in message.keys():
                self.logger.debug(f'{icmp_loss_percent} probe received from {self.topic_name}: {message[self.topic_name+"_"+icmp_loss_percent]}')
                self.update_generic_metric(float(message[self.topic_name+"_"+icmp_loss_percent]), self.topic_name, icmp_loss_percent)
            else:
                self.logger.warning(f'Configuration says {icmp_loss_percent} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')
                

        if http_avg_rtt_ms in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+http_avg_rtt_ms in message.keys():
                self.logger.debug(f'{http_avg_rtt_ms} probe received from {self.topic_name}: {message[self.topic_name+"_"+http_avg_rtt_ms]}')
                self.update_generic_metric(float(message[self.topic_name+"_"+http_avg_rtt_ms]), self.topic_name, http_avg_rtt_ms)
            else:
                self.logger.warning(f'Configuration says {http_avg_rtt_ms} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')
                
        if http_max_rtt_ms in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+http_max_rtt_ms in message.keys():
                self.logger.debug(f'{http_max_rtt_ms} probe received from {self.topic_name}: {message[self.topic_name+"_"+http_max_rtt_ms]}')
                self.update_generic_metric(float(message[self.topic_name+"_"+http_max_rtt_ms]), self.topic_name, http_max_rtt_ms)
            else:
                self.logger.warning(f'Configuration says {http_max_rtt_ms} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')
                
        
        if http_min_rtt_ms in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+http_min_rtt_ms in message.keys():
                self.logger.debug(f'{http_min_rtt_ms} probe received from {self.topic_name}: {message[self.topic_name+"_"+http_min_rtt_ms]}')
                self.update_generic_metric(float(message[self.topic_name+"_"+http_min_rtt_ms]), self.topic_name, http_min_rtt_ms)
            else:
                self.logger.warning(f'Configuration says {http_min_rtt_ms} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')
                

        if inbound_MBps in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+inbound_MBps in message.keys():
                self.logger.debug(f'{inbound_MBps} probe received from {self.topic_name}: {message[self.topic_name+"_"+inbound_MBps]}')
                self.update_incoming_traffic_metric(float(message[self.topic_name+"_"+inbound_MBps]), self.topic_name)
            else:
                self.logger.warning(f'Configuration says {inbound_MBps} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')


        if inbound_packets_per_second in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+inbound_packets_per_second in message.keys():
                self.logger.debug(f'{inbound_packets_per_second} probe received from {self.topic_name}: {message[self.topic_name+"_"+inbound_packets_per_second]}')
                self.update_generic_metric(float(message[self.topic_name+"_"+inbound_packets_per_second]), self.topic_name, inbound_packets_per_second)
            else:
                self.logger.warning(f'Configuration says {inbound_packets_per_second} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')


        if outbound_packets_per_second in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+outbound_packets_per_second in message.keys():
                self.logger.debug(f'{outbound_packets_per_second} probe received from {self.topic_name}: {message[self.topic_name+"_"+outbound_packets_per_second]}')
                self.update_generic_metric(float(message[self.topic_name+"_"+outbound_packets_per_second]), self.topic_name, outbound_packets_per_second)
            else:
                self.logger.warning(f'Configuration says {outbound_packets_per_second} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')
                

        if outbound_MBps in self.kwargs['health']['probe_metrics']:
            if self.topic_name+"_"+outbound_MBps in message.keys():
                self.logger.debug(f'{outbound_MBps} probe received from {self.topic_name}: {message[self.topic_name+"_"+outbound_MBps]}')
                self.update_outcoming_traffic_metric(float(message[self.topic_name+"_"+outbound_MBps]), self.topic_name)
            else:
                self.logger.warning(f'Configuration says {outbound_MBps} is among the metrics to collect. \n' +\
                                    f'However, a message without such metric was NOT received in topic {self.topic_name}! \n' +\
                                    'The corresponding feature vecs will be -1s')


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
                    if msg.error().code() == KafkaError._PARTITION_EOF:
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


    