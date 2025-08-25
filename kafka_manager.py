import os
import signal
import logging
import subprocess
import threading
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from fastapi.responses import JSONResponse
import shutil

logger = logging.getLogger("kafka-manager")

KAFKA_CONFIG_FILE = "kafka_server.properties"
KAFKA_PROCESS: Optional[subprocess.Popen] = None
KAFKA_THREAD: Optional[threading.Thread] = None
KAFKA_LOCK = threading.Lock()
LAST_CONFIG: Optional["KafkaConfig"] = None  # updated by config_kafka()


class KafkaConfig(BaseModel):
    # Core Kafka server configuration
    broker_id: int = 0
    listeners: str = "PLAINTEXT://192.168.1.1:9092"
    advertised_listeners: Optional[str] = None
    listener_security_protocol_map: Optional[str] = None
    
    # Network configuration
    num_network_threads: int = 3
    num_io_threads: int = 8
    socket_send_buffer_bytes: int = 102400
    socket_receive_buffer_bytes: int = 102400
    socket_request_max_bytes: int = 104857600
    
    # Log configuration
    log_dirs: str = "/tmp/kafka-logs"
    num_partitions: int = 1
    num_recovery_threads_per_data_dir: int = 1
    
    # Replication configuration
    offsets_topic_replication_factor: int = 1
    transaction_state_log_replication_factor: int = 1
    transaction_state_log_min_isr: int = 1
    
    # Log retention configuration
    log_flush_interval_messages: Optional[int] = None
    log_flush_interval_ms: Optional[int] = None
    log_retention_hours: int = 168
    log_retention_check_interval_ms: int = 300000
    
    # ZooKeeper configuration
    zookeeper_connect: str = "localhost:2181"
    zookeeper_connection_timeout_ms: int = 18000
    
    # Group coordination
    group_initial_rebalance_delay_ms: int = 0
    
    @validator("log_dirs")
    def _non_empty_dir(cls, v: str) -> str:
        assert isinstance(v, str) and len(v) > 0
        return v
    
    @validator("broker_id")
    def _valid_broker_id(cls, v: int) -> int:
        assert isinstance(v, int) and v >= 0
        return v

    # --------------------------
    # Constructor for hierarchical or flat payloads
    # --------------------------
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "KafkaConfig":
        """
        Accepts flat JSON payload with Kafka configuration:
        {
            "broker_id": 0,
            "listeners": "PLAINTEXT://192.168.1.1:9092",
            "advertised_listeners": null,
            "num_network_threads": 3,
            "num_io_threads": 8,
            "socket_send_buffer_bytes": 102400,
            "socket_receive_buffer_bytes": 102400,
            "socket_request_max_bytes": 104857600,
            "log_dirs": "/tmp/kafka-logs",
            "num_partitions": 1,
            "num_recovery_threads_per_data_dir": 1,
            "offsets_topic_replication_factor": 1,
            "transaction_state_log_replication_factor": 1,
            "transaction_state_log_min_isr": 1,
            "log_retention_hours": 168,
            "log_retention_check_interval_ms": 300000,
            "zookeeper_connect": "localhost:2181",
            "zookeeper_connection_timeout_ms": 18000,
            "group_initial_rebalance_delay_ms": 0
        }
        Missing fields fall back to defaults.
        """
        payload = payload or {}

        # Get value with fallbacks: support both underscore and dot notation variants
        def get_with_fallback(key: str, default=None):
            # Try exact key first
            if key in payload:
                return payload[key]
            
            # Try common variants (dot notation, camelCase, etc.)
            variants = [
                key.replace("_", "."),  # broker.id
                key.replace("_", ""),   # brokerid
                "".join(word.capitalize() if i > 0 else word for i, word in enumerate(key.split("_")))  # brokerId
            ]
            
            for variant in variants:
                if variant in payload:
                    return payload[variant]
            
            return default

        # Extract all configuration values with fallbacks
        broker_id = get_with_fallback("broker_id", default=0)
        listeners = get_with_fallback("listeners", default="PLAINTEXT://192.168.1.1:9092")
        advertised_listeners = get_with_fallback("advertised_listeners")
        listener_security_protocol_map = get_with_fallback("listener_security_protocol_map")
        
        num_network_threads = get_with_fallback("num_network_threads", default=3)
        num_io_threads = get_with_fallback("num_io_threads", default=8)
        socket_send_buffer_bytes = get_with_fallback("socket_send_buffer_bytes", default=102400)
        socket_receive_buffer_bytes = get_with_fallback("socket_receive_buffer_bytes", default=102400)
        socket_request_max_bytes = get_with_fallback("socket_request_max_bytes", default=104857600)
        
        log_dirs = get_with_fallback("log_dirs", default="/tmp/kafka-logs")
        num_partitions = get_with_fallback("num_partitions", default=1)
        num_recovery_threads_per_data_dir = get_with_fallback("num_recovery_threads_per_data_dir", default=1)
        
        offsets_topic_replication_factor = get_with_fallback("offsets_topic_replication_factor", default=1)
        transaction_state_log_replication_factor = get_with_fallback("transaction_state_log_replication_factor", default=1)
        transaction_state_log_min_isr = get_with_fallback("transaction_state_log_min_isr", default=1)
        
        log_flush_interval_messages = get_with_fallback("log_flush_interval_messages")
        log_flush_interval_ms = get_with_fallback("log_flush_interval_ms")
        log_retention_hours = get_with_fallback("log_retention_hours", default=168)
        log_retention_check_interval_ms = get_with_fallback("log_retention_check_interval_ms", default=300000)
        
        zookeeper_connect = get_with_fallback("zookeeper_connect", default="localhost:2181")
        zookeeper_connection_timeout_ms = get_with_fallback("zookeeper_connection_timeout_ms", default=18000)
        
        group_initial_rebalance_delay_ms = get_with_fallback("group_initial_rebalance_delay_ms", default=0)

        return cls(
            broker_id=int(broker_id),
            listeners=str(listeners),
            advertised_listeners=str(advertised_listeners) if advertised_listeners is not None else None,
            listener_security_protocol_map=str(listener_security_protocol_map) if listener_security_protocol_map is not None else None,
            num_network_threads=int(num_network_threads),
            num_io_threads=int(num_io_threads),
            socket_send_buffer_bytes=int(socket_send_buffer_bytes),
            socket_receive_buffer_bytes=int(socket_receive_buffer_bytes),
            socket_request_max_bytes=int(socket_request_max_bytes),
            log_dirs=str(log_dirs),
            num_partitions=int(num_partitions),
            num_recovery_threads_per_data_dir=int(num_recovery_threads_per_data_dir),
            offsets_topic_replication_factor=int(offsets_topic_replication_factor),
            transaction_state_log_replication_factor=int(transaction_state_log_replication_factor),
            transaction_state_log_min_isr=int(transaction_state_log_min_isr),
            log_flush_interval_messages=int(log_flush_interval_messages) if log_flush_interval_messages is not None else None,
            log_flush_interval_ms=int(log_flush_interval_ms) if log_flush_interval_ms is not None else None,
            log_retention_hours=int(log_retention_hours),
            log_retention_check_interval_ms=int(log_retention_check_interval_ms),
            zookeeper_connect=str(zookeeper_connect),
            zookeeper_connection_timeout_ms=int(zookeeper_connection_timeout_ms),
            group_initial_rebalance_delay_ms=int(group_initial_rebalance_delay_ms),
        )

    # --------------------------
    # Properties file rendering
    # --------------------------
    def to_properties_content(self) -> str:
        """Generate Kafka server.properties file content."""
        lines = [
            "# Kafka server configuration file",
            "",
            "# Broker configuration",
            f"broker.id={self.broker_id}",
            f"listeners={self.listeners}",
        ]
        
        if self.advertised_listeners:
            lines.append(f"advertised.listeners={self.advertised_listeners}")
        else:
            lines.append("#advertised.listeners=PLAINTEXT://your.host.name:9092")
        
        if self.listener_security_protocol_map:
            lines.append(f"listener.security.protocol.map={self.listener_security_protocol_map}")
        else:
            lines.append("#listener.security.protocol.map=PLAINTEXT:PLAINTEXT,SSL:SSL,SASL_PLAINTEXT:SASL_PLAINTEXT,SASL_SSL:SASL_SSL")
        
        lines.extend([
            "",
            "# Network configuration",
            f"num.network.threads={self.num_network_threads}",
            f"num.io.threads={self.num_io_threads}",
            f"socket.send.buffer.bytes={self.socket_send_buffer_bytes}",
            f"socket.receive.buffer.bytes={self.socket_receive_buffer_bytes}",
            f"socket.request.max.bytes={self.socket_request_max_bytes}",
            "",
            "# Log configuration",
            f"log.dirs={self.log_dirs}",
            f"num.partitions={self.num_partitions}",
            f"num.recovery.threads.per.data.dir={self.num_recovery_threads_per_data_dir}",
            "",
            "# Replication configuration",
            f"offsets.topic.replication.factor={self.offsets_topic_replication_factor}",
            f"transaction.state.log.replication.factor={self.transaction_state_log_replication_factor}",
            f"transaction.state.log.min.isr={self.transaction_state_log_min_isr}",
            "",
            "# Log flush configuration",
        ])
        
        if self.log_flush_interval_messages is not None:
            lines.append(f"log.flush.interval.messages={self.log_flush_interval_messages}")
        else:
            lines.append("#log.flush.interval.messages=10000")
        
        if self.log_flush_interval_ms is not None:
            lines.append(f"log.flush.interval.ms={self.log_flush_interval_ms}")
        else:
            lines.append("#log.flush.interval.ms=1000")
        
        lines.extend([
            "",
            "# Log retention configuration",
            f"log.retention.hours={self.log_retention_hours}",
            f"log.retention.check.interval.ms={self.log_retention_check_interval_ms}",
            "",
            "# ZooKeeper configuration",
            f"zookeeper.connect={self.zookeeper_connect}",
            f"zookeeper.connection.timeout.ms={self.zookeeper_connection_timeout_ms}",
            "",
            "# Group coordination",
            f"group.initial.rebalance.delay.ms={self.group_initial_rebalance_delay_ms}",
        ])
        
        return "\n".join(lines) + "\n"


# --------------------------
# Public functions used by your FastAPI
# --------------------------
def config_kafka(config_payload: Dict[str, Any]):
    """
    Build config from payload and write kafka_server.properties.
    Also tracks LAST_CONFIG for runtime usage.
    """
    global LAST_CONFIG
    cfg = KafkaConfig.from_payload(config_payload)
    properties_content = cfg.to_properties_content()

    # Ensure log directories exist
    log_dirs = cfg.log_dirs.split(",")
    for log_dir in log_dirs:
        log_dir = log_dir.strip()
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not ensure log directory exists ({log_dir}): {e}")

    with open(KAFKA_CONFIG_FILE, "w") as f:
        f.write(properties_content)

    LAST_CONFIG = cfg
    logger.info(f"Kafka configuration written to {KAFKA_CONFIG_FILE}")
    response_content = {"msg": f"Config written to {KAFKA_CONFIG_FILE}"}
    return JSONResponse(content=response_content, status_code=200)


def start_kafka():
    """
    Start Kafka as a background thread using the configured properties file.
    """
    global KAFKA_PROCESS, KAFKA_THREAD, LAST_CONFIG

    with KAFKA_LOCK:
        if KAFKA_PROCESS and KAFKA_PROCESS.poll() is None:
            return {"status_code": 304, "msg": "Kafka already running."}

        # Ensure log directories exist
        if LAST_CONFIG:
            log_dirs = LAST_CONFIG.log_dirs.split(",")
            for log_dir in log_dirs:
                log_dir = log_dir.strip()
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    logger.info(f"Log directory created: {log_dir}")
                except Exception as e:
                    logger.warning(f"Could not ensure log directory exists ({log_dir}): {e}")

        def run_kafka():
            global KAFKA_PROCESS
            cmd = [
                "kafka-server-start.sh",
                KAFKA_CONFIG_FILE
            ]
            logger.info("Starting Kafka: %s", " ".join(cmd))
            KAFKA_PROCESS = subprocess.Popen(cmd)
            rc = KAFKA_PROCESS.wait()
            logger.info("Kafka process ended with code %s.", rc)

        KAFKA_THREAD = threading.Thread(target=run_kafka, daemon=True)
        KAFKA_THREAD.start()
        response_content = {"msg": "Kafka started."}
        return JSONResponse(content=response_content, status_code=200)


def check_kafka():
    """
    Check Kafka process status.
    Returns: (running, pid, error)
    """
    pid = None
    running = None
    error = None
    with KAFKA_LOCK:
        if KAFKA_PROCESS is None:
            running = False
        elif KAFKA_PROCESS.poll() is None:
            running = True
            pid = KAFKA_PROCESS.pid
        else:
            running = False
            error = KAFKA_PROCESS.returncode
            
        return running, pid, error


def stop_kafka():
    """
    Stop Kafka process gracefully.
    """
    global KAFKA_PROCESS
    with KAFKA_LOCK:
        if KAFKA_PROCESS is None or KAFKA_PROCESS.poll() is not None:
            response_content = {"msg": "Kafka is not running."}
            return JSONResponse(content=response_content, status_code=202)
        
        pid = KAFKA_PROCESS.pid
        logger.info(f"Stopping Kafka (PID={pid})")
        
        try:
            KAFKA_PROCESS.send_signal(signal.SIGTERM)
            KAFKA_PROCESS.wait(timeout=15)  # Kafka might take longer to shut down
            logger.info("Kafka stopped gracefully.")

            try:
                shutil.rmtree("/opt/kafka/logs")
                log_dirs = LAST_CONFIG.log_dirs.split(",")
                logger.info("Kafka log directory removed: /opt/kafka/logs")
            except Exception as e:
                logger.warning(f"Could not remove Kafka log directory: {e}")
            
            for log_dir in log_dirs:
                log_dir = log_dir.strip()
                try:
                    shutil.rmtree(log_dir)
                    logger.info(f"Log directory removed: {log_dir}")
                except Exception as e:
                    logger.warning(f"Could not remove Kafka log directory: {e}")
                    
        except subprocess.TimeoutExpired:
            logger.warning("Kafka did not stop gracefully, forcing termination.")
            KAFKA_PROCESS.kill()
            KAFKA_PROCESS.wait()
            logger.info("Kafka force-stopped.")
        
        response_content = {"msg": "Kafka stopped.", "pid": pid}
        return JSONResponse(content=response_content, status_code=200)


# --------------------------
# Additional utility functions
# --------------------------
def get_current_config():
    """
    Get the current Kafka configuration.
    """
    global LAST_CONFIG
    if LAST_CONFIG is None:
        return {"msg": "No configuration set yet."}
    
    return {
        "broker_id": LAST_CONFIG.broker_id,
        "listeners": LAST_CONFIG.listeners,
        "advertised_listeners": LAST_CONFIG.advertised_listeners,
        "listener_security_protocol_map": LAST_CONFIG.listener_security_protocol_map,
        "num_network_threads": LAST_CONFIG.num_network_threads,
        "num_io_threads": LAST_CONFIG.num_io_threads,
        "socket_send_buffer_bytes": LAST_CONFIG.socket_send_buffer_bytes,
        "socket_receive_buffer_bytes": LAST_CONFIG.socket_receive_buffer_bytes,
        "socket_request_max_bytes": LAST_CONFIG.socket_request_max_bytes,
        "log_dirs": LAST_CONFIG.log_dirs,
        "num_partitions": LAST_CONFIG.num_partitions,
        "num_recovery_threads_per_data_dir": LAST_CONFIG.num_recovery_threads_per_data_dir,
        "offsets_topic_replication_factor": LAST_CONFIG.offsets_topic_replication_factor,
        "transaction_state_log_replication_factor": LAST_CONFIG.transaction_state_log_replication_factor,
        "transaction_state_log_min_isr": LAST_CONFIG.transaction_state_log_min_isr,
        "log_flush_interval_messages": LAST_CONFIG.log_flush_interval_messages,
        "log_flush_interval_ms": LAST_CONFIG.log_flush_interval_ms,
        "log_retention_hours": LAST_CONFIG.log_retention_hours,
        "log_retention_check_interval_ms": LAST_CONFIG.log_retention_check_interval_ms,
        "zookeeper_connect": LAST_CONFIG.zookeeper_connect,
        "zookeeper_connection_timeout_ms": LAST_CONFIG.zookeeper_connection_timeout_ms,
        "group_initial_rebalance_delay_ms": LAST_CONFIG.group_initial_rebalance_delay_ms,
    }