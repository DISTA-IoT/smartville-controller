import os
import signal
import logging
import subprocess
import threading
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from fastapi.responses import JSONResponse

logger = logging.getLogger("zookeeper-manager")

ZOOKEEPER_CONFIG_FILE = "zookeeper.properties"
ZOOKEEPER_PROCESS: Optional[subprocess.Popen] = None
ZOOKEEPER_THREAD: Optional[threading.Thread] = None
ZOOKEEPER_LOCK = threading.Lock()
LAST_CONFIG: Optional["ZooKeeperConfig"] = None  # updated by config_zookeeper()


class ZooKeeperConfig(BaseModel):
    # Flat logical model for ZooKeeper configuration
    data_dir: str = "/tmp/zookeeper"
    client_port: int = 2181
    max_client_cnxns: int = 0
    admin_enable_server: bool = False
    admin_server_port: int = 8080
    tick_time: int = 2000
    init_limit: int = 10
    sync_limit: int = 5
    
    @validator("data_dir")
    def _non_empty_dir(cls, v: str) -> str:
        assert isinstance(v, str) and len(v) > 0
        return v
    
    @validator("client_port", "admin_server_port")
    def _valid_port(cls, v: int) -> int:
        assert isinstance(v, int) and 1 <= v <= 65535
        return v

    # --------------------------
    # Constructor for hierarchical or flat payloads
    # --------------------------
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ZooKeeperConfig":
        """
        Accepts flat JSON payload with ZooKeeper configuration:
        {
            "data_dir": "/tmp/zookeeper",
            "client_port": 2181,
            "max_client_cnxns": 0,
            "admin_enable_server": false,
            "admin_server_port": 8080,
            "tick_time": 2000,
            "init_limit": 10,
            "sync_limit": 5
        }
        Missing fields fall back to defaults.
        """
        payload = payload or {}

        # Get value with fallbacks: support both underscore and camelCase variants
        def get_with_fallback(key: str, default=None):
            # Try exact key first
            if key in payload:
                return payload[key]
            
            # Try common variants
            variants = [
                key.replace("_", ""),  # dataDir
                key.replace("_", "."),  # data.dir
                "".join(word.capitalize() if i > 0 else word for i, word in enumerate(key.split("_")))  # dataDir
            ]
            
            for variant in variants:
                if variant in payload:
                    return payload[variant]
            
            return default

        data_dir = get_with_fallback("data_dir", default="/tmp/zookeeper")
        client_port = get_with_fallback("client_port", default=2181)
        max_client_cnxns = get_with_fallback("max_client_cnxns", default=0)
        admin_enable_server = get_with_fallback("admin_enable_server", default=False)
        admin_server_port = get_with_fallback("admin_server_port", default=8080)
        tick_time = get_with_fallback("tick_time", default=2000)
        init_limit = get_with_fallback("init_limit", default=10)
        sync_limit = get_with_fallback("sync_limit", default=5)

        return cls(
            data_dir=data_dir,
            client_port=int(client_port),
            max_client_cnxns=int(max_client_cnxns),
            admin_enable_server=bool(admin_enable_server),
            admin_server_port=int(admin_server_port),
            tick_time=int(tick_time),
            init_limit=int(init_limit),
            sync_limit=int(sync_limit),
        )

    # --------------------------
    # Properties file rendering
    # --------------------------
    def to_properties_content(self) -> str:
        """Generate ZooKeeper properties file content."""
        lines = [
            "# ZooKeeper configuration file",
            "# the directory where the snapshot is stored.",
            f"dataDir={self.data_dir}",
            "# the port at which the clients will connect",
            f"clientPort={self.client_port}",
            "# disable the per-ip limit on the number of connections since this is a non-production config",
            f"maxClientCnxns={self.max_client_cnxns}",
            "# Basic time unit in milliseconds used by ZooKeeper",
            f"tickTime={self.tick_time}",
            "# Amount of time to allow followers to connect and sync to a leader",
            f"initLimit={self.init_limit}",
            "# Amount of time to allow followers to sync with ZooKeeper",
            f"syncLimit={self.sync_limit}",
            "# Admin server configuration",
            f"admin.enableServer={str(self.admin_enable_server).lower()}",
        ]
        
        if self.admin_enable_server:
            lines.append(f"admin.serverPort={self.admin_server_port}")
        else:
            lines.append(f"admin.serverPort={self.admin_server_port}")
        
        return "\n".join(lines) + "\n"


# --------------------------
# Public functions used by your FastAPI
# --------------------------
def config_zookeeper(config_payload: Dict[str, Any]):
    """
    Build config from payload and write zookeeper.properties.
    Also tracks LAST_CONFIG for runtime usage.
    """
    global LAST_CONFIG
    cfg = ZooKeeperConfig.from_payload(config_payload)
    properties_content = cfg.to_properties_content()

    # Ensure data directory exists
    try:
        os.makedirs(cfg.data_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not ensure data directory exists ({cfg.data_dir}): {e}")

    with open(ZOOKEEPER_CONFIG_FILE, "w") as f:
        f.write(properties_content)

    LAST_CONFIG = cfg
    logger.info(f"ZooKeeper configuration written to {ZOOKEEPER_CONFIG_FILE}")
    response_content = {"msg": f"Config written to {ZOOKEEPER_CONFIG_FILE}"}
    return JSONResponse(content=response_content, status_code=200)


def start_zookeeper():
    """
    Start ZooKeeper as a background thread using the configured properties file.
    """
    global ZOOKEEPER_PROCESS, ZOOKEEPER_THREAD, LAST_CONFIG

    with ZOOKEEPER_LOCK:
        if ZOOKEEPER_PROCESS and ZOOKEEPER_PROCESS.poll() is None:
            return {"status_code": 304, "msg": "ZooKeeper already running."}

        # Ensure data directory exists
        if LAST_CONFIG:
            try:
                os.makedirs(LAST_CONFIG.data_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not ensure data directory exists: {e}")

        def run_zookeeper():
            global ZOOKEEPER_PROCESS
            cmd = [
                "zookeeper-server-start.sh",
                ZOOKEEPER_CONFIG_FILE
            ]
            logger.info("Starting ZooKeeper: %s", " ".join(cmd))
            ZOOKEEPER_PROCESS = subprocess.Popen(cmd)
            rc = ZOOKEEPER_PROCESS.wait()
            logger.info("ZooKeeper process ended with code %s.", rc)

        ZOOKEEPER_THREAD = threading.Thread(target=run_zookeeper, daemon=True)
        ZOOKEEPER_THREAD.start()
        response_content = {"msg": "ZooKeeper started."}
        return JSONResponse(content=response_content, status_code=200)


def check_zookeeper():
    """
    Check ZooKeeper process status.
    Returns: (running, pid, error)
    """
    pid = None
    running = None
    error = None
    with ZOOKEEPER_LOCK:
        if ZOOKEEPER_PROCESS is None:
            running = False
        elif ZOOKEEPER_PROCESS.poll() is None:
            running = True
            pid = ZOOKEEPER_PROCESS.pid
        else:
            running = False
            error = ZOOKEEPER_PROCESS.returncode
            
        return running, pid, error


def stop_zookeeper():
    """
    Stop ZooKeeper process gracefully.
    """
    global ZOOKEEPER_PROCESS
    with ZOOKEEPER_LOCK:
        if ZOOKEEPER_PROCESS is None or ZOOKEEPER_PROCESS.poll() is not None:
            response_content = {"msg": "ZooKeeper is not running."}
            return JSONResponse(content=response_content, status_code=202)
        
        pid = ZOOKEEPER_PROCESS.pid
        logger.info(f"Stopping ZooKeeper (PID={pid})")
        
        try:
            ZOOKEEPER_PROCESS.send_signal(signal.SIGTERM)
            ZOOKEEPER_PROCESS.wait(timeout=10)
            logger.info("ZooKeeper stopped gracefully.")
        except subprocess.TimeoutExpired:
            logger.warning("ZooKeeper did not stop gracefully, forcing termination.")
            ZOOKEEPER_PROCESS.kill()
            ZOOKEEPER_PROCESS.wait()
            logger.info("ZooKeeper force-stopped.")
        
        response_content = {"msg": "ZooKeeper stopped.", "pid": pid}
        return JSONResponse(content=response_content, status_code=200)


# --------------------------
# Additional utility functions
# --------------------------
def get_current_config():
    """
    Get the current ZooKeeper configuration.
    """
    global LAST_CONFIG
    if LAST_CONFIG is None:
        return {"msg": "No configuration set yet."}
    
    return {
        "data_dir": LAST_CONFIG.data_dir,
        "client_port": LAST_CONFIG.client_port,
        "max_client_cnxns": LAST_CONFIG.max_client_cnxns,
        "admin_enable_server": LAST_CONFIG.admin_enable_server,
        "admin_server_port": LAST_CONFIG.admin_server_port,
        "tick_time": LAST_CONFIG.tick_time,
        "init_limit": LAST_CONFIG.init_limit,
        "sync_limit": LAST_CONFIG.sync_limit,
    }