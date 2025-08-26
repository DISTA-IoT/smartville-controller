import os
import yaml
import signal
import logging
import subprocess
import threading
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from fastapi.responses import JSONResponse

logger = logging.getLogger("grafana-manager")

GRAFANA_HOME_PATH = "/usr/share/grafana"
GRAFANA_DATASOURCE_FILE = "datasources.yaml"
GRAFANA_DATASOURCE_PATH = os.path.join(GRAFANA_HOME_PATH, "conf/provisioning/datasources", GRAFANA_DATASOURCE_FILE)
GRAFANA_PROCESS: Optional[subprocess.Popen] = None
GRAFANA_THREAD: Optional[threading.Thread] = None
GRAFANA_LOCK = threading.Lock()
LAST_CONFIG: Optional["GrafanaConfig"] = None  # updated by config_grafana()


class DataSource(BaseModel):
    """Individual datasource configuration."""
    name: str = "prometheus"
    type: str = "prometheus"
    access: str = "proxy"
    orgId: int = 1
    uid: str = "my_unique_uid"
    url: str = "http://localhost:9090"
    user: Optional[str] = None
    database: Optional[str] = None
    # basicAuth: Optional[bool] = None
    # basicAuthUser: Optional[str] = None
    # withCredentials: Optional[bool] = None
    # isDefault: Optional[bool] = None
    jsonData: Dict[str, Any] = Field(default_factory=lambda: {
        "graphiteVersion": "1.1",
        "tlsAuth": False,
        "tlsAuthWithCACert": False
    })
    secureJsonData: Dict[str, Any] = Field(default_factory=lambda: {
        "tlsCACert": "...",
        "tlsClientCert": "...",
        "tlsClientKey": "...",
        # "password": None,
        # "basicAuthPassword": None
    })
    version: int = 1
    editable: bool = False


class DeleteDataSource(BaseModel):
    """Datasource deletion configuration."""
    name: str = "prometheus"
    orgId: int = 1


class GrafanaConfig(BaseModel):
    """Grafana configuration model."""
    apiVersion: int = 1
    home_path: str = "/usr/share/grafana"
    deleteDatasources: List[DeleteDataSource] = Field(default_factory=lambda: [
        DeleteDataSource()
    ])
    datasources: List[DataSource] = Field(default_factory=lambda: [
        DataSource()
    ])
    
    @validator("home_path")
    def _non_empty_path(cls, v: str) -> str:
        assert isinstance(v, str) and len(v) > 0
        return v

    # --------------------------
    # Constructor for hierarchical or flat payloads
    # --------------------------
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "GrafanaConfig":
        """
        Accepts payload with Grafana configuration:
        {
            "apiVersion": 1,
            "home_path": "/usr/share/grafana",
            "deleteDatasources": [
                {"name": "prometheus", "orgId": 1}
            ],
            "datasources": [
                {
                    "name": "prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "orgId": 1,
                    "uid": "my_unique_uid",
                    "url": "http://localhost:9090",
                    "user": null,
                    "database": null,
                    "basicAuth": null,
                    "basicAuthUser": null,
                    "withCredentials": null,
                    "isDefault": null,
                    "jsonData": {
                        "graphiteVersion": "1.1",
                        "tlsAuth": false,
                        "tlsAuthWithCACert": false
                    },
                    "secureJsonData": {
                        "tlsCACert": "...",
                        "tlsClientCert": "...",
                        "tlsClientKey": "...",
                        "password": null,
                        "basicAuthPassword": null
                    },
                    "version": 1,
                    "editable": false
                }
            ]
        }
        Missing fields fall back to defaults.
        """
        payload = payload or {}

        # Get value with fallbacks
        def get_with_fallback(key: str, default=None):
            if key in payload:
                return payload[key]
            
            # Try common variants
            variants = [
                key.replace("_", ""),  # homepath
                "".join(word.capitalize() if i > 0 else word for i, word in enumerate(key.split("_")))  # homePath
            ]
            
            for variant in variants:
                if variant in payload:
                    return payload[variant]
            
            return default

        api_version = get_with_fallback("apiVersion", default=1)
        home_path = get_with_fallback("home_path", default="/usr/share/grafana")
        
        # Handle deleteDatasources
        delete_datasources_data = payload.get("deleteDatasources", [{"name": "prometheus", "orgId": 1}])
        delete_datasources = []
        for ds_data in delete_datasources_data:
            delete_datasources.append(DeleteDataSource(**ds_data))
        
        # Handle datasources
        datasources_data = payload.get("datasources", [{}])  # Default to empty dict to use DataSource defaults
        datasources = []
        for ds_data in datasources_data:
            # Merge with defaults for any missing fields
            datasource = DataSource(**ds_data)
            datasources.append(datasource)

        return cls(
            apiVersion=int(api_version),
            home_path=str(home_path),
            deleteDatasources=delete_datasources,
            datasources=datasources,
        )

    # --------------------------
    # YAML rendering
    # --------------------------
    def to_yaml_dict(self) -> Dict[str, Any]:
        """Generate Grafana datasources YAML structure."""
        def clean_dict(d):
            """Remove None values and empty strings from dict."""
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() if v is not None and v != ""}
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            else:
                return d

        # Convert deleteDatasources
        delete_datasources = []
        for ds in self.deleteDatasources:
            delete_datasources.append(clean_dict(ds.dict()))

        # Convert datasources
        datasources = []
        for ds in self.datasources:
            ds_dict = ds.dict()
            # Clean up None values but keep structure
            cleaned_ds = {}
            for k, v in ds_dict.items():
                if k in ["jsonData", "secureJsonData"]:
                    # Keep these dicts even if they have None values
                    cleaned_ds[k] = v
                elif v is not None and v != "":
                    cleaned_ds[k] = v
                else:
                    # Keep the key with empty value for YAML structure
                    cleaned_ds[k] = ""
            datasources.append(cleaned_ds)

        return {
            "apiVersion": self.apiVersion,
            "deleteDatasources": delete_datasources,
            "datasources": datasources,
        }


# --------------------------
# Public functions used by your FastAPI
# --------------------------
def config_grafana(config_payload: Dict[str, Any]):
    """
    Build config from payload and write datasources.yaml to Grafana provisioning directory.
    Also tracks LAST_CONFIG for runtime usage.
    """
    global LAST_CONFIG
    cfg = GrafanaConfig.from_payload(config_payload)
    yaml_dict = cfg.to_yaml_dict()

    # Ensure provisioning directory exists
    datasource_dir = os.path.dirname(GRAFANA_DATASOURCE_PATH)
    try:
        os.makedirs(datasource_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not ensure datasource directory exists ({datasource_dir}): {e}")

    # Write the datasource configuration
    with open(GRAFANA_DATASOURCE_PATH, "w") as f:
        yaml.safe_dump(yaml_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    LAST_CONFIG = cfg
    logger.info(f"Grafana datasource configuration written to {GRAFANA_DATASOURCE_PATH}")
    response_content = {"msg": f"Config written to {GRAFANA_DATASOURCE_PATH}"}
    return JSONResponse(content=response_content, status_code=200)


def start_grafana():
    """
    Start Grafana as a background thread using the configured home path.
    """
    global GRAFANA_PROCESS, GRAFANA_THREAD, LAST_CONFIG

    with GRAFANA_LOCK:
        if GRAFANA_PROCESS and GRAFANA_PROCESS.poll() is None:
            return {"status_code": 304, "msg": "Grafana already running."}

        home_path = (LAST_CONFIG.home_path if LAST_CONFIG else GRAFANA_HOME_PATH)
        
        # Ensure home path exists
        try:
            if not os.path.exists(home_path):
                logger.error(f"Grafana home path does not exist: {home_path}")
                response_content = {"msg": f"Grafana home path does not exist: {home_path}"}
                return JSONResponse(content=response_content, status_code=400)
        except Exception as e:
            logger.warning(f"Could not verify home path exists ({home_path}): {e}")

        def run_grafana():
            global GRAFANA_PROCESS
            cmd = [
                "/usr/share/grafana/bin/grafana",
                "server",
                "-homepath",
                home_path
            ]
            logger.info("Starting Grafana: %s", " ".join(cmd))
            GRAFANA_PROCESS = subprocess.Popen(cmd)
            rc = GRAFANA_PROCESS.wait()
            logger.info("Grafana process ended with code %s.", rc)

        GRAFANA_THREAD = threading.Thread(target=run_grafana, daemon=True)
        GRAFANA_THREAD.start()
        response_content = {"msg": "Grafana started."}
        return JSONResponse(content=response_content, status_code=200)


def check_grafana():
    """
    Check Grafana process status.
    Returns: (running, pid, error)
    """
    pid = None
    running = None
    error = None
    with GRAFANA_LOCK:
        if GRAFANA_PROCESS is None:
            running = False
        elif GRAFANA_PROCESS.poll() is None:
            running = True
            pid = GRAFANA_PROCESS.pid
        else:
            running = False
            error = GRAFANA_PROCESS.returncode
            
        return running, pid, error


def stop_grafana():
    """
    Stop Grafana process gracefully.
    """
    global GRAFANA_PROCESS
    with GRAFANA_LOCK:
        if GRAFANA_PROCESS is None or GRAFANA_PROCESS.poll() is not None:

            response_content = {"msg": "Grafana is not running."}
            return JSONResponse(content=response_content, status_code=202)
        
        pid = GRAFANA_PROCESS.pid
        logger.info(f"Stopping Grafana (PID={pid})")
        
        try:
            GRAFANA_PROCESS.send_signal(signal.SIGTERM)
            GRAFANA_PROCESS.wait(timeout=10)
            logger.info("Grafana stopped gracefully.")
        except subprocess.TimeoutExpired:
            logger.warning("Grafana did not stop gracefully, forcing termination.")
            GRAFANA_PROCESS.kill()
            GRAFANA_PROCESS.wait()
            logger.info("Grafana force-stopped.")
        
        response_content = {"msg": "Grafana stopped.", "pid": pid}
        return JSONResponse(content=response_content, status_code=200)


# --------------------------
# Additional utility functions
# --------------------------
def get_current_config():
    """
    Get the current Grafana configuration.
    """
    global LAST_CONFIG
    if LAST_CONFIG is None:
        return {"msg": "No configuration set yet."}
    
    return {
        "apiVersion": LAST_CONFIG.apiVersion,
        "home_path": LAST_CONFIG.home_path,
        "deleteDatasources": [ds.dict() for ds in LAST_CONFIG.deleteDatasources],
        "datasources": [ds.dict() for ds in LAST_CONFIG.datasources],
    }


def add_datasource(datasource_config: Dict[str, Any]):
    """
    Add a new datasource to the current configuration.
    """
    global LAST_CONFIG
    if LAST_CONFIG is None:
        # Initialize with default config
        LAST_CONFIG = GrafanaConfig()
    
    new_datasource = DataSource(**datasource_config)
    LAST_CONFIG.datasources.append(new_datasource)
    
    # Regenerate the YAML file
    yaml_dict = LAST_CONFIG.to_yaml_dict()
    with open(GRAFANA_DATASOURCE_PATH, "w") as f:
        yaml.safe_dump(yaml_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    logger.info(f"Added new datasource: {new_datasource.name}")
    response_content = {"msg": f"Datasource '{new_datasource.name}' added successfully."}
    return JSONResponse(content=response_content, status_code=200)


def remove_datasource(datasource_name: str):
    """
    Remove a datasource from the current configuration.
    """
    global LAST_CONFIG
    if LAST_CONFIG is None:
        response_content = {"msg": "No configuration loaded."}
        return JSONResponse(content=response_content, status_code=400)
    
    # Remove from datasources list
    original_count = len(LAST_CONFIG.datasources)
    LAST_CONFIG.datasources = [ds for ds in LAST_CONFIG.datasources if ds.name != datasource_name]
    
    if len(LAST_CONFIG.datasources) == original_count:
        response_content = {"msg": f"Datasource '{datasource_name}' not found."}
        return JSONResponse(content=response_content, status_code=404)
    
    # Add to deleteDatasources if not already there
    delete_names = [ds.name for ds in LAST_CONFIG.deleteDatasources]
    if datasource_name not in delete_names:
        LAST_CONFIG.deleteDatasources.append(DeleteDataSource(name=datasource_name))
    
    # Regenerate the YAML file
    yaml_dict = LAST_CONFIG.to_yaml_dict()
    with open(GRAFANA_DATASOURCE_PATH, "w") as f:
        yaml.safe_dump(yaml_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    logger.info(f"Removed datasource: {datasource_name}")
    response_content = {"msg": f"Datasource '{datasource_name}' removed successfully."}
    return JSONResponse(content=response_content, status_code=200)