import os
import yaml
import signal
import logging
import subprocess
import threading
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from fastapi.responses import JSONResponse

logger = logging.getLogger("prometheus-manager")

PROMETHEUS_CONFIG_FILE = "prometheus.yml"
PROMETHEUS_PROCESS: Optional[subprocess.Popen] = None
PROMETHEUS_THREAD: Optional[threading.Thread] = None
PROMETHEUS_LOCK = threading.Lock()
LAST_CONFIG: Optional["PrometheusConfig"] = None  # updated by config_prometheus()


class PrometheusConfig(BaseModel):
    # Flat logical model
    scrape_interval: str = "5s"
    evaluation_interval: str = "5s"
    external_labels: Dict[str, Any] = Field(default_factory=lambda: {"monitor": "example"})
    alertmanagers: List[Dict[str, Any]] = Field(
        default_factory=lambda: [{"static_configs": [{"targets": ["localhost:9093"]}]}]
    )
    rule_files: List[str] = Field(default_factory=list)
    scrape_configs: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "job_name": "prometheus",
                "scrape_interval": "5s",
                "scrape_timeout": "5s",
                "static_configs": [{"targets": ["localhost:9090"]}],
            },
            {
                "job_name": "system_metrics",
                "static_configs": [{"targets": ["localhost:8000"]}],
            },
        ]
    )
    # Not a YAML field: used as CLI flag, but we persist it along with the config
    storage_tsdb_path: str = "PrometheusLogs/"

    @validator("scrape_interval", "evaluation_interval", "storage_tsdb_path")
    def _non_empty(cls, v: str) -> str:
        assert isinstance(v, str) and len(v) > 0
        return v

    # --------------------------
    # Constructor for hierarchical or flat payloads
    # --------------------------
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "PrometheusConfig":
        """
        Accepts either:
          - Flat JSON:
              {
                "scrape_interval": "...",
                "evaluation_interval": "...",
                "external_labels": {...},
                "alertmanagers": [...],
                "rule_files": [...],
                "scrape_configs": [...],
                "storage_tsdb_path": "..."
              }
          - Hierarchical JSON/YAML-shaped:
              {
                "global": {
                  "scrape_interval": "...",
                  "evaluation_interval": "...",
                  "external_labels": {...}
                },
                "alerting": { "alertmanagers": [...] },
                "rule_files": [...],
                "scrape_configs": [...],
                "storage.tsdb.path": "..."  # or nested {"storage": {"tsdb": {"path": "..."}}}
              }
        Missing fields fall back to defaults.
        """
        payload = payload or {}
        g = payload.get("global", {}) or {}

        # Get value with fallbacks: flat key → nested → default
        def get_with_fallback(flat_key: str, *paths, default=None):
            if flat_key in payload:
                return payload[flat_key]
            # Support dash/camel variants if needed
            alt_keys = {flat_key, flat_key.replace("-", "_"), flat_key.replace("_", "-")}
            for k in alt_keys:
                if k in payload:
                    return payload[k]
            # Nested paths
            for p in paths:
                cur = payload
                ok = True
                for key in p:
                    if isinstance(cur, dict) and key in cur:
                        cur = cur[key]
                    else:
                        ok = False
                        break
                if ok:
                    return cur
            return default

        scrape_interval = get_with_fallback(
            "scrape_interval",
            ("global", "scrape_interval"),
            default="5s",
        )
        evaluation_interval = get_with_fallback(
            "evaluation_interval",
            ("global", "evaluation_interval"),
            default="5s",
        )
        external_labels = get_with_fallback(
            "external_labels",
            ("global", "external_labels"),
            default={"monitor": "example"},
        )

        alertmanagers = get_with_fallback(
            "alertmanagers",
            ("alerting", "alertmanagers"),
            default=[{"static_configs": [{"targets": ["localhost:9093"]}]}],
        )

        rule_files = payload.get("rule_files", payload.get("ruleFiles", [])) or []
        scrape_configs = payload.get("scrape_configs", payload.get("scrapeConfigs", [])) or [
            {
                "job_name": "prometheus",
                "scrape_interval": "5s",
                "scrape_timeout": "5s",
                "static_configs": [{"targets": ["localhost:9090"]}],
            },
            {
                "job_name": "system_metrics",
                "static_configs": [{"targets": ["localhost:8000"]}],
            },
        ]

        # storage.tsdb.path detection (multiple styles)
        storage_tsdb_path = (
            payload.get("storage_tsdb_path")
            or payload.get("storage.tsdb.path")
            or (payload.get("storage", {}).get("tsdb", {}).get("path") if isinstance(payload.get("storage"), dict) else None)
            or "PrometheusLogs/"
        )

        # Allow hierarchical 'global' overrides when flat keys missing
        if isinstance(g, dict):
            scrape_interval = payload.get("scrape_interval", g.get("scrape_interval", scrape_interval))
            evaluation_interval = payload.get("evaluation_interval", g.get("evaluation_interval", evaluation_interval))
            external_labels = payload.get("external_labels", g.get("external_labels", external_labels))

        return cls(
            scrape_interval=scrape_interval,
            evaluation_interval=evaluation_interval,
            external_labels=external_labels,
            alertmanagers=alertmanagers,
            rule_files=rule_files,
            scrape_configs=scrape_configs,
            storage_tsdb_path=storage_tsdb_path,
        )

    # --------------------------
    # YAML rendering
    # --------------------------
    def to_yaml_dict(self) -> Dict[str, Any]:
        """Shape as Prometheus expects in YAML."""
        return {
            "global": {
                "scrape_interval": self.scrape_interval,
                "evaluation_interval": self.evaluation_interval,
                "external_labels": self.external_labels,
            },
            "alerting": {"alertmanagers": self.alertmanagers},
            "rule_files": self.rule_files,
            "scrape_configs": self.scrape_configs,
        }


# --------------------------
# Public functions used by your FastAPI
# --------------------------
def config_prometheus(config_payload: Dict[str, Any]):
    """
    Build config from (possibly hierarchical) payload and write prometheus.yml.
    Also tracks LAST_CONFIG for runtime flags (e.g., storage.tsdb.path).
    """
    global LAST_CONFIG
    cfg = PrometheusConfig.from_payload(config_payload)
    yaml_dict = cfg.to_yaml_dict()

    # Ensure TSDB dir exists (not part of YAML but used at start)
    try:
        os.makedirs(cfg.storage_tsdb_path, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not ensure TSDB path exists ({cfg.storage_tsdb_path}): {e}")

    with open(PROMETHEUS_CONFIG_FILE, "w") as f:
        yaml.safe_dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    LAST_CONFIG = cfg
    logger.info(f"Prometheus configuration written to {PROMETHEUS_CONFIG_FILE}")
    response_content = {"msg": f"Config written to {PROMETHEUS_CONFIG_FILE}"}
    return JSONResponse(content=response_content, status_code=200)


def start_prometheus():
    """
    Start Prometheus as a background thread using the last configured TSDB path.
    """
    global PROMETHEUS_PROCESS, PROMETHEUS_THREAD, LAST_CONFIG

    with PROMETHEUS_LOCK:
        if PROMETHEUS_PROCESS and PROMETHEUS_PROCESS.poll() is None:
            return {"status_code": 304, "msg": "Prometheus already running."}

        tsdb_path = (LAST_CONFIG.storage_tsdb_path if LAST_CONFIG else PrometheusConfig().storage_tsdb_path)
        os.makedirs(tsdb_path, exist_ok=True)

        def run_prometheus():
            global PROMETHEUS_PROCESS
            cmd = [
                "prometheus",
                f"--config.file={PROMETHEUS_CONFIG_FILE}",
                f"--storage.tsdb.path={tsdb_path}",
            ]
            logger.info("Starting Prometheus: %s", " ".join(cmd))
            PROMETHEUS_PROCESS = subprocess.Popen(cmd)
            rc = PROMETHEUS_PROCESS.wait()
            logger.info("Prometheus process ended with code %s.", rc)

        PROMETHEUS_THREAD = threading.Thread(target=run_prometheus, daemon=True)
        PROMETHEUS_THREAD.start()
        response_content =  {"msg": "Prometheus started."}
        return JSONResponse(content=response_content, status_code=200)


def check_prometheus():
    pid = None
    running = None
    error = None
    with PROMETHEUS_LOCK:
        if PROMETHEUS_PROCESS is None:
            running = False
        elif PROMETHEUS_PROCESS.poll() is None:
            running = True
            pid = PROMETHEUS_PROCESS.pid
        else:
            running = False
            error = PROMETHEUS_PROCESS.returncode
            
        return running, pid, error


def stop_prometheus():
    global PROMETHEUS_PROCESS
    with PROMETHEUS_LOCK:
        if PROMETHEUS_PROCESS is None or PROMETHEUS_PROCESS.poll() is not None:
            response_content = {"msg": "Prometheus is not running."}
            return JSONResponse(content=response_content, status_code=202)
        pid = PROMETHEUS_PROCESS.pid
        logger.info(f"Stopping Prometheus (PID={pid})")
        PROMETHEUS_PROCESS.send_signal(signal.SIGTERM)
        PROMETHEUS_PROCESS.wait(timeout=10)
        logger.info("Prometheus stopped.")
        response_content = {"msg": "Prometheus stopped.", "pid": pid}
        return JSONResponse(content=response_content, status_code=200)