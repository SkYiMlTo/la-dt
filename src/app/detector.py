"""
LA-DT Anomaly Detector — Pipeline Orchestrator
=================================================
Background thread that:
    1. Consumes readings from MockDataGenerator
    2. Scores each reading via AnomalyScorer (EWMA + z-score)
    3. When anomaly detected → triggers full LA-DT pipeline:
       a. t+1 physics check (simulator /check-physics)
       b. Multi-horizon simulation (simulator /simulate)
       c. Attribution analysis (FDI vs Byzantine vs Natural Drift)
    4. Stores results for API access
"""

import os
import time
import requests
import threading
from collections import deque

from ..models.lstm_model import AnomalyScorer
from ..attribution import AttributionEngine, AttributionResult
from .mock_generator import MockDataGenerator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIMULATOR_URL = os.environ.get("SIMULATOR_URL", "http://simulator:5002")
WEB_BACKEND_URL = os.environ.get("WEB_BACKEND_URL", "http://backend:4000")
NODERED_URL = os.environ.get("NODERED_URL", "http://nodered-dt:1880")

# Detection parameters
ANOMALY_THRESHOLD = 3.5           # z-score threshold
COOLDOWN_SECONDS = 1.0           # Minimum seconds between pipeline triggers
MAX_RESULTS = 100                 # Maximum stored results

# Simulation horizons
HORIZONS = [300, 600, 1800, 3600]  # 5min, 10min, 30min, 1hr
ACCEL_FACTOR = 120
SAMPLE_INTERVAL = 10


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

class DetectorState:
    """Thread-safe container for detector runtime state."""

    def __init__(self):
        self._lock = threading.Lock()
        self.status = "idle"
        self.total_readings = 0
        self.total_anomalies = 0
        self.last_anomaly_time: float | None = None
        self.last_trigger_time: float = 0.0
        self.results: deque[dict] = deque(maxlen=MAX_RESULTS)
        self.recent_scores: deque[dict] = deque(maxlen=50)
        self.active_scenario = "normal"
        self.start_time = time.time()

    def add_result(self, result: AttributionResult):
        with self._lock:
            self.results.appendleft(result.to_dict())
            self.total_anomalies += 1
            self.last_anomaly_time = time.time()

    def add_score(self, score_dict: dict):
        with self._lock:
            self.recent_scores.appendleft(score_dict)

    def get_status(self) -> dict:
        with self._lock:
            uptime = time.time() - self.start_time
            return {
                "status": self.status,
                "scenario": self.active_scenario,
                "total_readings": self.total_readings,
                "total_anomalies": self.total_anomalies,
                "last_anomaly": (
                    time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(self.last_anomaly_time),
                    )
                    if self.last_anomaly_time
                    else None
                ),
                "uptime_seconds": round(uptime, 1),
                "system_health": (
                    "nominal"
                    if self.total_anomalies == 0
                    else "alert" if self.last_anomaly_time
                    and time.time() - self.last_anomaly_time < 60
                    else "monitoring"
                ),
                "sensor_count": 5,
                "queue_depth": _generator.get_queue_size() if _generator else 0,
            }

    def get_results(self) -> list:
        with self._lock:
            return list(self.results)

    def get_result_by_id(self, result_id: str) -> dict | None:
        with self._lock:
            for r in self.results:
                if r["id"] == result_id:
                    return r
            return None


# Singleton instances
_state = DetectorState()
_scorer = AnomalyScorer(threshold=ANOMALY_THRESHOLD)
_attribution = AttributionEngine()
_generator = MockDataGenerator()


# ---------------------------------------------------------------------------
# Public API (called from routes.py)
# ---------------------------------------------------------------------------

def get_detector_state() -> DetectorState:
    return _state


def get_generator() -> MockDataGenerator:
    return _generator


def get_scorer() -> AnomalyScorer:
    return _scorer


def start_detector():
    """
    Entry point — called from __init__.py in a daemon thread.
    Starts the mock data generator and the detector loop.
    """
    _state.status = "starting"
    _generator.start()

    # Give the generator a moment to produce initial data
    time.sleep(2)

    _state.status = "running"
    _state.start_time = time.time()
    print("[Detector] LA-DT pipeline started.")

    while True:
        try:
            _detector_tick()
        except Exception as e:
            print(f"[Detector] Error in tick: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)


# ---------------------------------------------------------------------------
# Detector Loop
# ---------------------------------------------------------------------------

def _detector_tick():
    """Process one reading from the generator."""
    reading = _generator.get_reading(timeout=2.0)
    if reading is None:
        return

    _state.total_readings += 1
    node_id = reading["node_id"]
    ts = reading["timestamp"]

    # Score each sensor in the reading
    anomaly_triggered = False
    trigger_result = None

    for sensor_name, sensor_data in reading["readings"].items():
        value = sensor_data["value"]
        result = _scorer.score(
            node_id=node_id,
            sensor=sensor_name,
            value=value,
            timestamp=ts,
        )

        if result is None:
            continue  # Still bootstrapping

        score_dict = result.to_dict()
        _state.add_score(score_dict)

        if result.is_anomaly and not anomaly_triggered:
            anomaly_triggered = True
            trigger_result = result

    # Check cooldown
    if anomaly_triggered and trigger_result:
        now = time.time()
        # Bypass cooldown for massive anomalies (likely FDI)
        is_critical = trigger_result.z_score > 20.0

        if not is_critical and (now - _state.last_trigger_time < COOLDOWN_SECONDS):
            return  # In cooldown, skip

        _state.last_trigger_time = now
        print(
            f"[Detector] ANOMALY on node {trigger_result.node_id}"
            f" sensor={trigger_result.sensor}"
            f" z={trigger_result.z_score:.2f}"
            f" val={trigger_result.value:.4f}"
        )

        # Run full LA-DT pipeline in a separate thread to avoid blocking
        pipeline_thread = threading.Thread(
            target=_run_ladt_pipeline,
            args=(trigger_result, reading),
            daemon=True,
        )
        pipeline_thread.start()


# ---------------------------------------------------------------------------
# LA-DT Pipeline
# ---------------------------------------------------------------------------

def _run_ladt_pipeline(trigger, reading: dict):
    """
    Full Look-Ahead Digital Twin pipeline:
        1. t+1 physics check
        2. Multi-horizon simulation
        3. Attribution analysis
        4. Store result + notify backend
    """
    print(f"[LA-DT] Starting pipeline for node {trigger.node_id}...")

    # Build state vector from reading for physics check
    r = reading["readings"]
    state_vector = {
        "temperature": r.get("temperature", {}).get("value", 22.0),
        "humidity": r.get("humidity", {}).get("value", 45.0),
        "accel_x": r.get("accel_x", {}).get("value", 0.0),
        "accel_y": r.get("accel_y", {}).get("value", 0.0),
        "accel_z": r.get("accel_z", {}).get("value", 9.81),
    }

    # --- Step 1: t+1 Physics Check ---
    physics_result = {"valid": True, "violations": []}
    try:
        resp = requests.post(
            f"{SIMULATOR_URL}/check-physics",
            json={"node_id": trigger.node_id, "state": state_vector},
            timeout=3,
        )
        if resp.ok:
            physics_result = resp.json()
    except Exception as e:
        print(f"[LA-DT] Physics check failed: {e}")

    # --- Step 2: Multi-Horizon Simulation ---
    simulation_results = {"horizons": {}}
    try:
        resp = requests.post(
            f"{SIMULATOR_URL}/simulate",
            json={
                "horizons": HORIZONS,
                "accel_factor": ACCEL_FACTOR,
                "sample_interval": SAMPLE_INTERVAL,
            },
            timeout=10,
        )
        if resp.ok:
            simulation_results = resp.json()
    except Exception as e:
        print(f"[LA-DT] Simulation failed: {e}")

    # --- Step 3: Attribution ---
    scoring_dict = trigger.to_dict()
    attribution_result = _attribution.analyze(
        scoring_result=scoring_dict,
        physics_check=physics_result,
        simulation_results=simulation_results,
    )

    # Store trajectories (trimmed for storage)
    attribution_result.trajectories = _trim_trajectories(
        simulation_results.get("horizons", {})
    )

    print(
        f"[LA-DT] Pipeline complete → "
        f"verdict={attribution_result.verdict} "
        f"confidence={attribution_result.confidence:.2%}"
    )

    # --- Step 4: Store Result ---
    _state.add_result(attribution_result)

    # --- Step 5: Notify Backend (optional) ---
    _notify_backend(attribution_result)


def _trim_trajectories(horizons: dict, max_points: int = 30) -> dict:
    """Trim trajectory data for storage (keep first/last + evenly spaced)."""
    trimmed = {}
    for h, nodes in horizons.items():
        trimmed[h] = {}
        for nid, data in nodes.items():
            traj = data.get("trajectory", [])
            if len(traj) <= max_points:
                trimmed[h][nid] = traj
            else:
                step = len(traj) // max_points
                trimmed[h][nid] = [traj[i] for i in range(0, len(traj), step)]
    return trimmed


def _notify_backend(result: AttributionResult):
    """Send alert to the web backend for dashboard display."""
    try:
        alert_payload = {
            "device_id": f"device-node_{result.trigger_node}",
            "alert_level": (
                "danger" if result.verdict in ("fdi", "byzantine") else "warning"
            ),
            "alert_type": result._verdict_label(),
            "alert_message": (
                f"Anomaly detected on node {result.trigger_node} "
                f"({result.trigger_sensor}): "
                f"z-score={result.trigger_z_score:.2f}, "
                f"verdict={result._verdict_label()} "
                f"(confidence: {result.confidence:.0%})"
            ),
            "timestamp": result.timestamp,
        }
        requests.post(
            f"{WEB_BACKEND_URL}/network/add_alert",
            json=alert_payload,
            timeout=2,
        )
    except Exception as e:
        print(f"[LA-DT] Backend notification failed: {e}")


# ---------------------------------------------------------------------------
# Legacy API — analyze_payload (backward compat for Node-RED)
# ---------------------------------------------------------------------------

def analyze_payload(data: dict):
    """
    Legacy endpoint handler. Converts Node-RED push-style data into
    the new scoring pipeline format.
    """
    device_id = data.get("device_id", "")
    kind = data.get("kind_of_data", "reading")

    if kind != "reading":
        return  # Only score readings

    values = data.get("values", [])
    try:
        # Extract numeric node index
        import re
        m = re.search(r"(\d+)$", str(device_id))
        node_idx = int(m.group(1)) if m else 1
    except Exception:
        node_idx = 1

    for v in values:
        sensor_type = v.get("sensor_type", "unknown")
        value = v.get("value")
        if value is not None:
            _scorer.score(
                node_id=node_idx,
                sensor=sensor_type,
                value=float(value),
            )
