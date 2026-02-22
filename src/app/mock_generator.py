"""
Mock Data Generator
====================
Generates continuous synthetic sensor data for 5 IoT nodes at ~1 Hz.

Three switchable scenarios:
    1. **normal**    — ambient temperature with Brownian noise
    2. **byzantine** — 2 nodes gradually drift in opposite directions
    3. **fdi**       — 1 node injects sudden impossible values

Data is published to an internal queue consumed by the detector thread.
"""

import math
import queue
import random
import time
import threading
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_NODES = 5
PUBLISH_INTERVAL = 1.0  # seconds between readings per node
AMBIENT_TEMP = 22.0     # °C — baseline
AMBIENT_HUMIDITY = 45.0  # %

# Byzantine scenario parameters
BYZANTINE_DRIFT_RATE = 0.08   # °C per tick for drifting nodes
BYZANTINE_DRIFT_NODES = [2, 4]  # Nodes that will drift (0-indexed: nodes 2 & 4)

# FDI scenario parameters
FDI_TARGET_NODE = 3           # Node that gets injected
FDI_INJECT_INTERVAL = 10     # Inject every N ticks
FDI_INJECT_TEMP = 85.0       # °C — physically impossible indoor temp
FDI_INJECT_ACCEL = 25.0      # m/s² — impossible stationary accel


class MockDataGenerator:
    """
    Generates synthetic IoT sensor data with configurable attack scenarios.

    Usage:
        gen = MockDataGenerator()
        gen.start()

        # Read generated data
        while True:
            data = gen.get_reading(timeout=2.0)
            if data:
                process(data)

        # Switch scenario
        gen.set_scenario("byzantine")
    """

    def __init__(self):
        self._scenario: str = "normal"
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._tick: int = 0
        self._lock = threading.Lock()

        # Per-node state for Brownian walk
        self._node_state = {
            i: {
                "temperature": AMBIENT_TEMP + random.gauss(0, 0.3),
                "humidity": AMBIENT_HUMIDITY + random.gauss(0, 1.5),
                "accel_x": 0.0,
                "accel_y": 0.0,
                "accel_z": 9.81,
            }
            for i in range(1, NUM_NODES + 1)
        }

        # Byzantine drift accumulator
        self._byzantine_drift = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def scenario(self) -> str:
        return self._scenario

    def set_scenario(self, scenario: str):
        """Switch active scenario. Resets internal drift counters."""
        if scenario not in ("normal", "byzantine", "fdi"):
            raise ValueError(f"Unknown scenario: {scenario}")
        with self._lock:
            self._scenario = scenario
            self._byzantine_drift = 0.0
            self._tick = 0
            # Reset node state to baseline
            for i in range(1, NUM_NODES + 1):
                self._node_state[i] = {
                    "temperature": AMBIENT_TEMP + random.gauss(0, 0.3),
                    "humidity": AMBIENT_HUMIDITY + random.gauss(0, 1.5),
                    "accel_x": 0.0,
                    "accel_y": 0.0,
                    "accel_z": 9.81,
                }
        print(f"[MockGen] Scenario switched to: {scenario}")

    def start(self):
        """Start the generator in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[MockGen] Data generator started.")

    def stop(self):
        """Stop the generator."""
        self._running = False

    def get_reading(self, timeout: float = 1.0) -> Optional[dict]:
        """Get next generated reading from the queue."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_queue_size(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Generator Loop
    # ------------------------------------------------------------------

    def _run_loop(self):
        """Main generation loop."""
        while self._running:
            with self._lock:
                scenario = self._scenario
                self._tick += 1
                tick = self._tick

            for node_id in range(1, NUM_NODES + 1):
                reading = self._generate_reading(node_id, scenario, tick)
                try:
                    self._queue.put_nowait(reading)
                except queue.Full:
                    # Drop oldest if full
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(reading)

            time.sleep(PUBLISH_INTERVAL)

    def _generate_reading(
        self, node_id: int, scenario: str, tick: int
    ) -> dict:
        """Generate a single sensor reading based on active scenario."""
        state = self._node_state[node_id]
        ts = time.time()

        if scenario == "normal":
            return self._gen_normal(node_id, state, ts)
        elif scenario == "byzantine":
            return self._gen_byzantine(node_id, state, ts, tick)
        elif scenario == "fdi":
            return self._gen_fdi(node_id, state, ts, tick)
        else:
            return self._gen_normal(node_id, state, ts)

    # ------------------------------------------------------------------
    # Scenario: Normal
    # ------------------------------------------------------------------

    def _gen_normal(self, node_id: int, state: dict, ts: float) -> dict:
        """Normal Brownian-motion readings around ambient."""
        # Temperature: slow random walk
        state["temperature"] += random.gauss(0, 0.02)
        state["temperature"] = max(15.0, min(35.0, state["temperature"]))

        # Humidity: slow walk
        state["humidity"] += random.gauss(0, 0.05)
        state["humidity"] = max(20.0, min(80.0, state["humidity"]))

        # Accelerometer: small jitter
        state["accel_x"] = random.gauss(0, 0.02)
        state["accel_y"] = random.gauss(0, 0.02)
        state["accel_z"] = math.sqrt(
            max(0, 9.81 ** 2 - state["accel_x"] ** 2 - state["accel_y"] ** 2)
        )

        return self._format_reading(node_id, state, ts)

    # ------------------------------------------------------------------
    # Scenario: Byzantine Attack
    # ------------------------------------------------------------------

    def _gen_byzantine(
        self, node_id: int, state: dict, ts: float, tick: int
    ) -> dict:
        """
        Byzantine attack: 2 nodes gradually drift in opposite directions.
        Other nodes remain normal. The drift is slow enough to avoid
        immediate physics violations but causes exponential variance
        growth across horizons.
        """
        if node_id in BYZANTINE_DRIFT_NODES:
            # Accumulate drift (opposite for each Byzantine node)
            direction = 1.0 if node_id == BYZANTINE_DRIFT_NODES[0] else -1.0

            # Exponential-like drift: increases over time
            drift_amount = BYZANTINE_DRIFT_RATE * (1 + tick * 0.01)
            state["temperature"] += direction * drift_amount + random.gauss(0, 0.01)
            state["temperature"] = max(-5.0, min(55.0, state["temperature"]))

            # Also subtly manipulate accelerometer
            state["accel_x"] += direction * 0.005 * random.gauss(1, 0.1)
            state["accel_y"] += direction * 0.003 * random.gauss(1, 0.1)
            xy_sq = state["accel_x"] ** 2 + state["accel_y"] ** 2
            if xy_sq < 9.81 ** 2:
                state["accel_z"] = math.sqrt(9.81 ** 2 - xy_sq)
            else:
                scale = 9.81 / math.sqrt(xy_sq)
                state["accel_x"] *= scale
                state["accel_y"] *= scale
                state["accel_z"] = 0.0
        else:
            # Normal nodes
            state["temperature"] += random.gauss(0, 0.02)
            state["temperature"] = max(15.0, min(35.0, state["temperature"]))
            state["accel_x"] = random.gauss(0, 0.02)
            state["accel_y"] = random.gauss(0, 0.02)
            state["accel_z"] = math.sqrt(
                max(0, 9.81 ** 2 - state["accel_x"] ** 2 - state["accel_y"] ** 2)
            )

        state["humidity"] += random.gauss(0, 0.05)
        state["humidity"] = max(20.0, min(80.0, state["humidity"]))

        return self._format_reading(node_id, state, ts)

    # ------------------------------------------------------------------
    # Scenario: False Data Injection (FDI)
    # ------------------------------------------------------------------

    def _gen_fdi(
        self, node_id: int, state: dict, ts: float, tick: int
    ) -> dict:
        """
        FDI attack: one node periodically injects physically impossible
        values. The injection is sudden (step function), making it
        immediately detectable by t+1 physics checks.
        """
        if node_id == FDI_TARGET_NODE and tick % FDI_INJECT_INTERVAL == 0:
            # Inject impossible values
            fdi_reading = {
                "temperature": FDI_INJECT_TEMP + random.gauss(0, 0.5),
                "humidity": state["humidity"],
                "accel_x": FDI_INJECT_ACCEL * random.choice([-1, 1]),
                "accel_y": random.gauss(0, 2),
                "accel_z": random.gauss(0, 2),
            }
            return self._format_reading(node_id, fdi_reading, ts, injected=True)
        else:
            # Normal readings for all other ticks / nodes
            return self._gen_normal(node_id, state, ts)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_reading(
        node_id: int, state: dict, ts: float, injected: bool = False
    ) -> dict:
        """Format a reading into the standard payload structure."""
        return {
            "node_id": node_id,
            "device_name": f"node_{node_id}",
            "timestamp": ts,
            "readings": {
                "temperature": {
                    "value": round(state["temperature"], 4),
                    "unit": "°C",
                },
                "humidity": {
                    "value": round(state["humidity"], 4),
                    "unit": "%",
                },
                "accel_x": {
                    "value": round(state.get("accel_x", 0.0), 4),
                    "unit": "m/s²",
                },
                "accel_y": {
                    "value": round(state.get("accel_y", 0.0), 4),
                    "unit": "m/s²",
                },
                "accel_z": {
                    "value": round(state.get("accel_z", 9.81), 4),
                    "unit": "m/s²",
                },
            },
            "_meta": {
                "scenario": "fdi_injected" if injected else "generated",
                "tick": 0,  # Will be set by caller context if needed
            },
        }
