"""
LSTM-Like Anomaly Scorer
=========================
Statistical anomaly detection using Exponentially Weighted Moving Average (EWMA)
and z-score. Designed to be swapped for a real trained LSTM model when physical
node training data becomes available.

Architecture:
    1. Maintains a sliding window per (node_id, sensor_type)
    2. Computes EWMA as the "predicted next value"
    3. Anomaly score = |actual - predicted| / rolling_std
    4. Score > threshold → anomaly triggered
"""

import math
import time
import pickle
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_SIZE = 60          # Number of samples to maintain per sensor
EWMA_ALPHA = 0.3          # Smoothing factor (higher = more reactive)
MIN_SAMPLES = 10          # Minimum samples before scoring
DEFAULT_THRESHOLD = 4.0   # z-score threshold for anomaly
MIN_STD = 0.1             # Floor to prevent division by zero


@dataclass
class SensorBaseline:
    """Tracks running statistics for a single (node, sensor) pair."""
    window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    ewma: float = 0.0
    ewma_var: float = 0.0
    initialized: bool = False
    last_update: float = 0.0

    @property
    def std(self) -> float:
        return max(math.sqrt(self.ewma_var), MIN_STD)

    @property
    def sample_count(self) -> int:
        return len(self.window)


class AnomalyScorer:
    """
    Multi-sensor anomaly scorer using EWMA + z-score.

    Usage:
        scorer = AnomalyScorer(threshold=3.0)

        # Feed readings continuously
        result = scorer.score(node_id=1, sensor="temperature", value=22.5)
        if result and result.is_anomaly:
            trigger_ladt_pipeline(result)
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        # key: (node_id, sensor_type) -> SensorBaseline
        self._baselines: dict[tuple, SensorBaseline] = {}

    def _get_baseline(self, node_id: int, sensor: str) -> SensorBaseline:
        key = (node_id, sensor)
        if key not in self._baselines:
            self._baselines[key] = SensorBaseline()
        return self._baselines[key]

    def score(
        self,
        node_id: int,
        sensor: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> Optional["ScoringResult"]:
        """
        Ingest a new reading and compute anomaly score.

        Returns None if not enough data yet, otherwise a ScoringResult.
        """
        ts = timestamp or time.time()
        bl = self._get_baseline(node_id, sensor)
        bl.window.append(value)
        bl.last_update = ts

        if not bl.initialized:
            if bl.sample_count < MIN_SAMPLES:
                # Bootstrap: just accumulate
                bl.ewma = sum(bl.window) / len(bl.window)
                return None
            # Initialize EWMA from accumulated data
            bl.ewma = sum(bl.window) / len(bl.window)
            bl.ewma_var = sum((v - bl.ewma) ** 2 for v in bl.window) / len(bl.window)
            bl.initialized = True
            return None

        # Predict: EWMA is our "predicted next value"
        predicted = bl.ewma

        # Actual deviation
        error = abs(value - predicted)
        z_score = error / bl.std

        # Update EWMA (exponential smoothing)
        bl.ewma = EWMA_ALPHA * value + (1 - EWMA_ALPHA) * bl.ewma
        bl.ewma_var = (
            EWMA_ALPHA * (value - bl.ewma) ** 2
            + (1 - EWMA_ALPHA) * bl.ewma_var
        )

        is_anomaly = z_score > self.threshold

        return ScoringResult(
            node_id=node_id,
            sensor=sensor,
            value=value,
            predicted=predicted,
            error=error,
            z_score=z_score,
            threshold=self.threshold,
            is_anomaly=is_anomaly,
            baseline_std=bl.std,
            sample_count=bl.sample_count,
            timestamp=ts,
        )

    def get_all_baselines_summary(self) -> dict:
        """Return a summary of all tracked baselines for debugging."""
        summary = {}
        for (nid, sensor), bl in self._baselines.items():
            summary[f"node_{nid}_{sensor}"] = {
                "ewma": round(bl.ewma, 4),
                "std": round(bl.std, 4),
                "samples": bl.sample_count,
                "initialized": bl.initialized,
            }
        return summary

    def reset(self):
        """Clear all baselines (used when switching scenarios)."""
        self._baselines.clear()
    
    def save(self, path: Path) -> Path:
        """
        Save the anomaly scorer state to disk.
        
        Args:
            path: Path to save the scorer (.pkl file)
        
        Returns:
            Path where the scorer was saved
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return path
    
    @staticmethod
    def load(path: Path) -> "AnomalyScorer":
        """
        Load an anomaly scorer state from disk.
        
        Args:
            path: Path to the saved scorer (.pkl file)
        
        Returns:
            Loaded AnomalyScorer instance with previous state
        """
        path = Path(path)
        with open(path, 'rb') as f:
            return pickle.load(f)


@dataclass
class ScoringResult:
    """Result of scoring a single sensor reading."""
    node_id: int
    sensor: str
    value: float
    predicted: float
    error: float
    z_score: float
    threshold: float
    is_anomaly: bool
    baseline_std: float
    sample_count: int
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "sensor": self.sensor,
            "value": round(self.value, 4),
            "predicted": round(self.predicted, 4),
            "error": round(self.error, 4),
            "z_score": round(self.z_score, 4),
            "threshold": self.threshold,
            "is_anomaly": self.is_anomaly,
            "baseline_std": round(self.baseline_std, 4),
            "sample_count": self.sample_count,
            "timestamp": self.timestamp,
        }
