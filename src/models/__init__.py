"""
Neural network models for anomaly detection.

Includes:
- GAT_Byzantine_Detector: Graph Attention Network for sensor anomaly detection
- GAT_Config, GAT_Trainer: Configuration and training utilities
- AnomalyScorer: LSTM-like statistical anomaly scorer
"""

from .gat_model import (
    GAT_Config,
    GAT_Byzantine_Detector,
    GAT_Trainer,
    GAT_Evaluator,
)
from .lstm_model import (
    AnomalyScorer,
    SensorBaseline,
    WINDOW_SIZE,
    EWMA_ALPHA,
    MIN_SAMPLES,
    DEFAULT_THRESHOLD,
)

__all__ = [
    "GAT_Config",
    "GAT_Byzantine_Detector",
    "GAT_Trainer",
    "GAT_Evaluator",
    "AnomalyScorer",
    "SensorBaseline",
]
