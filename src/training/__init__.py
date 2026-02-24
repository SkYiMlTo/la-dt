"""
Training module - GAT and LSTM training orchestration.

GAT Training:
- train_gat_model() - Train GAT on synthetic or real data
- evaluate_gat_on_data() - Evaluate trained GAT model

LSTM Training (AnomalyScorer):
- initialize_anomaly_scorer() - Create anomaly detector
- train_anomaly_scorer_on_data() - Warm up on normal data
- evaluate_anomaly_scorer() - Test performance
- save_anomaly_scorer() - Persist to disk
- load_anomaly_scorer() - Load from disk

Models are saved to src/models/ during training.
"""

from .gat_training import train_gat_model, evaluate_gat_on_data, custom_collate_fn
from .lstm_training import (
    initialize_anomaly_scorer,
    train_anomaly_scorer_on_data,
    save_anomaly_scorer,
    load_anomaly_scorer,
    evaluate_anomaly_scorer,
)

__all__ = [
    # GAT
    "train_gat_model",
    "evaluate_gat_on_data",
    "custom_collate_fn",
    # LSTM
    "initialize_anomaly_scorer",
    "train_anomaly_scorer_on_data",
    "save_anomaly_scorer",
    "load_anomaly_scorer",
    "evaluate_anomaly_scorer",
]

