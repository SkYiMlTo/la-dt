"""
Utils module - Shared utilities for experiments.

Submodules:
- utilities: GAT training, evaluation, and metrics
- attack_data_generator: Byzantine attack data generation
- table_generator: LaTeX table generation for results

Attribution pipeline has been moved to src.attribution.
"""

from .utilities import (
    custom_collate_fn,
    compute_metrics,
    compute_attribution_accuracy,
    train_gat_model,
    evaluate_gat_on_data,
    run_attribution_at_horizon,
)
from .attack_data_generator import AttackDataGenerator

__all__ = [
    "custom_collate_fn",
    "compute_metrics",
    "compute_attribution_accuracy",
    "train_gat_model",
    "evaluate_gat_on_data",
    "run_attribution_at_horizon",
    "AttackDataGenerator",
]
