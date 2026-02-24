"""
LA-DT Experiments Module

Individual experiment files organized by research question:
- exp_01_attack_robustness.py ~  RQ1: Byzantine attack robustness
- exp_02_multi_horizon.py      ~  RQ2: Attribution at multiple horizons
- exp_03_scalability.py        ~  RQ3: Scalability benchmarks
- exp_04_swat_validation.py    ~  RQ4: Real-world SWAT validation
- exp_05_ai_dataset.py         ~  RQ5: Cross-domain AI dataset
- exp_06_ablation.py           ~  RQ6: Ablation study
- exp_07_swat_attribution.py   ~  RQ7: SWAT attribution analysis

Run all experiments with:
    python -m src.experiments.main_run_all_experiments
"""

from .exp_01_attack_robustness import experiment_1_attack_robustness
from .exp_02_multi_horizon import experiment_2_multi_horizon
from .exp_03_scalability import experiment_3_scalability
from .exp_04_swat_validation import experiment_4_swat_validation
from .exp_05_ai_dataset import experiment_5_ai_dataset
from .exp_06_ablation import experiment_6_ablation
from .exp_07_swat_attribution import experiment_7_swat_attribution
from src.utils import (
    AttackDataGenerator,
    train_gat_model,
    evaluate_gat_on_data,
    run_attribution_at_horizon,
    compute_metrics,
    compute_attribution_accuracy,
    custom_collate_fn,
)

__all__ = [
    "experiment_1_attack_robustness",
    "experiment_2_multi_horizon",
    "experiment_3_scalability",
    "experiment_4_swat_validation",
    "experiment_5_ai_dataset",
    "experiment_6_ablation",
    "experiment_7_swat_attribution",
    "AttackDataGenerator",
    "train_gat_model",
    "evaluate_gat_on_data",
    "run_attribution_at_horizon",
    "compute_metrics",
    "compute_attribution_accuracy",
    "custom_collate_fn",
]
