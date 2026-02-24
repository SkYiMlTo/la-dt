"""
exp_01_attack_robustness.py
============================
Experiment 1: Byzantine Attack Robustness Matrix

Evaluates GAT detection accuracy across 6 Byzantine attack classes
(S1-S6). Tests both detectable (S1-S4), stealthy (S5) and fundamental limit (S6) attack patterns.

Outputs:
  - Detection F1 scores per attack class (mean ± std over seeds)
  - Performance table ready for publication
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict
from collections import defaultdict

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT / "models"))
sys.path.insert(0, str(SRC_ROOT / "data"))
sys.path.insert(0, str(SRC_ROOT))

from gat_data_generator import create_sensor_graph_fully_connected, SensorGraphDataset
from torch.utils.data import DataLoader
import torch

# Import shared utilities
from src.utils import (
    AttackDataGenerator,
    train_gat_model,
    evaluate_gat_on_data,
    custom_collate_fn,
    compute_metrics,
)


def experiment_1_attack_robustness(num_seeds: int = 5) -> Dict:
    """
    Test robustness against 6 Byzantine attack classes.
    
    Trains GAT on 6 attack types (S1-S6), evaluates on all 6. Uses mixed-difficulty test sets to simulate
    realistic attack spectrum.
    
    Args:
        num_seeds: Number of independent random seeds to run (for std dev)
    
    Returns:
        Dictionary with F1 scores per attack class:
        {
            "S1": {"f1": 0.941, "std": 0.043, ...},
            "S2": {...},
            ...
        }
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Byzantine Attack Robustness Matrix")
    print("=" * 80)

    NUM_NODES = 5
    num_samples = 400

    attack_classes = {
        "S1": ("Linear Drift [DETECTABLE]", lambda ag: ag.linear_drift(delta=0.02)),
        "S2": ("Exponential Surge [DETECTABLE]", lambda ag: ag.exponential_drift(delta=0.02, alpha=3.0)),
        "S3": ("Wave Oscillation [DETECTABLE]", lambda ag: ag.frogging_attack(delta=0.02, switch_period=4)),
        "S4": ("FDI Step Change [DETECTABLE]", lambda ag: ag.fdi_step_change(magnitude=2.0)),
        "S5": ("Persistent Bias [STEALTHY]", lambda ag: ag.polynomial_drift(delta=0.015, power=2.0)),
        "S6": ("All Nodes Compromised [FUNDAMENTAL LIMIT]", lambda ag: ag.majority_compromised(delta=0.002)),
    }

    # Run multiple times with different seeds to get mean ± std
    seed_results = defaultdict(list)

    for seed_idx in range(num_seeds):
        np.random.seed(100 + seed_idx)
        torch.manual_seed(100 + seed_idx)

        print(f"\n  [Seed {seed_idx + 1}/{num_seeds}]")
        
        # Train GAT on training set (S1-S6 mixed)
        model, history = train_gat_model(
            num_nodes=NUM_NODES,
            seq_len=100,
            num_samples=num_samples,
            epochs=30,
            batch_size=16,
            verbose=False,
            diverse_attacks=True
        )

        # Evaluate on all 6 attack classes
        for attack_id, (attack_name, attack_fn) in attack_classes.items():
            att_gen = AttackDataGenerator(num_nodes=NUM_NODES, seq_len=100, num_samples=200)
            X_test, y_test, attrs_test = attack_fn(att_gen)
            
            metrics = evaluate_gat_on_data(
                model, X_test, y_test, attrs_test, 
                num_nodes=NUM_NODES, batch_size=32
            )
            
            seed_results[attack_id].append(metrics["f1"])
            print(f"    {attack_id}: {attack_name:40s} | F1={metrics['f1']:.4f}")

    # Aggregate results across seeds
    summary = {}
    print("\n  SUMMARY (Mean ± Std over {} seeds):".format(num_seeds))
    for attack_id in sorted(attack_classes.keys()):
        f1_scores = seed_results[attack_id]
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        summary[attack_id] = {
            "f1": round(mean_f1, 4),
            "std": round(std_f1, 4),
            "name": attack_classes[attack_id][0],
        }
        print(f"    {attack_id}: {mean_f1:.4f} ± {std_f1:.4f}")

    return summary
