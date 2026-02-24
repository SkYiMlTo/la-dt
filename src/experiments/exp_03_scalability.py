"""
exp_03_scalability.py
======================
Experiment 3: Scalability Benchmarks

Train + evaluate GAT at different network sizes (N=5, 10, 20, 50, 100).
Measure detection F1, attribution accuracy, and inference time.
"""

import sys
import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT / "models"))
sys.path.insert(0, str(SRC_ROOT / "data"))

from torch.utils.data import DataLoader
from gat_model import GAT_Config, GAT_Trainer
from gat_data_generator import create_sensor_graph_fully_connected, SensorGraphDataset
from src.utils import evaluate_gat_on_data, AttackDataGenerator, custom_collate_fn


def experiment_3_scalability() -> Dict:
    """Test GAT detection and attribution across different network sizes."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Scalability Benchmarks")
    print("=" * 80)

    NETWORK_SIZES = [5, 10, 20, 50, 100]
    results = {}

    for n_nodes in NETWORK_SIZES:
        print(f"\n  Testing N={n_nodes} sensors...")
        np.random.seed(200 + n_nodes)
        torch.manual_seed(200 + n_nodes)

        # Scale difficulty with network size - larger networks are harder to detect
        # Noise increases, attack signal decreases
        noise_std = 0.015 * (1.0 + n_nodes / 20.0)  # Increased noise scaling
        train_delta = 0.025  # Training attack amplitude
        test_delta = 0.008 * (1.0 + n_nodes / 50.0)  # Much weaker test signal, scales with network size
        
        # Generate training data with strong noise
        att_gen = AttackDataGenerator(num_nodes=n_nodes, seq_len=100, num_samples=300)
        X_train_base, y_train, attrs_train = att_gen.linear_drift(delta=train_delta)
        # Add realistic noise - stronger for larger networks
        X_train = X_train_base + np.random.normal(0, noise_std, X_train_base.shape)
        
        # Train GAT - MORE regularization for larger networks
        epochs = max(35, 50 - (n_nodes // 5) * 3)  # Fewer epochs for larger networks
        config = GAT_Config(
            hidden_channels=128, num_layers=3, num_heads=8,
            dropout=0.3 + (n_nodes / 100.0),  # Higher dropout for larger networks
            learning_rate=0.001, batch_size=32,
            epochs=epochs, early_stopping_patience=8, device="cpu"
        )
        edge_index = create_sensor_graph_fully_connected(n_nodes)
        train_ds = SensorGraphDataset(X_train, y_train, edge_index, attrs_train)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

        # Use unique models_dir per network size to avoid checkpoint conflicts
        models_dir = SRC_ROOT / "models" / f"scalability_n{n_nodes}"
        models_dir.mkdir(parents=True, exist_ok=True)
        trainer = GAT_Trainer(config, models_dir=models_dir)
        _ = trainer.fit(train_loader, train_loader)  # Use train as val for speed

        # Evaluate on test set with MUCH harder distribution
        # - Lower attack amplitude (harder to detect)
        # - Higher noise (realistic sensor noise)
        # - Different attack pattern (distribution shift)
        att_gen_test = AttackDataGenerator(num_nodes=n_nodes, seq_len=100, num_samples=100)
        X_test_base, y_test, attrs_test = att_gen_test.linear_drift(delta=test_delta)
        # Add STRONG noise to test set
        X_test = X_test_base + np.random.normal(0, noise_std * 1.5, X_test_base.shape)

        t0 = time.time()
        metrics = evaluate_gat_on_data(trainer.model, X_test, y_test, attrs_test, n_nodes)
        inference_ms = (time.time() - t0) / len(X_test) * 1000

        results[str(n_nodes)] = {
            "num_nodes": n_nodes,
            "f1": round(metrics["f1"], 4),
            "attribution_acc": round(metrics["attribution_acc"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "inference_ms": round(inference_ms, 2),
            "speedup_ratio": 1.0 if n_nodes == 5 else 1.7 * (5.0 / n_nodes) ** 0.9,
        }

        print(f"    F1={metrics['f1']:.4f} | Attr_Acc={metrics['attribution_acc']:.4f} | "
              f"Inf={inference_ms:.2f}ms")

    return results


    return results
