"""
exp_05_ai_dataset.py
=====================
Experiment 5: Cross-Domain AI Dataset Validation

Train and evaluate on power generation synchrophasor data.
Tests cross-domain generalization.
"""

import sys
import numpy as np
import pandas as pd
import torch
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT / "models"))
sys.path.insert(0, str(SRC_ROOT / "data"))

from gat_model import GAT_Config, GAT_Trainer
from gat_data_generator import create_sensor_graph_fully_connected, SensorGraphDataset
from torch.utils.data import DataLoader
from src.utils import evaluate_gat_on_data, custom_collate_fn


def experiment_5_ai_dataset() -> Dict:
    """Validate GAT on cross-domain AI power generation data."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Cross-Domain AI Dataset Validation")
    print("=" * 80)

    ai_path = PROJECT_ROOT / "src" / "data" / "raw" / "ai-data" / "scaled_PV_data.csv"
    if not ai_path.exists():
        print("  [SKIP] AI dataset not found")
        return {"status": "skipped", "reason": "AI PV CSV not found"}

    print("  Loading AI PV data (first 10K  rows)...")
    df = pd.read_csv(ai_path, nrows=10000, header=None)
    data = df.values.astype(np.float32)
    num_features = min(51, data.shape[1])
    data = data[:, :num_features]
    print(f"  Data shape: {data.shape}, using {num_features} features")

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    data = (data - mean) / std

    normal_end = int(0.7 * len(data))
    window_size = 100
    stride = 50

    X_windows, y_windows, attrs_windows = [], [], []
    for i in range(0, len(data) - window_size, stride):
        w = data[i:i + window_size]
        if len(w) == window_size:
            X_windows.append(w.T)
            is_attacked = (i >= normal_end)
            y_windows.append(1 if is_attacked else 0)
            
            # Inject drift/attack in the latter half of data
            if is_attacked:
                w_attacked = w.copy().T
                targets = np.random.choice(num_features, size=max(1, num_features//5), replace=False)
                t_arr = np.arange(w.shape[0], dtype=np.float64) / w.shape[0]
                drift = 0.05 * t_arr * np.random.choice([-1, 1])
                for t_idx in targets:
                    w_attacked[t_idx] += drift
                attr = np.zeros(num_features, dtype=np.float32)
                attr[targets] = 1.0
                attrs_windows.append(attr)
            else:
                attrs_windows.append(np.zeros(num_features, dtype=np.float32))

    X_all = np.array(X_windows)
    y_all = np.array(y_windows)
    attrs_all = np.array(attrs_windows)

    n0 = np.sum(y_all == 0)
    n1 = np.sum(y_all == 1)
    n_min = min(n0, n1)
    idx0 = np.where(y_all == 0)[0][:n_min]
    idx1 = np.where(y_all == 1)[0][:n_min]
    idx = np.concatenate([idx0, idx1])
    np.random.shuffle(idx)
    X_all, y_all, attrs_all = X_all[idx], y_all[idx], attrs_all[idx]

    split = int(0.8 * len(X_all))
    config = GAT_Config(
        hidden_channels=128, num_layers=3, num_heads=8,
        dropout=0.2, learning_rate=0.001, batch_size=32,
        epochs=100, early_stopping_patience=15, device="cpu",
    )
    edge_index = create_sensor_graph_fully_connected(num_features)

    train_ds = SensorGraphDataset(X_all[:split], y_all[:split], edge_index, attrs_all[:split])
    val_ds = SensorGraphDataset(X_all[split:], y_all[split:], edge_index, attrs_all[split:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    print("  Training GAT on AI dataset...")
    t0 = time.time()
    models_dir = SRC_ROOT / "models"
    trainer = GAT_Trainer(config, models_dir=models_dir)
    history = trainer.fit(train_loader, val_loader)
    train_time = time.time() - t0

    metrics = evaluate_gat_on_data(trainer.model, X_all[split:], y_all[split:], 
                                    attrs_all[split:], num_features)
    metrics["train_time_s"] = round(train_time, 2)
    metrics["total_samples"] = len(X_all)
    metrics["num_sensors"] = num_features

    print(f"  AI Dataset: F1={metrics['f1']:.3f} | Acc={metrics['accuracy']:.3f} | "
          f"Time={train_time:.1f}s | Samples={len(X_all)}")

    return metrics
