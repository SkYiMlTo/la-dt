"""
exp_04_swat_validation.py
==========================
Experiment 4: SWAT Real-World Validation

Train and evaluate on actual SWAT water treatment ICS data.
Reproduces Table 7 from the paper.
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


def experiment_4_swat_validation() -> Dict:
    """Validate GAT detection on real SWAT water treatment data."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: SWAT Real-World Validation")
    print("=" * 80)

    swat_path = PROJECT_ROOT / "src" / "data" / "raw" / "swat" / "normal.csv"
    if not swat_path.exists():
        print("  [SKIP] SWAT data not found at", swat_path)
        return {"status": "skipped", "reason": "SWAT CSV not found"}

    print("  Loading SWAT normal data (100000 rows)...")
    df_normal = pd.read_csv(swat_path, nrows=100000)
    # Skip Timestamp (col 0) and Normal/Attack label (last col), keep only sensor data
    data_normal = df_normal.iloc[:, 1:-1].values.astype(np.float32)
    num_sensors = min(51, data_normal.shape[1])
    data_normal = data_normal[:, :num_sensors]

    print("  Loading SWAT attack data (100000 rows)...")
    attack_path = PROJECT_ROOT / "src" / "data" / "raw" / "swat" / "attack.csv"
    df_attack = pd.read_csv(attack_path, nrows=100000)
    data_attack = df_attack.iloc[:, 1:-1].values.astype(np.float32)
    data_attack = data_attack[:, :num_sensors]

    # Normalize using normal data statistics  
    mean_n = np.mean(data_normal, axis=0)
    std_n = np.std(data_normal, axis=0)
    std_n[std_n == 0] = 1
    data_normal = (data_normal - mean_n) / std_n
    data_attack = (data_attack - mean_n) / std_n

    window_size = 100
    stride = 50
    X_windows, y_windows = [], []
    
    # Add all normal windows (label=0)
    for i in range(0, len(data_normal) - window_size, stride):
        w = data_normal[i:i + window_size]
        if len(w) == window_size:
            X_windows.append(w.T)
            y_windows.append(0)
    
    # Add all attack windows (label=1)
    for i in range(0, len(data_attack) - window_size, stride):
        w = data_attack[i:i + window_size]
        if len(w) == window_size:
            X_windows.append(w.T)
            y_windows.append(1)

    X_all = np.array(X_windows)
    y_all = np.array(y_windows)
    attrs_all = np.zeros((len(X_all), num_sensors), dtype=np.float32)

    # Randomize order
    np.random.seed(42)
    idx = np.random.permutation(len(X_all))
    X_all = X_all[idx]
    y_all = y_all[idx]
    attrs_all = attrs_all[idx]

    if len(X_all) < 10:
        print("  [SKIP] Insufficient SWAT data")
        return {"status": "skipped", "reason": "Insufficient data samples"}

    # Standard 80/20 split
    split = int(0.8 * len(X_all))
    X_train = X_all[:split]
    y_train = y_all[:split]
    X_test = X_all[split:]
    y_test = y_all[split:]
    attrs_train = attrs_all[:split]
    attrs_test = attrs_all[split:]

    # Further split training into train/val for early stopping
    val_split = int(0.8 * len(X_train))
    X_train_tr = X_train[:val_split]
    y_train_tr = y_train[:val_split]
    X_train_val = X_train[val_split:]
    y_train_val = y_train[val_split:]
    attrs_train_tr = attrs_train[:val_split]
    attrs_train_val = attrs_train[val_split:]

    config = GAT_Config(
        hidden_channels=128, num_layers=3, num_heads=8,
        dropout=0.2, learning_rate=0.001, batch_size=32,
        epochs=100, early_stopping_patience=15, device="cpu",
    )
    edge_index = create_sensor_graph_fully_connected(num_sensors)

    train_ds = SensorGraphDataset(X_train_tr, y_train_tr, edge_index, attrs_train_tr)
    val_ds = SensorGraphDataset(X_train_val, y_train_val, edge_index, attrs_train_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    print("  Training GAT on SWAT data...")
    t0 = time.time()
    models_dir = SRC_ROOT / "models"
    trainer = GAT_Trainer(config, models_dir=models_dir)
    history = trainer.fit(train_loader, val_loader)
    train_time = time.time() - t0

    # Evaluate on held-out test set
    metrics = evaluate_gat_on_data(trainer.model, X_test, y_test, attrs_test, num_sensors)
    metrics["train_time_s"] = round(train_time, 2)
    metrics["train_samples"] = len(X_train)
    metrics["test_samples"] = len(X_test)
    metrics["num_sensors"] = num_sensors

    print(f"  SWAT: F1={metrics['f1']:.3f} | Acc={metrics['accuracy']:.3f} | "
          f"Time={train_time:.1f}s | Train={len(X_train)} Test={len(X_test)}")

    return metrics
