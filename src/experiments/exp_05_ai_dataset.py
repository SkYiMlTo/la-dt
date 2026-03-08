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
    """Validate GAT on cross-domain AI power generation data using real labeled dataset."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Cross-Domain AI Dataset Validation")
    print("=" * 80)

    # Load real labeled AI power grid data
    x_train_path = PROJECT_ROOT / "src" / "data" / "raw" / "ai-data" / "x_train12"
    y_train_path = PROJECT_ROOT / "src" / "data" / "raw" / "ai-data" / "y_train12"
    
    if not x_train_path.exists() or not y_train_path.exists():
        print("  [SKIP] AI dataset not found")
        return {"status": "skipped", "reason": "AI labeled data not found"}

    print("  Loading pre-labeled AI power grid data...")
    # Load 2000 samples for balanced runtime and credibility
    x_train = pd.read_parquet(str(x_train_path))
    y_train = pd.read_parquet(str(y_train_path))
    
    X_data = x_train.iloc[:2000].values.astype(np.float32)
    y_labels = y_train.iloc[:2000].values.astype(np.float32)
    
    print(f"  Loaded {len(X_data)} samples from full dataset")

    # Normalize features
    mean = np.mean(X_data, axis=0)
    std = np.std(X_data, axis=0)
    std[std == 0] = 1
    X_data = (X_data - mean) / std
    
    # Create binary labels based on label variance
    label_variance = np.var(y_labels, axis=1)
    threshold = np.percentile(label_variance, 35)
    y_binary = (label_variance > threshold).astype(np.int64)
    
    print(f"  Normal samples: {(y_binary == 0).sum()}, Anomalies: {(y_binary == 1).sum()}")
    
    # Use 150 sensors for good representativeness
    num_sensors = 150
    X_data = X_data[:, :num_sensors]
    
    # Create windows with balanced parameters
    window_size = 35
    stride = 18
    
    X_windows, y_windows, attrs_windows = [], [], []
    for i in range(0, len(X_data) - window_size, stride):
        window = X_data[i:i + window_size]
        label = y_binary[i]
        
        X_windows.append(window.T)  # (num_sensors, window_size)
        y_windows.append(label)
        attrs_windows.append(np.ones(num_sensors, dtype=np.float32))
    
    X_all = np.array(X_windows, dtype=np.float32)
    y_all = np.array(y_windows, dtype=np.int64)
    attrs_all = np.array(attrs_windows, dtype=np.float32)
    
    print(f"  Created {len(X_all)} temporal windows")
    print(f"  Shape: X={X_all.shape}, y={y_all.shape}")
    print(f"  Normal windows: {(y_all == 0).sum()}, Anomalies: {(y_all == 1).sum()}")
    
    # Train/val split (80/20)
    split_idx = int(0.8 * len(X_all))
    idx = np.arange(len(X_all))
    np.random.seed(42)
    np.random.shuffle(idx)
    X_all, y_all, attrs_all = X_all[idx], y_all[idx], attrs_all[idx]
    
    # Optimized model with standard config from other experiments
    config = GAT_Config(
        hidden_channels=128,
        num_layers=3,
        num_heads=8,
        dropout=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        early_stopping_patience=15,
        device="cpu"
    )
    edge_index = create_sensor_graph_fully_connected(num_sensors)

    train_ds = SensorGraphDataset(X_all[:split_idx], y_all[:split_idx], edge_index, attrs_all[:split_idx])
    val_ds = SensorGraphDataset(X_all[split_idx:], y_all[split_idx:], edge_index, attrs_all[split_idx:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    print("  Training GAT on real AI power grid dataset...")
    t0 = time.time()
    models_dir = SRC_ROOT / "models"
    trainer = GAT_Trainer(config, models_dir=models_dir)
    history = trainer.fit(train_loader, val_loader)
    train_time = time.time() - t0

    metrics = evaluate_gat_on_data(trainer.model, X_all[split_idx:], y_all[split_idx:], 
                                    attrs_all[split_idx:], num_sensors)
    metrics["train_time_s"] = round(train_time, 2)
    metrics["total_samples"] = len(X_all)
    metrics["num_sensors"] = num_sensors

    print(f"  AI Dataset: F1={metrics['f1']:.3f} | Acc={metrics['accuracy']:.3f} | "
          f"Time={train_time:.1f}s | Samples={len(X_all)}")

    return metrics
