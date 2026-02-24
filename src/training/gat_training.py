"""
gat_training.py
================

GAT model training orchestration.

Trains Graph Attention Networks for Byzantine attack detection on sensor networks.
Supports diverse attack types for robust training.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT / "models"))
sys.path.insert(0, str(SRC_ROOT / "data"))
sys.path.insert(0, str(SRC_ROOT))

from gat_model import GAT_Config, GAT_Byzantine_Detector, GAT_Trainer
from gat_data_generator import (
    SyntheticDataGenerator,
    SensorGraphDataset,
    create_sensor_graph_fully_connected,
)
from torch.utils.data import DataLoader
import torch


def custom_collate_fn(batch):
    """Custom collate function for PyTorch DataLoader."""
    xs, ys, attrs, edge_indices = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(attrs), edge_indices[0]


def train_gat_model(
    num_nodes: int = 5,
    seq_len: int = 100,
    num_samples: int = 400,
    epochs: int = 50,
    batch_size: int = 16,
    verbose: bool = True,
    diverse_attacks: bool = True,
    attack_generator=None,
) -> Tuple[GAT_Byzantine_Detector, Dict]:
    """
    Train GAT model from scratch with diverse attack types.
    
    Args:
        num_nodes: Number of sensor nodes
        seq_len: Sequence length
        num_samples: Number of training samples
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Print training progress
        diverse_attacks: Use diverse attack types for training
        attack_generator: Optional custom AttackDataGenerator instance
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    config = GAT_Config(
        hidden_channels=64, num_layers=2, num_heads=4,
        dropout=0.2, learning_rate=0.001, batch_size=batch_size,
        epochs=epochs, early_stopping_patience=8, device="cpu",
    )

    if diverse_attacks and num_nodes == 5:
        if attack_generator is None:
            from src.utils.attack_data_generator import AttackDataGenerator
            attack_generator = AttackDataGenerator
        
        att_gen = attack_generator(num_nodes=num_nodes, seq_len=seq_len, num_samples=num_samples // 4)
        datasets = []
        for method, kwargs in [
            ("linear_drift", {"delta": 0.02}),
            ("exponential_drift", {"delta": 0.02, "alpha": 3.0}),
            ("fdi_step_change", {"magnitude": 2.0}),
            ("frogging_attack", {"delta": 0.02, "switch_period": 4}),
            ("polynomial_drift", {"delta": 0.015, "power": 2.0}),
            ("majority_compromised", {"delta": 0.002}),
        ]:
            X, y, a = getattr(att_gen, method)(**kwargs)
            datasets.append((X, y, a))

        X = np.concatenate([d[0] for d in datasets], axis=0)
        y = np.concatenate([d[1] for d in datasets], axis=0)
        node_attrs = np.concatenate([d[2] for d in datasets], axis=0)
        idx = np.random.permutation(len(X))
        X, y, node_attrs = X[idx], y[idx], node_attrs[idx]
    else:
        gen = SyntheticDataGenerator(
            num_nodes=num_nodes, sequence_length=seq_len,
            num_samples_per_class=num_samples // 2, random_seed=42,
        )
        X, y, node_attrs = gen.generate_dataset()

    split = int(0.8 * len(X))
    edge_index = create_sensor_graph_fully_connected(num_nodes)

    train_ds = SensorGraphDataset(X[:split], y[:split], edge_index, node_attrs[:split])
    val_ds = SensorGraphDataset(X[split:], y[split:], edge_index, node_attrs[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    models_dir = SRC_ROOT / "models"
    trainer = GAT_Trainer(config, models_dir=models_dir)
    if verbose:
        print(f"  Training GAT (N={num_nodes}, samples={num_samples}, epochs={epochs})...")
    history = trainer.fit(train_loader, val_loader)
    return trainer.model, history


def evaluate_gat_on_data(
    model: GAT_Byzantine_Detector,
    X: np.ndarray,
    y: np.ndarray,
    node_attrs: np.ndarray,
    num_nodes: int,
    batch_size: int = 32,
) -> Dict:
    """
    Evaluate trained GAT on given dataset.
    
    Args:
        model: Trained GAT model
        X: Input data (samples, nodes, timesteps)
        y: Ground truth labels
        node_attrs: Node attributes
        num_nodes: Number of nodes
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    from src.utils import compute_metrics, compute_attribution_accuracy
    
    edge_index = create_sensor_graph_fully_connected(num_nodes)
    ds = SensorGraphDataset(X, y, edge_index, node_attrs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    model.eval()
    all_preds, all_labels = [], []
    all_attr_preds, all_attr_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch, attr_batch, ei in loader:
            logits, attr_out = model(x_batch, ei)
            preds = logits.argmax(dim=1).numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
            all_attr_preds.append(attr_out.numpy())
            all_attr_labels.append(attr_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    attr_pred = np.concatenate(all_attr_preds)
    attr_true = np.concatenate(all_attr_labels)

    metrics = compute_metrics(y_true, y_pred)
    metrics["attribution_acc"] = compute_attribution_accuracy(attr_pred, attr_true)
    metrics["num_samples"] = len(y_true)
    return metrics
