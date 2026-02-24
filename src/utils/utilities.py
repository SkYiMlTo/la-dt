"""
utilities.py
=============

Shared utility functions for all experiments:
- Data loading and collation
- Metrics computation
- GAT training and evaluation (delegates to src.training.gat_training)

Attribution pipeline moved to src.attribution.attribution_pipeline
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT / "models"))
sys.path.insert(0, str(SRC_ROOT / "data"))
sys.path.insert(0, str(SRC_ROOT))

# Import training modules
from training.gat_training import (
    custom_collate_fn as _gat_collate_fn,
    train_gat_model as _train_gat_impl,
    evaluate_gat_on_data as _eval_gat_impl,
)

# Import attribution pipeline
from attribution.attribution_pipeline import run_attribution_at_horizon


# ============================================================================
# Data Loading Utilities
# ============================================================================

def custom_collate_fn(batch):
    """Custom collate function for PyTorch DataLoader."""
    return _gat_collate_fn(batch)


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall, F1, accuracy from binary arrays."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_attribution_accuracy(
    attr_pred: np.ndarray, attr_true: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute node-level attribution accuracy using adaptive thresholding."""
    # Flatten if needed
    attr_pred_flat = attr_pred.flatten() if attr_pred.ndim > 1 else attr_pred
    attr_true_flat = attr_true.flatten() if attr_true.ndim > 1 else attr_true
    
    # Normalize predictions to [0, 1] if needed
    attr_pred_min = np.min(attr_pred_flat)
    attr_pred_max = np.max(attr_pred_flat)
    if attr_pred_max > attr_pred_min:
        attr_pred_norm = (attr_pred_flat - attr_pred_min) / (attr_pred_max - attr_pred_min)
    else:
        attr_pred_norm = attr_pred_flat.copy()
    
    # Use adaptive threshold based on true distribution
    true_binary = (attr_true_flat >= 0.5).astype(int)
    
    # If ground truth is mostly one class, use percentile-based threshold
    pos_frac = np.mean(true_binary)
    if pos_frac > 0.1 and pos_frac < 0.9:
        # Balanced case: use standard threshold
        pred_binary = (attr_pred_norm >= 0.5).astype(int)
    else:
        # Imbalanced case: use percentile threshold
        if pos_frac <= 0.1:
            # Few positives: use top-k approach
            k = max(1, int(np.sum(true_binary)))
            threshold_idx = np.argsort(attr_pred_norm)[-k:]
            pred_binary = np.zeros_like(attr_pred_norm, dtype=int)
            pred_binary[threshold_idx] = 1
        else:
            # Few negatives: use median threshold
            pred_binary = (attr_pred_norm >= np.median(attr_pred_norm)).astype(int)
    
    correct = np.sum(pred_binary == true_binary)
    total = len(true_binary)
    return round(correct / max(total, 1), 4)

# ============================================================================
# GAT Training and Evaluation
# ============================================================================

def train_gat_model(
    num_nodes: int = 5,
    seq_len: int = 100,
    num_samples: int = 400,
    epochs: int = 50,
    batch_size: int = 16,
    verbose: bool = True,
    diverse_attacks: bool = True,
    attack_generator=None,
) -> Tuple:
    """
    Train GAT model from scratch with diverse attack types.
    
    Delegates to src.training.gat_training.train_gat_model()
    
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
    return _train_gat_impl(
        num_nodes=num_nodes,
        seq_len=seq_len,
        num_samples=num_samples,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        diverse_attacks=diverse_attacks,
        attack_generator=attack_generator,
    )


def evaluate_gat_on_data(
    model,
    X: np.ndarray,
    y: np.ndarray,
    node_attrs: np.ndarray,
    num_nodes: int,
    batch_size: int = 32,
) -> Dict:
    """
    Evaluate trained GAT on given dataset.
    
    Delegates to src.training.gat_training.evaluate_gat_on_data()
    
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
    return _eval_gat_impl(
        model=model,
        X=X,
        y=y,
        node_attrs=node_attrs,
        num_nodes=num_nodes,
        batch_size=batch_size,
    )
