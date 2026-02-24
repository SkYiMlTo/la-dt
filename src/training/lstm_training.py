"""
lstm_training.py
=================

LSTM-like anomaly scorer initialization and management.

The AnomalyScorer uses Exponentially Weighted Moving Average (EWMA) + z-score
for statistical anomaly detection. It learns baselines incrementally from data
without a traditional training phase.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT / "models"))
sys.path.insert(0, str(SRC_ROOT))

from lstm_model import AnomalyScorer
import numpy as np


def initialize_anomaly_scorer(
    threshold: float = 4.0,
    window_size: int = 60,
    ewma_alpha: float = 0.3,
    min_samples: int = 10,
) -> AnomalyScorer:
    """
    Initialize an AnomalyScorer with specified parameters.
    
    Args:
        threshold: z-score threshold for anomaly detection
        window_size: Size of sliding window per sensor
        ewma_alpha: EWMA smoothing factor (0-1, higher = more reactive)
        min_samples: Minimum samples before scoring
    
    Returns:
        Initialized AnomalyScorer instance
    """
    scorer = AnomalyScorer(threshold=threshold)
    
    # Store configuration for later reference
    scorer._window_size = window_size
    scorer._ewma_alpha = ewma_alpha
    scorer._min_samples = min_samples
    
    return scorer


def train_anomaly_scorer_on_data(
    scorer: AnomalyScorer,
    X: np.ndarray,
    num_nodes: int,
    verbose: bool = True,
) -> AnomalyScorer:
    """
    Warm up the AnomalyScorer on normal (non-attack) data.
    
    The scorer learns baselines incrementally by processing normal readings.
    This builds up the EWMA and variance statistics before deployment.
    
    Args:
        scorer: AnomalyScorer instance to warm up
        X: Normal sensor data (samples, nodes, timesteps) or flattened readings
        num_nodes: Number of sensor nodes
        verbose: Print progress
    
    Returns:
        Updated AnomalyScorer ready for anomaly detection
    """
    if verbose:
        print(f"  Warming up AnomalyScorer on {len(X)} normal samples...")
    
    # Process data to build baselines
    if len(X.shape) == 3:
        # Shape: (samples, nodes, timesteps)
        for sample_idx, sample in enumerate(X):
            for node_idx in range(min(num_nodes, sample.shape[0])):
                for t in range(sample.shape[1]):
                    value = float(sample[node_idx, t])
                    scorer.score(
                        node_id=node_idx,
                        sensor=f"sensor_{node_idx}",
                        value=value,
                    )
            if verbose and (sample_idx + 1) % max(1, len(X) // 10) == 0:
                print(f"    Processed {sample_idx + 1}/{len(X)} samples")
    
    if verbose:
        print(f"  AnomalyScorer warm-up complete")
        print(f"    Baselines learned for {len(scorer._baselines)} sensor(s)")
    
    return scorer


def save_anomaly_scorer(scorer: AnomalyScorer, path: Path) -> Path:
    """
    Save trained AnomalyScorer to disk.
    
    Args:
        scorer: AnomalyScorer instance to save
        path: Path to save to
    
    Returns:
        Path where scorer was saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scorer.save(path)
    return path


def load_anomaly_scorer(path: Path) -> AnomalyScorer:
    """
    Load previously saved AnomalyScorer.
    
    Args:
        path: Path to saved scorer
    
    Returns:
        Loaded AnomalyScorer instance
    """
    return AnomalyScorer.load(Path(path))


def evaluate_anomaly_scorer(
    scorer: AnomalyScorer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_nodes: int,
) -> Dict:
    """
    Evaluate AnomalyScorer on test data.
    
    Args:
        scorer: Trained scorer
        X_test: Test sensor data (samples, nodes, timesteps)
        y_test: Ground truth labels (0=normal, 1=attack)
        num_nodes: Number of nodes
    
    Returns:
        Dictionary with evaluation metrics
    """
    from src.utils import compute_metrics
    
    y_pred = []
    
    # Score each sample
    for sample in X_test:
        # Count anomalies detected in this sample
        num_anomalies = 0
        num_readings = 0
        
        for node_idx in range(min(num_nodes, sample.shape[0])):
            for t in range(sample.shape[1]):
                value = float(sample[node_idx, t])
                result = scorer.score(
                    node_id=node_idx,
                    sensor=f"sensor_{node_idx}",
                    value=value,
                )
                num_readings += 1
                if result and result.is_anomaly:
                    num_anomalies += 1
        
        # Classify: if > threshold% of readings are anomalies, classify as attack
        anomaly_ratio = num_anomalies / max(1, num_readings)
        pred = 1 if anomaly_ratio > 0.1 else 0
        y_pred.append(pred)
    
    y_pred = np.array(y_pred)
    
    metrics = compute_metrics(y_test, y_pred)
    metrics["num_samples"] = len(y_test)
    return metrics
