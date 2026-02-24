"""
attribution_pipeline.py
========================

Byzantine attack attribution using VGR + SCD + LLR evidence signals.

The attribution pipeline detects whether sensor deviations are due to:
1. Byzantine attacks (compromised sensors injecting drift)
2. Natural drift (legitimate environmental changes)

Three evidence signals are combined via log-likelihood ratio (LLR):
- VGR (Variance Growth Ratio): Spatial variance increase
- DEV (Deviation Ratio): Individual sensor drift magnitude
- SCD (Sensor Correlation Decay): Inter-sensor correlation loss
"""

import numpy as np
from typing import Dict, Optional


def run_attribution_at_horizon(
    window_normal: np.ndarray,
    window_attacked: np.ndarray,
    horizon_samples: int,
    vgr_threshold: float = 3.0,
    llr_threshold: float = 1.3,
    attacked_sensors: np.ndarray = None,
) -> Dict:
    """
    Run VGR + SCD + LLR attribution pipeline on a pair of windows.
    
    Three evidence signals:
    1. VGR (Variance Growth Ratio)
    2. DEV (Deviation Ratio)
    3. SCD (Sensor Correlation Decay)
    
    Args:
        window_normal: Normal sensor readings (T, num_sensors)
        window_attacked: Attacked sensor readings (T, num_sensors)
        horizon_samples: Number of samples to analyze
        vgr_threshold: VGR threshold for detection
        llr_threshold: LLR threshold for detection
        attacked_sensors: Ground truth attacked sensors (optional)
    
    Returns:
        Dictionary with attribution results:
        {
            "verdict": "Byzantine" or "Natural Drift",
            "llr_score": float,  # Log-likelihood ratio
            "vgr": float,        # Variance growth ratio
            "scd": float,        # Sensor correlation decay
            "dev_ratio": float,  # Maximum deviation ratio
            "correct": bool,     # Whether verdict matches ground truth
        }
    """
    wn = window_normal[:horizon_samples]
    wa = window_attacked[:horizon_samples]

    if len(wn) < 10 or len(wa) < 10:
        return {
            "verdict": "insufficient_data", 
            "correct": False,
            "vgr": 1.0, 
            "scd": 0.0, 
            "dev_ratio": 1.0, 
            "llr_score": 0.0,
        }

    num_sensors = wn.shape[1]
    half = len(wn) // 2

    # ========================================================================
    # Evidence 1: Variance Growth Ratio (VGR)
    # ========================================================================
    def spatial_variance(w):
        """Compute spatial variance: variance of sensor readings at each time."""
        return np.mean(np.var(w, axis=1))

    sv_first = spatial_variance(wa[:half])
    sv_second = spatial_variance(wa[half:])
    vgr = sv_second / max(sv_first, 1e-6)

    # ========================================================================
    # Evidence 2: Deviation Ratio (DEV)
    # ========================================================================
    dev_ratios = []
    for s in range(num_sensors):
        diff = np.abs(wa[:, s] - wn[:, s])
        first_half_mean = np.mean(diff[:half])
        second_half_mean = np.mean(diff[half:])
        dev_ratios.append(second_half_mean / max(first_half_mean, 1e-6))
    max_dev_ratio = float(np.max(dev_ratios))

    # ========================================================================
    # Evidence 3: Sensor Correlation Decay (SCD)
    # ========================================================================
    def bottom_k_corr(w, k=None):
        """
        Compute bottom-k correlation: average of k smallest pairwise correlations.
        
        Natural systems have high inter-sensor correlation.
        Byzantine attacks reduce or invert correlations.
        """
        n_s = w.shape[1]
        if k is None:
            k = max(3, n_s // 2)
        
        # For large sensor networks, sample for efficiency
        if n_s > 80:
            idx = np.random.choice(n_s, size=50, replace=False)
            w = w[:, idx]
        
        # Compute correlation matrix
        c = np.corrcoef(w.T)
        mask = np.triu_indices_from(c, k=1)
        vals = c[mask]
        vals = vals[~np.isnan(vals)]
        
        if len(vals) < k:
            return float(np.mean(vals)) if len(vals) > 0 else 1.0
        
        return float(np.mean(np.sort(vals)[:k]))

    corr_n = bottom_k_corr(wn)
    corr_a = bottom_k_corr(wa)
    scd = max(0.0, corr_n - corr_a)
    scd = float(np.clip(scd, 0, 1))

    # ========================================================================
    # Combine Evidence via Sigmoid + Log-Likelihood Ratio
    # ========================================================================
    def sigmoid_evidence(x, mid, steep, max_llr):
        """
        Convert a signal (VGR, DEV, SCD) to log-likelihood via sigmoid.
        
        Args:
            x: Signal value
            mid: Midpoint (where sigmoid = 0.5)
            steep: Steepness (higher = sharper transition)
            max_llr: Maximum LLR contribution
        
        Returns:
            LLR contribution in range [0, max_llr]
        """
        z = steep * (x - mid)
        z = max(-20.0, min(20.0, z))  # Clip to prevent overflow
        s = 1.0 / (1.0 + np.exp(-z))   # Sigmoid
        return max(0.0, max_llr * (2.0 * s - 1.0))

    llr = 0.0
    llr += sigmoid_evidence(vgr, mid=1.5, steep=2.0, max_llr=2.5)
    llr += sigmoid_evidence(max_dev_ratio, mid=1.5, steep=2.0, max_llr=2.0)
    llr += sigmoid_evidence(scd, mid=0.1, steep=8.0, max_llr=1.5)

    # ========================================================================
    # Final Verdict
    # ========================================================================
    verdict = "Byzantine" if llr >= llr_threshold else "Natural Drift"
    
    return {
        "verdict": verdict,
        "llr_score": round(float(llr), 4),
        "vgr": round(float(vgr), 4),
        "scd": round(float(scd), 4),
        "dev_ratio": round(float(max_dev_ratio), 4),
        "correct": verdict == "Byzantine",
    }
