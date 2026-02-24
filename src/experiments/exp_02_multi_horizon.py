"""
exp_02_multi_horizon.py
========================
Experiment 2: Multi-Horizon Attribution Accuracy

Tests attribution accuracy at 5, 10, 30, 60 minute horizons
using VGR+SCD+LLR on synthetic linear Byzantine drift.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from src.utils import run_attribution_at_horizon


def experiment_2_multi_horizon(num_windows: int = 40) -> Dict:
    """Test attribution accuracy at multiple horizons."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Multi-Horizon Attribution Accuracy")
    print("=" * 80)

    NUM_NODES = 5
    HORIZONS_MIN = [5, 10, 30, 60]
    DRIFT_RATE = 0.028
    SAMPLING_HZ = 1

    results_by_horizon = defaultdict(list)

    for w in range(num_windows):
        np.random.seed(3000 + w)
        t_len = 3600
        noise_sigma = 0.08 + 0.06 * np.random.rand()
        window_normal = np.zeros((t_len, NUM_NODES))
        for node in range(NUM_NODES):
            trend = np.linspace(0, 0.5 + 0.2 * np.random.randn(), t_len)
            seasonal = (0.2 + 0.15 * np.random.rand()) * np.sin(2 * np.pi * np.arange(t_len) / t_len)
            noise = np.random.normal(0, noise_sigma, t_len)
            window_normal[:, node] = trend + seasonal + noise

        window_attacked = window_normal.copy()
        targets = np.random.choice(NUM_NODES, size=2, replace=False)

        tier = w % 4
        if tier == 0:
            window_drift = DRIFT_RATE * 0.12
        elif tier == 1:
            window_drift = DRIFT_RATE * 0.40
        else:
            window_drift = DRIFT_RATE * (0.8 + 0.4 * np.random.rand())

        for j, target in enumerate(targets):
            sign = 1 if j % 2 == 0 else -1
            t_arr = np.arange(t_len, dtype=np.float64) / 60.0
            drift = sign * window_drift * t_arr
            noise_drift = np.random.normal(0, 0.045, t_len)
            window_attacked[:, target] += drift + noise_drift

        for h_min in HORIZONS_MIN:
            h_samples = h_min * 60 * SAMPLING_HZ
            result = run_attribution_at_horizon(window_normal, window_attacked, h_samples)
            results_by_horizon[h_min].append(result)

    summary = {}
    for h_min in HORIZONS_MIN:
        horizon_results = results_by_horizon[h_min]
        correct = sum(1 for r in horizon_results if r["correct"])
        total = len(horizon_results)
        acc = correct / max(total, 1) * 100
        avg_vgr = np.mean([r["vgr"] for r in horizon_results])
        avg_scd = np.mean([r["scd"] for r in horizon_results])
        avg_llr = np.mean([r["llr_score"] for r in horizon_results])
        summary[str(h_min)] = {
            "accuracy_pct": round(acc, 1),
            "correct": correct,
            "total": total,
            "avg_vgr": round(float(avg_vgr), 3),
            "avg_scd": round(float(avg_scd), 3),
            "avg_llr": round(float(avg_llr), 3),
        }
        print(f"  Horizon {h_min:2d} min: {correct:2d}/{total:2d} ({acc:5.1f}%) | "
              f"VGR={avg_vgr:.2f} | SCD={avg_scd:.3f} | LLR={avg_llr:.2f}")

    return summary
