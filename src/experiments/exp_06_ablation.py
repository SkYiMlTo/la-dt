"""
exp_06_ablation.py
===================
Experiment 6: Ablation Study

Remove each LLR evidence signal and measure attribution accuracy.
Tests importance of VGR, DEV, and SCD across attack scenarios.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from src.utils import AttackDataGenerator


def experiment_6_ablation(num_windows: int = 100) -> Dict:
    """Run ablation study on attribution signals."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: Ablation Study (LLR Signal Contributions)")
    print("=" * 80)

    NUM_NODES = 5
    attack_scenarios = [
        ("Linear_15min", "linear", 0.012, 900),
        ("FDI_Step_30min", "fdi_step", 0.6, 1800),
        ("Correlated_15min", "correlated", 0.012, 900),
        ("Moderate_30min", "linear", 0.008, 1800),
        ("Strong_60min", "linear", 0.014, 3600),
    ]

    configs = {
        "Full_LA-DT": {"use_vgr": True, "use_dev": True, "use_scd": True},
        "w/o_VGR": {"use_vgr": False, "use_dev": True, "use_scd": True},
        "w/o_DEV": {"use_vgr": True, "use_dev": False, "use_scd": True},
        "w/o_SCD": {"use_vgr": True, "use_dev": True, "use_scd": False},
        "VGR_only": {"use_vgr": True, "use_dev": False, "use_scd": False},
        "DEV_only": {"use_vgr": False, "use_dev": True, "use_scd": False},
    }

    def sigmoid_evidence(x, mid, steep, max_llr):
        z = steep * (x - mid)
        z = max(-20.0, min(20.0, z))
        s = 1.0 / (1.0 + np.exp(-z))
        return max(0.0, max_llr * (2.0 * s - 1.0))

    def run_ablation_attribution(wn, wa, h_samples, use_vgr, use_dev, use_scd):
        wn_sub = wn[:h_samples]
        wa_sub = wa[:h_samples]
        half = len(wa_sub) // 2
        num_s = wn_sub.shape[1]

        llr = 0.0

        if use_vgr:
            sv_first = np.mean(np.var(wa_sub[:half], axis=1))
            sv_second = np.mean(np.var(wa_sub[half:], axis=1))
            vgr = sv_second / max(sv_first, 1e-6)
            llr += sigmoid_evidence(vgr, mid=1.5, steep=1.4, max_llr=1.2)

        if use_dev:
            dev_ratios = []
            for s in range(num_s):
                diff = np.abs(wa_sub[:, s] - wn_sub[:, s])
                first_half_diff = np.mean(diff[:half])
                second_half_diff = np.mean(diff[half:])
                dev_ratios.append(second_half_diff / max(first_half_diff, 1e-6))
            max_dev = float(np.max(dev_ratios))
            llr += sigmoid_evidence(max_dev, mid=1.5, steep=1.4, max_llr=2.0)

        if use_scd:
            def bottom_k_corr(w, k=5):
                c = np.corrcoef(w.T)
                mask = np.triu_indices_from(c, k=1)
                vals = c[mask]
                vals = vals[~np.isnan(vals)]
                if len(vals) < k:
                    return float(np.mean(vals)) if len(vals) > 0 else 1.0
                return float(np.mean(np.sort(vals)[:k]))
            corr_n = bottom_k_corr(wn_sub)
            corr_a = bottom_k_corr(wa_sub)
            scd = max(0.0, corr_n - corr_a)
            scd = float(np.clip(scd, 0, 1))
            llr += sigmoid_evidence(scd, mid=0.2, steep=3.0, max_llr=1.4)

        return "Byzantine" if llr >= 1.48 else "Natural Drift"

    def generate_attack_window(attack_type, drift_rate, t_len=3600, num_nodes=5):
        wn = np.zeros((t_len, num_nodes))
        noise_sigma = 0.08 + 0.06 * np.random.rand()
        for node in range(num_nodes):
            trend = np.linspace(0, 0.5 + 0.15 * np.random.randn(), t_len)
            seasonal = (0.2 + 0.1 * np.random.rand()) * np.sin(2 * np.pi * np.arange(t_len) / t_len)
            noise = np.random.normal(0, noise_sigma, t_len)
            wn[:, node] = trend + seasonal + noise

        wa = wn.copy()
        t_arr = np.arange(t_len, dtype=np.float64) / 60.0

        if attack_type == "linear":
            targets = np.random.choice(num_nodes, size=2, replace=False)
            for j, target in enumerate(targets):
                sign = 1 if j % 2 == 0 else -1
                wa[:, target] += sign * drift_rate * t_arr + np.random.normal(0, 0.015, t_len)
        elif attack_type == "fdi_step":
            targets = np.random.choice(num_nodes, size=2, replace=False)
            onset = t_len // 4
            for j, target in enumerate(targets):
                sign = 1 if j % 2 == 0 else -1
                step_signal = np.zeros(t_len)
                step_signal[onset:] = sign * drift_rate
                wa[:, target] += step_signal + np.random.normal(0, 0.02, t_len)
        elif attack_type == "correlated":
            for node in range(num_nodes):
                wa[:, node] += drift_rate * t_arr + np.random.normal(0, 0.015, t_len)

        return wn, wa

    results = {}
    full_accuracy = None

    for config_name, config_kwargs in configs.items():
        correct_total = 0
        window_count = 0

        for scenario_name, attack_type, base_drift, t_len in attack_scenarios:
            for window in range(num_windows):
                np.random.seed(5000 + hash(config_name + scenario_name) % 1000 + window)
                wn, wa = generate_attack_window(attack_type, base_drift, t_len, NUM_NODES)
                h_samples = t_len // 2
                verdict = run_ablation_attribution(wn, wa, h_samples, **config_kwargs)
                if verdict == "Byzantine":
                    correct_total += 1
                window_count += 1

        accuracy_pct = (correct_total / max(window_count, 1)) * 100
        if config_name == "Full_LA-DT":
            full_accuracy = accuracy_pct

        impact = 0.0 if full_accuracy is None else accuracy_pct - full_accuracy
        results[config_name] = {
            "accuracy_pct": round(accuracy_pct, 1),
            "impact_pct": round(impact, 1),
            "correct": correct_total,
            "total": window_count,
        }

        print(f"  {config_name:20s}: {accuracy_pct:5.1f}% (impact: {impact:+5.1f}%)")

    return results
