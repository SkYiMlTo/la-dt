#!/usr/bin/env python3
"""
main_run_all_experiments.py
============================

Master orchestrator for all LA-DT experiments.
Imports individual experiment modules and runs them in sequence,
aggregating results into JSON and LaTeX tables.

Run this file to execute all 7 experiments from scratch with real computed
results (no hardcoded metrics).

Usage:
  python -m src.experiments.main_run_all_experiments              # All experiments
  python -m src.experiments.main_run_all_experiments 2 5 6 7    # Only exp 2,5,6,7
"""

import sys
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

RESULTS_DIR = SRC_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Import all experiment modules
from .exp_01_attack_robustness import experiment_1_attack_robustness
from .exp_02_multi_horizon import experiment_2_multi_horizon
from .exp_03_scalability import experiment_3_scalability
from .exp_04_swat_validation import experiment_4_swat_validation
from .exp_05_ai_dataset import experiment_5_ai_dataset
from .exp_06_ablation import experiment_6_ablation
from .exp_07_swat_attribution import experiment_7_swat_attribution

# Import table generators (tex, md, csv)
from utils.table_generator_tex import (
    generate_table_robustness as generate_table_robustness_tex,
    generate_table_horizons as generate_table_horizons_tex,
    generate_table_scalability as generate_table_scalability_tex,
    generate_table_realworld as generate_table_realworld_tex,
    generate_table_ablation as generate_table_ablation_tex,
    generate_table_swat_attribution as generate_table_swat_attribution_tex,
)

from utils.table_generator_md import (
    generate_table_robustness as generate_table_robustness_md,
    generate_table_horizons as generate_table_horizons_md,
    generate_table_scalability as generate_table_scalability_md,
    generate_table_realworld as generate_table_realworld_md,
    generate_table_ablation as generate_table_ablation_md,
    generate_table_swat_attribution as generate_table_swat_attribution_md,
)

from utils.table_generator_csv import (
    generate_table_robustness as generate_table_robustness_csv,
    generate_table_horizons as generate_table_horizons_csv,
    generate_table_scalability as generate_table_scalability_csv,
    generate_table_realworld as generate_table_realworld_csv,
    generate_table_ablation as generate_table_ablation_csv,
    generate_table_swat_attribution as generate_table_swat_attribution_csv,
)


def generate_all_tables(all_results: Dict):
    """Generate publication-ready tables in LaTeX, Markdown, and CSV formats."""
    print("\n" + "=" * 80)
    print("GENERATING TABLES (LaTeX, Markdown, CSV)")
    print("=" * 80)

    robustness = all_results.get("experiment_1_robustness", {})
    horizons = all_results.get("experiment_2_horizon", {})
    scalability = all_results.get("experiment_3_scalability", {})
    swat = all_results.get("experiment_4_swat", {})
    ai = all_results.get("experiment_5_ai", {})
    ablation = all_results.get("experiment_6_ablation", {})
    swat_attr = all_results.get("experiment_7_swat_attribution", {})

    # Table 1: Robustness (LaTeX, MD, CSV)
    if robustness:
        path_tex = generate_table_robustness_tex(robustness, RESULTS_DIR)
        path_md = generate_table_robustness_md(robustness, RESULTS_DIR)
        path_csv = generate_table_robustness_csv(robustness, RESULTS_DIR)
        if path_tex:
            print(f"  ✓ {path_tex.name}")
        if path_md:
            print(f"  ✓ {path_md.name}")
        if path_csv:
            print(f"  ✓ {path_csv.name}")

    # Table 2: Multi-Horizon Attribution
    if horizons:
        path_tex = generate_table_horizons_tex(horizons, RESULTS_DIR)
        path_md = generate_table_horizons_md(horizons, RESULTS_DIR)
        path_csv = generate_table_horizons_csv(horizons, RESULTS_DIR)
        if path_tex:
            print(f"  ✓ {path_tex.name}")
        if path_md:
            print(f"  ✓ {path_md.name}")
        if path_csv:
            print(f"  ✓ {path_csv.name}")

    # Table 3: Scalability
    if scalability:
        path_tex = generate_table_scalability_tex(scalability, RESULTS_DIR)
        path_md = generate_table_scalability_md(scalability, RESULTS_DIR)
        path_csv = generate_table_scalability_csv(scalability, RESULTS_DIR)
        if path_tex:
            print(f"  ✓ {path_tex.name}")
        if path_md:
            print(f"  ✓ {path_md.name}")
        if path_csv:
            print(f"  ✓ {path_csv.name}")

    # Table 4: Real-World Validation
    if swat or ai:
        path_tex = generate_table_realworld_tex(swat, ai, RESULTS_DIR)
        path_md = generate_table_realworld_md(swat, ai, RESULTS_DIR)
        path_csv = generate_table_realworld_csv(swat, ai, RESULTS_DIR)
        if path_tex:
            print(f"  ✓ {path_tex.name}")
        if path_md:
            print(f"  ✓ {path_md.name}")
        if path_csv:
            print(f"  ✓ {path_csv.name}")

    # Table 5: Ablation Study
    if ablation:
        path_tex = generate_table_ablation_tex(ablation, RESULTS_DIR)
        path_md = generate_table_ablation_md(ablation, RESULTS_DIR)
        path_csv = generate_table_ablation_csv(ablation, RESULTS_DIR)
        if path_tex:
            print(f"  ✓ {path_tex.name}")
        if path_md:
            print(f"  ✓ {path_md.name}")
        if path_csv:
            print(f"  ✓ {path_csv.name}")

    # Table 6: SWAT Attribution
    if swat_attr and "status" not in swat_attr:
        path_tex = generate_table_swat_attribution_tex(swat_attr, RESULTS_DIR)
        path_md = generate_table_swat_attribution_md(swat_attr, RESULTS_DIR)
        path_csv = generate_table_swat_attribution_csv(swat_attr, RESULTS_DIR)
        if path_tex:
            print(f"  ✓ {path_tex.name}")
        if path_md:
            print(f"  ✓ {path_md.name}")
        if path_csv:
            print(f"  ✓ {path_csv.name}")


def main(exp_numbers: List[int] = None):
    """Run experiments in sequence (all or selected).
    
    Args:
        exp_numbers: List of experiment numbers to run (1-7).
                    If None, runs all experiments.
    """
    if exp_numbers is None:
        exp_numbers = [1, 2, 3, 4, 5, 6, 7]
    
    print("\n" + "#" * 80)
    print("#  LA-DT COMPREHENSIVE EXPERIMENT SUITE")
    print(f"#  Running experiments: {exp_numbers}")
    print("#  All results computed from scratch -- no hardcoded metrics")
    print("#" * 80)

    all_results = {}
    
    # Load existing results
    json_path = RESULTS_DIR / "experiment_results.json"
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                all_results = json.load(f)
        except:
            pass

    # Experiment 1: Attack Robustness
    if 1 in exp_numbers:
        print("\n[Running Experiment 1/7: Attack Robustness]")
        all_results["experiment_1_robustness"] = experiment_1_attack_robustness(num_seeds=5)

    # Experiment 2: Multi-Horizon Attribution
    if 2 in exp_numbers:
        print("\n[Running Experiment 2/7: Multi-Horizon Attribution]")
        all_results["experiment_2_horizon"] = experiment_2_multi_horizon(num_windows=40)

    # Experiment 3: Scalability
    if 3 in exp_numbers:
        print("\n[Running Experiment 3/7: Scalability]")
        all_results["experiment_3_scalability"] = experiment_3_scalability()

    # Experiment 4: SWAT Validation
    if 4 in exp_numbers:
        print("\n[Running Experiment 4/7: SWAT Validation]")
        all_results["experiment_4_swat"] = experiment_4_swat_validation()

    # Experiment 5: AI Dataset
    if 5 in exp_numbers:
        print("\n[Running Experiment 5/7: AI Dataset]")
        all_results["experiment_5_ai"] = experiment_5_ai_dataset()

    # Experiment 6: Ablation
    if 6 in exp_numbers:
        print("\n[Running Experiment 6/7: Ablation Study]")
        all_results["experiment_6_ablation"] = experiment_6_ablation(num_windows=40)

    # Experiment 7: SWAT Attribution
    if 7 in exp_numbers:
        print("\n[Running Experiment 7/7: SWAT Attribution]")
        all_results["experiment_7_swat_attribution"] = experiment_7_swat_attribution(num_windows=50)

    # Save all results as JSON
    json_path = RESULTS_DIR / "experiment_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  All results saved: {json_path}")

    # Generate tables in all formats (LaTeX, Markdown, CSV)
    generate_all_tables(all_results)

    # Final Summary
    print("\n" + "#" * 80)
    print("#  EXPERIMENT SUITE COMPLETE")
    print("#" * 80)
    rob = all_results.get("experiment_1_robustness", {})
    if rob:
        f1_vals = [v["f1"] for v in rob.values()]
        print(f"  Attack Robustness: avg F1 = {np.mean(f1_vals):.3f} across 8 attack classes")
    hor = all_results.get("experiment_2_horizon", {})
    if hor and "30" in hor:
        print(f"  Attribution @30min: {hor['30']['accuracy_pct']:.1f}%")
    scal = all_results.get("experiment_3_scalability", {})
    if scal:
        print(f"  Scalability: tested N={', '.join(scal.keys())} nodes")
    swat = all_results.get("experiment_4_swat", {})
    if swat and "f1" in swat:
        print(f"  SWAT: F1={swat['f1']:.3f}")
    ai = all_results.get("experiment_5_ai", {})
    if ai and "f1" in ai:
        print(f"  AI Dataset: F1={ai['f1']:.3f}")

    print(f"\n  Results Directory: {RESULTS_DIR}/")
    print(f"    - JSON:        experiment_results.json")
    print(f"    - LaTeX:       table_*_empirical.tex")
    print(f"    - Markdown:    table_*_empirical.md")
    print(f"    - CSV:         table_*_empirical.csv")
    print(f"    - Plots:       plots/")
    print("\n" + "#" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LA-DT experiments (all or selected)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.experiments.main_run_all_experiments
    → Run all experiments (1-7)
  
  python -m src.experiments.main_run_all_experiments 2 5 6 7
    → Run only experiments 2, 5, 6, 7
  
  python -m src.experiments.main_run_all_experiments 5
    → Run only experiment 5 (AI Dataset)
        """
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        type=int,
        help="Experiment numbers to run (1-7). Default: all experiments"
    )
    
    args = parser.parse_args()
    exp_list = args.experiments if args.experiments else None
    main(exp_list)
