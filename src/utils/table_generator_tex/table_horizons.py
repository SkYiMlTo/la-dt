"""
table_horizons.py
=======================
Generate LaTeX table for multi-horizon attribution accuracy.

Results from Experiment 2: Attribution accuracy at 5, 10, 30, 60-minute horizons.
"""

from pathlib import Path
from typing import Dict


def generate_table_horizons(results: Dict, output_dir: Path) -> Path:
    """
    Generate multi-horizon attribution table.
    
    Args:
        results: Dict from experiment_2_multi_horizon with accuracies per horizon
        output_dir: Directory to save the LaTeX table
        
    Returns:
        Path to the generated .tex file
    """
    if not results:
        return None
    
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Attribution Accuracy at Multiple Detection Horizons}")
    lines.append(r"\label{tab:horizon_empirical}")
    lines.append(r"\begin{tabular}{ccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Horizon (min)} & \textbf{Correct} & \textbf{Accuracy (\%)} & \textbf{Avg VGR} & \textbf{Avg LLR} \\")
    lines.append(r"\midrule")
    
    for h in ["5", "10", "30", "60"]:
        if h in results:
            d = results[h]
            lines.append(f"{h} & {d['correct']:2d}/{d['total']:2d} & {d['accuracy_pct']:5.1f}\\% & {d['avg_vgr']:.2f} & {d['avg_llr']:.2f} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    table_path = output_dir / "table_horizon_empirical.tex"
    table_path.write_text("\n".join(lines))
    
    return table_path
