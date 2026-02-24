"""
table_swat_attribution.py
=======================
Generate LaTeX table for SWAT attribution analysis results.

Results from Experiment 7: Attribution accuracy on real SWAT data at multiple horizons.
"""

from pathlib import Path
from typing import Dict


def generate_table_swat_attribution(results: Dict, output_dir: Path) -> Path:
    """
    Generate SWAT attribution analysis table.
    
    Args:
        results: Dict from experiment_7_swat_attribution with accuracies per horizon
        output_dir: Directory to save the LaTeX table
        
    Returns:
        Path to the generated .tex file
    """
    if not results:
        return None
    
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{SWAT Attribution Analysis: Accuracy at Multiple Horizons}")
    lines.append(r"\label{tab:swat_attribution_empirical}")
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
    
    table_path = output_dir / "table_swat_attribution_empirical.tex"
    table_path.write_text("\n".join(lines))
    
    return table_path
