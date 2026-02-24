"""
table_ablation.py
=======================
Generate LaTeX table for ablation study results.

Results from Experiment 6: Impact of removing each attribution signal (VGR, DEV, SCD).
"""

from pathlib import Path
from typing import Dict


def generate_table_ablation(results: Dict, output_dir: Path) -> Path:
    """
    Generate ablation study results table.
    
    Args:
        results: Dict from experiment_6_ablation with accuracy and impact per configuration
        output_dir: Directory to save the LaTeX table
        
    Returns:
        Path to the generated .tex file
    """
    if not results:
        return None
    
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation Study: Attribution Signal Contribution Analysis}")
    lines.append(r"\label{tab:ablation_empirical}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{Accuracy (\%)} & \textbf{Impact (\%)} & \textbf{Support} \\")
    lines.append(r"\midrule")
    
    config_order = ["Full_LA-DT", "w/o_VGR", "w/o_DEV", "w/o_SCD", "VGR_only", "DEV_only"]
    
    for config in config_order:
        if config in results:
            d = results[config]
            acc = d.get("accuracy_pct", 0.0)
            impact = d.get("impact_pct", 0.0)
            correct = d.get("correct", 0)
            total = d.get("total", 0)
            
            if config == "Full_LA-DT":
                lines.append(f"\\textbf{{{config}}} & \\textbf{{{acc:.1f}}} & \\textbf{{baseline}} & {correct}/{total} \\\\")
            else:
                lines.append(f"{config} & {acc:.1f}\\% & {impact:+.1f}\\% & {correct}/{total} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    table_path = output_dir / "table_ablation_empirical.tex"
    table_path.write_text("\n".join(lines))
    
    return table_path
