"""
table_robustness.py
=======================
Generate LaTeX table for attack robustness results.

Results from Experiment 1: Byzantine attack robustness (F1 scores across 8 attack types).
"""

from pathlib import Path
from typing import Dict


def generate_table_robustness(results: Dict, output_dir: Path) -> Path:
    """
    Generate attack robustness table.
    
    Args:
        results: Dict from experiment_1_attack_robustness with F1 scores per attack class
        output_dir: Directory to save the LaTeX table
        
    Returns:
        Path to the generated .tex file
    """
    if not results:
        return None
    
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Byzantine Attack Robustness: F1 Scores}")
    lines.append(r"\label{tab:robustness_empirical}")
    lines.append(r"\begin{tabular}{ccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Attack Class} & \textbf{F1 Score} & \textbf{Std Dev} \\")
    lines.append(r"\midrule")
    
    attack_order = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    
    for attack_id in attack_order:
        if attack_id in results:
            d = results[attack_id]
            name = d.get("name", attack_id)
            f1 = d.get("f1", 0.0)
            std = d.get("std", 0.0)
            lines.append(f"{attack_id} ({name}) & {f1:.3f} & {std:.3f} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    table_path = output_dir / "table_robustness_empirical.tex"
    table_path.write_text("\n".join(lines))
    
    return table_path
