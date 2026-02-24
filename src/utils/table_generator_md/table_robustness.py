"""
table_robustness.py (Markdown version)
=======================
Generate Markdown table for attack robustness results.

Results from Experiment 1: Byzantine attack robustness (F1 scores across 8 attack types).
"""

from pathlib import Path
from typing import Dict


def generate_table_robustness(results: Dict, output_dir: Path) -> Path:
    """
    Generate attack robustness table in Markdown format.
    
    Args:
        results: Dict from experiment_1_attack_robustness with F1 scores per attack class
        output_dir: Directory to save the Markdown table
        
    Returns:
        Path to the generated .md file
    """
    if not results:
        return None
    
    lines = []
    lines.append("# Byzantine Attack Robustness: F1 Scores")
    lines.append("")
    lines.append("| Attack Class | F1 Score | Std Dev |")
    lines.append("|---|---|---|")
    
    attack_order = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    
    for attack_id in attack_order:
        if attack_id in results:
            d = results[attack_id]
            name = d.get("name", attack_id)
            f1 = d.get("f1", 0.0)
            std = d.get("std", 0.0)
            lines.append(f"| {attack_id} ({name}) | {f1:.4f} | {std:.4f} |")
    
    lines.append("")
    
    table_path = output_dir / "table_robustness_empirical.md"
    table_path.write_text("\n".join(lines))
    
    return table_path
