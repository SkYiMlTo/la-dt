"""
table_horizons.py (Markdown version)
=======================
Generate Markdown table for multi-horizon attribution accuracy.

Results from Experiment 2: Attribution accuracy at 5, 10, 30, 60-minute horizons.
"""

from pathlib import Path
from typing import Dict


def generate_table_horizons(results: Dict, output_dir: Path) -> Path:
    """
    Generate multi-horizon attribution table in Markdown format.
    
    Args:
        results: Dict from experiment_2_multi_horizon with accuracies per horizon
        output_dir: Directory to save the Markdown table
        
    Returns:
        Path to the generated .md file
    """
    if not results:
        return None
    
    lines = []
    lines.append("# Attribution Accuracy at Multiple Detection Horizons")
    lines.append("")
    lines.append("| Horizon (min) | Correct | Accuracy (%) | Avg VGR | Avg LLR |")
    lines.append("|---|---|---|---|---|")
    
    for h in ["5", "10", "30", "60"]:
        if h in results:
            d = results[h]
            lines.append(f"| {h} | {d['correct']:2d}/{d['total']:2d} | {d['accuracy_pct']:5.1f}% | {d['avg_vgr']:.2f} | {d['avg_llr']:.2f} |")
    
    lines.append("")
    
    table_path = output_dir / "table_horizon_empirical.md"
    table_path.write_text("\n".join(lines))
    
    return table_path
