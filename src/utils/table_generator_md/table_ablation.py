"""
table_ablation.py (Markdown version)
=======================
Generate Markdown table for ablation study results.

Results from Experiment 6: Impact of removing each attribution signal (VGR, DEV, SCD).
"""

from pathlib import Path
from typing import Dict


def generate_table_ablation(results: Dict, output_dir: Path) -> Path:
    """
    Generate ablation study results table in Markdown format.
    
    Args:
        results: Dict from experiment_6_ablation with accuracy and impact per configuration
        output_dir: Directory to save the Markdown table
        
    Returns:
        Path to the generated .md file
    """
    if not results:
        return None
    
    lines = []
    lines.append("# Ablation Study: Attribution Signal Contribution Analysis")
    lines.append("")
    lines.append("| Configuration | Accuracy (%) | Impact (%) | Support |")
    lines.append("|---|---|---|---|")
    
    config_order = ["Full_LA-DT", "w/o_VGR", "w/o_DEV", "w/o_SCD", "VGR_only", "DEV_only"]
    
    for config in config_order:
        if config in results:
            d = results[config]
            acc = d.get("accuracy_pct", 0.0)
            impact = d.get("impact_pct", 0.0)
            correct = d.get("correct", 0)
            total = d.get("total", 0)
            
            if config == "Full_LA-DT":
                impact_str = "baseline"
            else:
                impact_str = f"{impact:+.1f}%"
            
            lines.append(f"| **{config}** | **{acc:.1f}%** | **{impact_str}** | {correct}/{total} |" if config == "Full_LA-DT" else f"| {config} | {acc:.1f}% | {impact_str} | {correct}/{total} |")
    
    lines.append("")
    
    table_path = output_dir / "table_ablation_empirical.md"
    table_path.write_text("\n".join(lines))
    
    return table_path
