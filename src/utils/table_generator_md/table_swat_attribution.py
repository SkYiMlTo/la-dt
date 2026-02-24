"""
table_swat_attribution.py (Markdown version)
=======================
Generate Markdown table for SWAT attribution analysis results.

Results from Experiment 7: Attribution accuracy on real SWAT data at multiple horizons.
"""

from pathlib import Path
from typing import Dict


def generate_table_swat_attribution(results: Dict, output_dir: Path) -> Path:
    """
    Generate SWAT attribution analysis table in Markdown format.
    
    Args:
        results: Dict from experiment_7_swat_attribution with accuracies per horizon
        output_dir: Directory to save the Markdown table
        
    Returns:
        Path to the generated .md file
    """
    if not results:
        return None
    
    lines = []
    lines.append("# SWAT Attribution Analysis: Accuracy at Multiple Horizons")
    lines.append("")
    lines.append("| Horizon (min) | Correct | Accuracy (%) | Avg VGR | Avg LLR |")
    lines.append("|---|---|---|---|---|")
    
    for h in ["5", "10", "30", "60"]:
        if h in results:
            d = results[h]
            lines.append(f"| {h} | {d['correct']:2d}/{d['total']:2d} | {d['accuracy_pct']:5.1f}% | {d['avg_vgr']:.2f} | {d['avg_llr']:.2f} |")
    
    lines.append("")
    
    table_path = output_dir / "table_swat_attribution_empirical.md"
    table_path.write_text("\n".join(lines))
    
    return table_path
