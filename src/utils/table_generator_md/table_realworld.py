"""
table_realworld.py (Markdown version)
=======================
Generate Markdown table for real-world validation results.

Results from Experiment 4 (SWAT) and Experiment 5 (AI Dataset): 
Real-world performance on SWAT water treatment facility and AI power grid data.
"""

from pathlib import Path
from typing import Dict


def generate_table_realworld(swat_results: Dict, ai_results: Dict, output_dir: Path) -> Path:
    """
    Generate real-world validation results table in Markdown format.
    
    Args:
        swat_results: Dict from experiment_4_swat
        ai_results: Dict from experiment_5_ai_dataset
        output_dir: Directory to save the Markdown table
        
    Returns:
        Path to the generated .md file
    """
    if not swat_results and not ai_results:
        return None
    
    lines = []
    lines.append("# Real-World Validation: SWAT & AI Dataset Performance")
    lines.append("")
    lines.append("| Dataset | F1 Score | Accuracy | Num Sensors | Train Time (s) |")
    lines.append("|---|---|---|---|---|")
    
    # SWAT results
    if swat_results:
        f1 = swat_results.get("f1", 0.0)
        acc = swat_results.get("accuracy", 0.0)
        num_sensors = swat_results.get("num_sensors", "N/A")
        train_time = swat_results.get("train_time_s", 0.0)
        lines.append(f"| SWAT (ICS) | {f1:.3f} | {acc:.3f} | {num_sensors} | {train_time:.2f} |")
    
    # AI Dataset results
    if ai_results:
        f1 = ai_results.get("f1", 0.0)
        acc = ai_results.get("accuracy", 0.0)
        num_sensors = ai_results.get("num_sensors", "N/A")
        train_time = ai_results.get("train_time_s", 0.0)
        lines.append(f"| AI Power Grid | {f1:.3f} | {acc:.3f} | {num_sensors} | {train_time:.2f} |")
    
    lines.append("")
    
    table_path = output_dir / "table_realworld_empirical.md"
    table_path.write_text("\n".join(lines))
    
    return table_path
