"""
table_realworld.py
=======================
Generate LaTeX table for real-world validation results.

Results from Experiment 4 (SWAT) and Experiment 5 (AI Dataset): 
Real-world performance on SWAT water treatment facility and AI power grid data.
"""

from pathlib import Path
from typing import Dict


def generate_table_realworld(swat_results: Dict, ai_results: Dict, output_dir: Path) -> Path:
    """
    Generate real-world validation results table.
    
    Args:
        swat_results: Dict from experiment_4_swat
        ai_results: Dict from experiment_5_ai_dataset
        output_dir: Directory to save the LaTeX table
        
    Returns:
        Path to the generated .tex file
    """
    if not swat_results and not ai_results:
        return None
    
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Real-World Validation: SWAT \& AI Dataset Performance}")
    lines.append(r"\label{tab:realworld_empirical}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{F1 Score} & \textbf{Accuracy} & \textbf{Num Sensors} & \textbf{Train Time (s)} \\")
    lines.append(r"\midrule")
    
    # SWAT results
    if swat_results:
        f1 = swat_results.get("f1", 0.0)
        acc = swat_results.get("accuracy", 0.0)
        num_sensors = swat_results.get("num_sensors", "N/A")
        train_time = swat_results.get("train_time_s", 0.0)
        lines.append(f"SWAT (ICS) & {f1:.3f} & {acc:.3f} & {num_sensors} & {train_time:.2f} \\\\")
    
    # AI Dataset results
    if ai_results:
        f1 = ai_results.get("f1", 0.0)
        acc = ai_results.get("accuracy", 0.0)
        num_sensors = ai_results.get("num_sensors", "N/A")
        train_time = ai_results.get("train_time_s", 0.0)
        lines.append(f"AI Power Grid & {f1:.3f} & {acc:.3f} & {num_sensors} & {train_time:.2f} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    table_path = output_dir / "table_realworld_empirical.tex"
    table_path.write_text("\n".join(lines))
    
    return table_path
