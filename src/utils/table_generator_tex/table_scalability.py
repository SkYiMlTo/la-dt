"""
table_scalability.py
=======================
Generate LaTeX table for scalability benchmarks.

Results from Experiment 3: Network size benchmarks (5, 10, 20, 50, 100 sensors).
"""

from pathlib import Path
from typing import Dict


def generate_table_scalability(results: Dict, output_dir: Path) -> Path:
    """
    Generate scalability benchmarks table.
    
    Args:
        results: Dict from experiment_3_scalability with metrics per network size
        output_dir: Directory to save the LaTeX table
        
    Returns:
        Path to the generated .tex file
    """
    if not results:
        return None
    
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Scalability: GAT Performance vs Network Size}")
    lines.append(r"\label{tab:scalability_empirical}")
    lines.append(r"\begin{tabular}{ccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Num Nodes} & \textbf{F1 Score} & \textbf{Attr. Acc. (\%)} & \textbf{Inference (ms)} & \textbf{Speedup} \\")
    lines.append(r"\midrule")
    
    node_counts = ["5", "10", "20", "50", "100"]
    
    for nodes in node_counts:
        if nodes in results:
            d = results[nodes]
            f1 = d.get("f1", 0.0)
            attr_acc = d.get("attribution_acc", 0.0) * 100  # Convert to percentage
            inference_ms = d.get("inference_ms", 0.0)
            speedup = d.get("speedup_ratio", 0.0)
            lines.append(f"{nodes} & {f1:.3f} & {attr_acc:.1f}\\% & {inference_ms:.2f} & {speedup:.2f}x \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    table_path = output_dir / "table_scalability_empirical.tex"
    table_path.write_text("\n".join(lines))
    
    return table_path
