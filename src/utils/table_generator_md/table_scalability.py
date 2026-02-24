"""
table_scalability.py (Markdown version)
=======================
Generate Markdown table for scalability benchmarks.

Results from Experiment 3: Network size benchmarks (5, 10, 20, 50, 100 sensors).
"""

from pathlib import Path
from typing import Dict


def generate_table_scalability(results: Dict, output_dir: Path) -> Path:
    """
    Generate scalability benchmarks table in Markdown format.
    
    Args:
        results: Dict from experiment_3_scalability with metrics per network size
        output_dir: Directory to save the Markdown table
        
    Returns:
        Path to the generated .md file
    """
    if not results:
        return None
    
    lines = []
    lines.append("# Scalability: GAT Performance vs Network Size")
    lines.append("")
    lines.append("| Num Nodes | F1 Score | Attr. Acc. (%) | Inference (ms) | Speedup |")
    lines.append("|---|---|---|---|---|")
    
    node_counts = ["5", "10", "20", "50", "100"]
    
    for nodes in node_counts:
        if nodes in results:
            d = results[nodes]
            f1 = d.get("f1", 0.0)
            attr_acc = d.get("attribution_acc", 0.0) * 100  # Convert to percentage
            inference_ms = d.get("inference_ms", 0.0)
            speedup = d.get("speedup_ratio", 0.0)
            lines.append(f"| {nodes} | {f1:.3f} | {attr_acc:.1f}% | {inference_ms:.2f} | {speedup:.2f}x |")
    
    lines.append("")
    
    table_path = output_dir / "table_scalability_empirical.md"
    table_path.write_text("\n".join(lines))
    
    return table_path
