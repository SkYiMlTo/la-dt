"""
table_scalability.py (CSV version)
=======================
Generate CSV table for scalability benchmarks.

Results from Experiment 3: Network size benchmarks (5, 10, 20, 50, 100 sensors).
"""

import csv
from pathlib import Path
from typing import Dict


def generate_table_scalability(results: Dict, output_dir: Path) -> Path:
    """
    Generate scalability benchmarks table in CSV format.
    
    Args:
        results: Dict from experiment_3_scalability with metrics per network size
        output_dir: Directory to save the CSV table
        
    Returns:
        Path to the generated .csv file
    """
    if not results:
        return None
    
    table_path = output_dir / "table_scalability_empirical.csv"
    
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Num Nodes', 'F1 Score', 'Attr. Acc. (%)', 'Inference (ms)', 'Speedup'])
        
        node_counts = ["5", "10", "20", "50", "100"]
        
        for nodes in node_counts:
            if nodes in results:
                d = results[nodes]
                f1 = d.get("f1", 0.0)
                attr_acc = d.get("attribution_acc", 0.0) * 100
                inference_ms = d.get("inference_ms", 0.0)
                speedup = d.get("speedup_ratio", 0.0)
                writer.writerow([
                    nodes,
                    f"{f1:.3f}",
                    f"{attr_acc:.1f}",
                    f"{inference_ms:.2f}",
                    f"{speedup:.2f}x"
                ])
    
    return table_path
