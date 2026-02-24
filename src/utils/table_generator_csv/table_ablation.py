"""
table_ablation.py (CSV version)
=======================
Generate CSV table for ablation study results.

Results from Experiment 6: Impact of removing each attribution signal (VGR, DEV, SCD).
"""

import csv
from pathlib import Path
from typing import Dict


def generate_table_ablation(results: Dict, output_dir: Path) -> Path:
    """
    Generate ablation study results table in CSV format.
    
    Args:
        results: Dict from experiment_6_ablation with accuracy and impact per configuration
        output_dir: Directory to save the CSV table
        
    Returns:
        Path to the generated .csv file
    """
    if not results:
        return None
    
    table_path = output_dir / "table_ablation_empirical.csv"
    
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Configuration', 'Accuracy (%)', 'Impact (%)', 'Correct', 'Total'])
        
        config_order = ["Full_LA-DT", "w/o_VGR", "w/o_DEV", "w/o_SCD", "VGR_only", "DEV_only"]
        
        for config in config_order:
            if config in results:
                d = results[config]
                acc = d.get("accuracy_pct", 0.0)
                impact = d.get("impact_pct", 0.0)
                correct = d.get("correct", 0)
                total = d.get("total", 0)
                
                impact_str = "baseline" if config == "Full_LA-DT" else f"{impact:+.1f}"
                writer.writerow([config, f"{acc:.1f}", impact_str, correct, total])
    
    return table_path
