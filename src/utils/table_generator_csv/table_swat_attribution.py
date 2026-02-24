"""
table_swat_attribution.py (CSV version)
=======================
Generate CSV table for SWAT attribution analysis results.

Results from Experiment 7: Attribution accuracy on real SWAT data at multiple horizons.
"""

import csv
from pathlib import Path
from typing import Dict


def generate_table_swat_attribution(results: Dict, output_dir: Path) -> Path:
    """
    Generate SWAT attribution analysis table in CSV format.
    
    Args:
        results: Dict from experiment_7_swat_attribution with accuracies per horizon
        output_dir: Directory to save the CSV table
        
    Returns:
        Path to the generated .csv file
    """
    if not results:
        return None
    
    table_path = output_dir / "table_swat_attribution_empirical.csv"
    
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Horizon (min)', 'Correct', 'Total', 'Accuracy (%)', 'Avg VGR', 'Avg LLR'])
        
        for h in ["5", "10", "30", "60"]:
            if h in results:
                d = results[h]
                writer.writerow([
                    h,
                    d['correct'],
                    d['total'],
                    f"{d['accuracy_pct']:.1f}",
                    f"{d['avg_vgr']:.2f}",
                    f"{d['avg_llr']:.2f}"
                ])
    
    return table_path
