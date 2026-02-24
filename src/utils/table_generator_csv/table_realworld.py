"""
table_realworld.py (CSV version)
=======================
Generate CSV table for real-world validation results.

Results from Experiment 4 (SWAT) and Experiment 5 (AI Dataset): 
Real-world performance on SWAT water treatment facility and AI power grid data.
"""

import csv
from pathlib import Path
from typing import Dict


def generate_table_realworld(swat_results: Dict, ai_results: Dict, output_dir: Path) -> Path:
    """
    Generate real-world validation results table in CSV format.
    
    Args:
        swat_results: Dict from experiment_4_swat
        ai_results: Dict from experiment_5_ai_dataset
        output_dir: Directory to save the CSV table
        
    Returns:
        Path to the generated .csv file
    """
    if not swat_results and not ai_results:
        return None
    
    table_path = output_dir / "table_realworld_empirical.csv"
    
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'F1 Score', 'Accuracy', 'Num Sensors', 'Train Time (s)'])
        
        # SWAT results
        if swat_results:
            f1 = swat_results.get("f1", 0.0)
            acc = swat_results.get("accuracy", 0.0)
            num_sensors = swat_results.get("num_sensors", "N/A")
            train_time = swat_results.get("train_time_s", 0.0)
            writer.writerow(['SWAT (ICS)', f"{f1:.3f}", f"{acc:.3f}", num_sensors, f"{train_time:.2f}"])
        
        # AI Dataset results
        if ai_results:
            f1 = ai_results.get("f1", 0.0)
            acc = ai_results.get("accuracy", 0.0)
            num_sensors = ai_results.get("num_sensors", "N/A")
            train_time = ai_results.get("train_time_s", 0.0)
            writer.writerow(['AI Power Grid', f"{f1:.3f}", f"{acc:.3f}", num_sensors, f"{train_time:.2f}"])
    
    return table_path
