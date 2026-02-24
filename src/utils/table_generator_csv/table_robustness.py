"""
table_robustness.py (CSV version)
=======================
Generate CSV table for attack robustness results.

Results from Experiment 1: Byzantine attack robustness (F1 scores across 8 attack types).
"""

import csv
from pathlib import Path
from typing import Dict


def generate_table_robustness(results: Dict, output_dir: Path) -> Path:
    """
    Generate attack robustness table in CSV format.
    
    Args:
        results: Dict from experiment_1_attack_robustness with F1 scores per attack class
        output_dir: Directory to save the CSV table
        
    Returns:
        Path to the generated .csv file
    """
    if not results:
        return None
    
    table_path = output_dir / "table_robustness_empirical.csv"
    
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Attack Class', 'Name', 'F1 Score', 'Std Dev'])
        
        attack_order = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
        
        for attack_id in attack_order:
            if attack_id in results:
                d = results[attack_id]
                name = d.get("name", attack_id)
                f1 = d.get("f1", 0.0)
                std = d.get("std", 0.0)
                writer.writerow([attack_id, name, f"{f1:.4f}", f"{std:.4f}"])
    
    return table_path
