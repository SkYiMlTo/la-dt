"""
Initialize table_generator module.
Exports all table generation functions for easy importing.
"""

from .table_robustness import generate_table_robustness
from .table_horizons import generate_table_horizons
from .table_scalability import generate_table_scalability
from .table_realworld import generate_table_realworld
from .table_ablation import generate_table_ablation
from .table_swat_attribution import generate_table_swat_attribution

__all__ = [
    "generate_table_robustness",
    "generate_table_horizons",
    "generate_table_scalability",
    "generate_table_realworld",
    "generate_table_ablation",
    "generate_table_swat_attribution",
]
