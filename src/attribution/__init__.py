"""
Attribution module - Byzantine attack attribution pipeline.

Analyzes sensor readings to determine if deviations are due to:
- Byzantine attacks (compromised sensors)
- Natural drift (legitimate environmental changes)

Uses three evidence signals:
- VGR (Variance Growth Ratio)
- DEV (Deviation Ratio)
- SCD (Sensor Correlation Decay)

Combined via log-likelihood ratio (LLR) for final verdict.
"""

from .attribution_pipeline import run_attribution_at_horizon

__all__ = ["run_attribution_at_horizon"]
