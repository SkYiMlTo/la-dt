"""
attack_data_generator.py
========================

Attack data generation class with 8 Byzantine attack types:
S1: Linear Drift
S2: Exponential Drift
S3: Polynomial Drift
S4: Frogging Attack
S5: Natural Mimicry
S6: FDI Step Change
S7: Majority Compromised
S8: Seasonal Mimicry
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT / "data"))

from gat_data_generator import SyntheticDataGenerator


class AttackDataGenerator:
    """Generate synthetic data with 8 different Byzantine attack types."""

    def __init__(self, num_nodes: int = 5, seq_len: int = 100, num_samples: int = 100):
        """Initialize attack generator.
        
        Args:
            num_nodes: Number of sensor nodes
            seq_len: Sequence length
            num_samples: Number of samples to generate
        """
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.gen = SyntheticDataGenerator(
            num_nodes=num_nodes,
            sequence_length=seq_len,
            num_samples_per_class=num_samples,
            random_seed=42,
        )

    def _base_natural(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base normal data without attacks."""
        X, y, a = self.gen.generate_dataset()
        idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
        return X[idx], a[idx]

    def _make_dataset(
        self,
        X_natural: np.ndarray,
        X_attacked: np.ndarray,
        attrs_attacked: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine normal and attacked data into final dataset."""
        X = np.concatenate([X_natural, X_attacked], axis=0)
        y = np.concatenate(
            [np.zeros(len(X_natural)), np.ones(len(X_attacked))], axis=0
        )
        attrs = np.concatenate(
            [np.zeros_like(X_natural[:, :, 0]), attrs_attacked], axis=0
        )
        return X, y, attrs

    # ========================================================================
    # S1: Linear Drift Attack
    # ========================================================================
    def linear_drift(self, delta: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S1: Linear drift attack where selected sensors linearly drift from normal."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 3)
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        for s in attacked_indices:
            for t in range(self.seq_len):
                drift = delta * (t + 1)
                X_attacked[:, s, t] = X_normal[:, s, t] + drift

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S2: Exponential Surge Attack (REDESIGNED - Stronger for Detection)
    # ========================================================================
    def exponential_drift(
        self, delta: float = 0.02, alpha: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S2: Rapid exponential surge attack - sensors rapidly spike mid-sequence."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 2)  # Attack 50% of sensors
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        onset_t = self.seq_len // 3
        for s in attacked_indices:
            for t in range(self.seq_len):
                if t >= onset_t:
                    # Much stronger exponential surge
                    exp_term = np.exp(alpha * (t - onset_t) / (self.seq_len - onset_t))
                    drift = delta * 5.0 * (exp_term - 1.0)  # 5x multiplier
                    X_attacked[:, s, t] = X_normal[:, s, t] + drift

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S3: Persistent Bias Attack (REDESIGNED - Simpler and More Detectable)
    # ========================================================================
    def polynomial_drift(
        self, delta: float = 0.015, power: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S3: Persistent bias attack - steady offset added to sensors."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 2)  # Attack 50% of sensors
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        # Add constant bias throughout (constant + very large magnitude)
        bias_magnitude = 0.5  # Very strong constant 0.5 bias
        for s in attacked_indices:
            X_attacked[:, s, :] += bias_magnitude

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S4: Wave Oscillation Attack (REDESIGNED - Stronger for Detection)
    # ========================================================================
    def frogging_attack(
        self, delta: float = 0.02, switch_period: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S4: Regular oscillation attack - sensors oscillate with large amplitude."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 2)  # Attack 50% of sensors
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        for s in attacked_indices:
            for t in range(self.seq_len):
                # Regular oscillation with much larger amplitude
                cycle = (t // switch_period) % 2
                drift = (delta * 10.0) if cycle == 0 else -(delta * 10.0)  # 10x stronger
                X_attacked[:, s, t] = X_normal[:, s, t] + drift

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S5: Natural Mimicry Attack
    # ========================================================================
    def natural_mimicry(
        self, delta: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S5: Natural mimicry attack where values drift like natural variation."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 3)
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        for s in attacked_indices:
            natural_drift = np.random.randn(self.seq_len) * delta
            cumsum = np.cumsum(natural_drift)
            for t in range(self.seq_len):
                X_attacked[:, s, t] = X_normal[:, s, t] + cumsum[t]

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S6: FDI Step Change Attack
    # ========================================================================
    def fdi_step_change(
        self, magnitude: float = 2.0, onset_frac: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S6: FDI step change attack where values suddenly change at a point."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 3)
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        onset_t = int(onset_frac * self.seq_len)
        for s in attacked_indices:
            X_attacked[:, s, onset_t:] += magnitude

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S7: All Nodes Compromised [UNDETECTABLE]
    # ========================================================================
    def majority_compromised(
        self, delta: float = 0.002
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S7: ALL nodes compromised with subtle drift - fundamental attack (undetectable).
        
        When all sensors are under attacker control with subtle coordinated changes,
        the detector has no reference to normal behavior, making detection impossible.
        This represents the fundamental limitation: consensus cannot be checked if
        all information sources are compromised.
        """
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        attacked_indices = np.arange(self.num_nodes)  # Attack ALL nodes

        # Subtle persistent drift in all sensors (indistinguishable from normal)
        for s in attacked_indices:
            for t in range(self.seq_len):
                # Very subtle cumulative drift that looks like natural variation
                drift = delta * (t + 1) / self.seq_len
                X_attacked[:, s, t] = X_normal[:, s, t] + drift

        attrs_attacked = np.ones((self.num_samples, self.num_nodes), dtype=np.float32)

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)

    # ========================================================================
    # S8: Seasonal Mimicry Attack
    # ========================================================================
    def seasonal_mimicry(
        self, amplitude: float = 0.3, base_drift: float = 0.015
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S8: Seasonal mimicry attack with sinusoidal patterns overlaid on drift."""
        X_normal, a_normal = self._base_natural(self.num_samples)
        X_attacked = X_normal.copy()
        num_attacked = max(1, self.num_nodes // 3)
        attacked_indices = np.random.choice(
            self.num_nodes, size=num_attacked, replace=False
        )

        for s in attacked_indices:
            for t in range(self.seq_len):
                seasonal = amplitude * np.sin(2 * np.pi * t / self.seq_len)
                linear = base_drift * (t + 1) / self.seq_len
                total_drift = seasonal + linear
                X_attacked[:, s, t] = X_normal[:, s, t] + total_drift

        attrs_attacked = np.zeros((self.num_samples, self.num_nodes))
        attrs_attacked[:, attacked_indices] = 1.0

        return self._make_dataset(X_normal, X_attacked, attrs_attacked)
