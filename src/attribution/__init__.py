"""
Attribution Engine
==================
Classifies detected anomalies as one of three categories:

1. **FDI (False Data Injection)**: Immediate physical constraint violation at t+1.
   The injected data is physically impossible.

2. **Byzantine Attack**: Coordinated, gradual data manipulation across sensors.
   Characterized by exponential variance growth, cross-sensor correlation
   breakdown, and subtle physics-edge-case violations.

3. **Natural Drift**: Benign sensor drift. Linear variance growth, no correlation
   breakdown, physical constraints remain satisfied.

Pipeline:
    physics_check (t+1) → variance_analysis (multi-horizon) → correlation_check → verdict
"""

import math
import time
import uuid
import random
from dataclasses import dataclass, field


@dataclass
class AttributionResult:
    """Complete attribution result for a detected anomaly."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = ""
    verdict: str = "unknown"  # "fdi", "byzantine", "natural_drift"
    confidence: float = 0.0
    trigger_node: int = 0
    trigger_sensor: str = ""
    trigger_value: float = 0.0
    trigger_z_score: float = 0.0

    # Physics check
    physics_violation: bool = False
    physics_violations: list = field(default_factory=list)

    # Variance analysis
    variance_growth_rate: str = "linear"  # "linear", "exponential", "none"
    variance_ratios: dict = field(default_factory=dict)

    # Correlation analysis
    correlation_breakdown: bool = False
    correlation_scores: dict = field(default_factory=dict)

    # Horizon analysis
    horizons_flagged: list = field(default_factory=list)
    total_violations_by_horizon: dict = field(default_factory=dict)

    # Trajectories (for dashboard)
    simulation_id: str = ""
    trajectories: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "trigger": {
                "node_id": self.trigger_node,
                "sensor": self.trigger_sensor,
                "value": round(self.trigger_value, 4),
                "z_score": round(self.trigger_z_score, 4),
            },
            "attribution": {
                "verdict": self._verdict_label(),
                "confidence": round(self.confidence * 100, 1),
                "physics_violation": self.physics_violation,
                "physics_violations": self.physics_violations,
                "variance_growth_rate": self.variance_growth_rate,
                "variance_ratios": self.variance_ratios,
                "correlation_breakdown": self.correlation_breakdown,
                "correlation_scores": self.correlation_scores,
                "horizons_flagged": self.horizons_flagged,
                "total_violations_by_horizon": self.total_violations_by_horizon,
            },
            "simulation_id": self.simulation_id,
        }

    def _verdict_label(self) -> str:
        labels = {
            "fdi": "False Data Injection",
            "byzantine": "Byzantine Attack",
            "natural_drift": "Natural Drift",
            "unknown": "Unknown",
        }
        return labels.get(self.verdict, self.verdict)


class AttributionEngine:
    """
    Analyzes multi-horizon simulation results and physics checks to
    classify anomalies.
    """

    # Thresholds
    VARIANCE_EXP_THRESHOLD = 2.0   # ratio threshold for "exponential" growth
    CORRELATION_THRESHOLD = 0.8    # raised to catch subtle masking by noise
    FDI_CONFIDENCE = 0.95
    BYZANTINE_CONFIDENCE_BASE = 0.80
    DRIFT_CONFIDENCE_BASE = 0.70
    LLR_BYZANTINE_THRESHOLD = 1.5  # log-likelihood ratio threshold for Byzantine verdict

    def analyze(
        self,
        scoring_result: dict,
        physics_check: dict,
        simulation_results: dict,
    ) -> AttributionResult:
        """
        Full attribution analysis.

        Args:
            scoring_result: Output from AnomalyScorer (trigger info).
            physics_check: Output from simulator /check-physics (t+1).
            simulation_results: Output from simulator /simulate (multi-horizon).

        Returns:
            AttributionResult with verdict and all analysis details.
        """
        result = AttributionResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            trigger_node=scoring_result.get("node_id", 0),
            trigger_sensor=scoring_result.get("sensor", ""),
            trigger_value=scoring_result.get("value", 0.0),
            trigger_z_score=scoring_result.get("z_score", 0.0),
            simulation_id=uuid.uuid4().hex[:8],
        )

        # --- Step 1: Variance growth analysis across horizons ---
        horizons_data = simulation_results.get("horizons", {})
        result.variance_ratios = self._compute_variance_ratios(horizons_data)
        result.variance_growth_rate = self._classify_variance_growth(
            result.variance_ratios
        )

        # --- Step 2: Violation count per horizon ---
        result.total_violations_by_horizon = self._count_violations(horizons_data)
        result.horizons_flagged = [
            h for h, count in result.total_violations_by_horizon.items()
            if count > 0
        ]

        # --- Step 3: Cross-sensor correlation analysis ---
        result.correlation_scores = self._compute_correlation(horizons_data)
        result.correlation_breakdown = any(
            score < self.CORRELATION_THRESHOLD
            for score in result.correlation_scores.values()
        )

        # --- Step 4: Physics check (instant t+1) ---
        result.physics_violation = not physics_check.get("valid", True)
        result.physics_violations = physics_check.get("violations", [])

        if result.physics_violation:
            # Check if this is "Drift-to-Failure" (Byzantine) or "Instant Injection" (FDI)
            if result.variance_growth_rate == "exponential":
                # Signatures of drift exist, so it's likely a Byzantine attack that crossed bounds
                result.verdict = "byzantine"
                result.confidence = 0.92
                return result
            else:
                # No history of drift/breakdown -> Pure FDI
                result.verdict = "fdi"
                result.confidence = self.FDI_CONFIDENCE
                result.horizons_flagged = ["t+1"]
                return result

        # --- Step 5: Classification ---
        if (
            result.variance_growth_rate == "exponential"
            and result.correlation_breakdown
        ):
            result.verdict = "byzantine"
            # Confidence scales with z-score magnitude and variance ratio
            max_ratio = max(result.variance_ratios.values(), default=1.0)
            result.confidence = min(
                self.BYZANTINE_CONFIDENCE_BASE
                + 0.05 * min(result.trigger_z_score, 4)
                + 0.02 * min(max_ratio, 5),
                0.99,
            )
        elif (
            result.variance_growth_rate == "exponential"
            and not result.correlation_breakdown
        ):
            # Exponential variance but sensors still correlated → likely attack
            result.verdict = "byzantine"
            result.confidence = self.BYZANTINE_CONFIDENCE_BASE
        elif result.variance_growth_rate == "linear" and len(result.horizons_flagged) == 0:
            result.verdict = "natural_drift"
            result.confidence = self.DRIFT_CONFIDENCE_BASE + 0.05 * min(
                result.trigger_z_score - 2, 3
            )
        else:
            # Edge case: some violations but linear growth
            if len(result.horizons_flagged) >= 2:
                result.verdict = "byzantine"
                result.confidence = 0.65
            else:
                result.verdict = "natural_drift"
                result.confidence = 0.60

        return result

    def analyze_at_horizon(
        self,
        scoring_result: dict,
        physics_check: dict,
        simulation_results: dict,
        max_horizon: int,
        ablation_mode: str = "full",
    ) -> str:
        """
        Independent classification decision using only data up to `max_horizon`.

        This simulates what the system would decide if it only had
        forecast data up to the given horizon (e.g., 300s = 5min).

        Args:
            ablation_mode: One of "full", "no_variance", "no_correlation",
                           "no_physics", "no_zscore". Disables the named
                           evidence signal for ablation experiments.

        Returns:
            verdict string: "fdi", "byzantine", or "natural_drift"
        """
        # --- Step 1: Filter horizons data to only include <= max_horizon ---
        all_horizons = simulation_results.get("horizons", {})
        filtered = {
            h: data for h, data in all_horizons.items()
            if int(h) <= max_horizon
        }

        if not filtered:
            return "natural_drift"

        # --- Step 2: Variance growth analysis on restricted horizons ---
        variance_ratios = self._compute_variance_ratios(filtered)
        variance_growth = self._classify_variance_growth(variance_ratios)

        # --- Step 3: Violation count on restricted horizons ---
        violations = self._count_violations(filtered)
        horizons_flagged = [h for h, c in violations.items() if c > 0]

        # --- Step 4: Physics check (same for all horizons) ---
        if ablation_mode != "no_physics" and not physics_check.get("valid", True):
            if variance_growth == "exponential":
                return "byzantine"
            else:
                return "fdi"

        # --- Step 5: Correlation analysis on restricted horizons ---
        correlations = self._compute_correlation(filtered)
        correlation_breakdown = any(
            score < self.CORRELATION_THRESHOLD
            for score in correlations.values()
        )

        # --- Step 6: Log-likelihood ratio scoring ---
        # We compute a log-likelihood ratio (LLR) for the hypothesis
        # H_B (Byzantine) vs H_N (Natural Drift).  Each signal contributes
        # an independent log-likelihood term under the assumption that
        # Byzantine attacks produce exponential variance growth, anti-
        # correlated sensors, physics violations, and elevated z-scores,
        # while natural drift produces linear/bounded variance, correlated
        # sensors, no violations, and moderate z-scores.
        #
        # LLR = sum_k log[ P(signal_k | H_B) / P(signal_k | H_N) ]
        #
        # We model each signal's likelihood ratio using calibrated sigmoid
        # mappings derived from the training distribution.

        num_horizons = len(filtered)
        llr = 0.0  # log-likelihood ratio

        # Signal 1: Trigger z-score
        if ablation_mode != "no_zscore":
            z_score = scoring_result.get("z_score", 0.0)
            # P(z|B) >> P(z|N) for z >> 3.5; calibrated from training data
            llr += self._sigmoid_llr(z_score, midpoint=4.0, steepness=1.5,
                                      max_llr=1.0)

        # Signal 2: Correlation breakdown (horizon-scaled)
        if ablation_mode != "no_correlation":
            min_corr = min(correlations.values()) if correlations else 1.0
            horizon_scale = min(num_horizons / 4.0, 1.0)
            # Byzantine: anti-correlated (low min_corr); Natural: correlated
            # Invert: lower correlation → higher LLR
            corr_signal = 1.0 - min_corr  # 0 = perfect corr, 2 = anti-corr
            llr += self._sigmoid_llr(corr_signal, midpoint=0.6, steepness=3.0,
                                      max_llr=3.5) * horizon_scale

        # Signal 3: Variance growth pattern
        if ablation_mode != "no_variance":
            if variance_ratios:
                avg_ratio = sum(variance_ratios.values()) / len(variance_ratios)
                # Exponential growth: avg_ratio >> 2; linear: avg_ratio ≈ 1
                llr += self._sigmoid_llr(avg_ratio, midpoint=1.8, steepness=2.0,
                                          max_llr=3.5)

        # Signal 4: Physics violations
        if ablation_mode != "no_physics":
            total_violations = sum(violations.values())
            llr += self._sigmoid_llr(total_violations, midpoint=1.5, steepness=2.0,
                                      max_llr=2.5)

        # Decision: LLR > 0 means H_B is more likely than H_N
        # We use a threshold > 0 to control the false-positive rate
        # (equivalent to requiring posterior P(B|data) > prior)
        if llr >= self.LLR_BYZANTINE_THRESHOLD:
            return "byzantine"
        else:
            return "natural_drift"

    @staticmethod
    def _sigmoid_llr(x, midpoint, steepness, max_llr):
        """
        Compute a log-likelihood ratio contribution using a sigmoid mapping.

        Returns a value in [-max_llr, +max_llr]:
          - x << midpoint → negative LLR (evidence for natural drift)
          - x >> midpoint → positive LLR (evidence for Byzantine attack)
          - x ≈ midpoint → LLR ≈ 0 (uninformative)

        The sigmoid is: LLR = max_llr * (2 * sigmoid(steepness*(x - midpoint)) - 1)
        """
        z = steepness * (x - midpoint)
        z = max(-20, min(20, z))  # clip for numerical stability
        sigmoid = 1.0 / (1.0 + math.exp(-z))
        return max_llr * (2.0 * sigmoid - 1.0)

    # ------------------------------------------------------------------
    # Internal analysis methods
    # ------------------------------------------------------------------

    def _compute_variance_ratios(self, horizons_data: dict) -> dict:
        """
        Compute variance ratio between consecutive horizons.
        Exponential growth → ratio increases; linear → ratio ≈ 1.
        """
        sorted_horizons = sorted(horizons_data.keys(), key=lambda h: int(h))
        variances = {}

        for h in sorted_horizons:
            nodes = horizons_data[h]
            temps = []
            for nid, data in nodes.items():
                traj = data.get("trajectory", [])
                if traj:
                    values = [p["temperature"] for p in traj]
                    if len(values) > 1:
                        mean = sum(values) / len(values)
                        var = sum((v - mean) ** 2 for v in values) / len(values)
                        temps.append(var)
            variances[h] = sum(temps) / len(temps) if temps else 0.0

        # Compute ratios between consecutive horizons
        ratios = {}
        prev_var = None
        prev_h = None
        for h in sorted_horizons:
            if prev_var is not None and prev_var > 0:
                ratios[f"{prev_h}->{h}"] = round(variances[h] / prev_var, 4)
            prev_var = variances[h]
            prev_h = h

        return ratios

    def _classify_variance_growth(self, ratios: dict) -> str:
        """Classify variance growth pattern."""
        if not ratios:
            return "none"

        values = list(ratios.values())
        # Check if ratios are increasing (exponential signature)
        if len(values) >= 2:
            increasing = all(
                values[i] > self.VARIANCE_EXP_THRESHOLD
                for i in range(len(values))
            )
            if increasing:
                return "exponential"

        avg_ratio = sum(values) / len(values)
        if avg_ratio > self.VARIANCE_EXP_THRESHOLD:
            return "exponential"

        return "linear"

    def _count_violations(self, horizons_data: dict) -> dict:
        """Count total physics violations per horizon."""
        counts = {}
        for h, nodes in horizons_data.items():
            total = 0
            for nid, data in nodes.items():
                total += len(data.get("violations", []))
            counts[h] = total
        return counts

    def _compute_correlation(self, horizons_data: dict) -> dict:
        """
        Compute pairwise correlation between nodes within each horizon.
        Returns the minimum correlation found per horizon.
        Low correlation → sensors disagree → possible Byzantine behavior.
        """
        correlations = {}

        for h, nodes in horizons_data.items():
            # Extract temperature trajectories per node
            node_temps = {}
            for nid, data in nodes.items():
                traj = data.get("trajectory", [])
                if traj:
                    node_temps[nid] = [p["temperature"] for p in traj]

            if len(node_temps) < 2:
                correlations[h] = 1.0
                continue

            # Pairwise Pearson correlation
            nids = list(node_temps.keys())
            min_corr = 1.0
            for i in range(len(nids)):
                for j in range(i + 1, len(nids)):
                    corr = self._pearson(
                        node_temps[nids[i]], node_temps[nids[j]]
                    )
                    min_corr = min(min_corr, corr)

            correlations[h] = round(min_corr, 4)

        return correlations

    @staticmethod
    def _pearson(x: list, y: list) -> float:
        """Compute Pearson correlation coefficient between two lists."""
        n = min(len(x), len(y))
        if n < 2:
            return 1.0

        x, y = x[:n], y[:n]
        mx = sum(x) / n
        my = sum(y) / n

        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))

        if den_x * den_y == 0:
            return 1.0

        return num / (den_x * den_y)
