"""Consensus result value object."""

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..exceptions import InsufficientDataError, ValidationError


@dataclass(frozen=True)
class ConsensusResult:
    """Value object representing consensus from multiple judges."""

    consensus_score: Decimal
    confidence_interval: Tuple[Decimal, Decimal]  # (lower_bound, upper_bound)
    agreement_level: Decimal  # Inter-rater reliability (0-1)
    judge_scores: Dict[str, Decimal]  # judge_id -> score
    judge_weights: Dict[str, Decimal]  # judge_id -> weight
    outlier_judges: List[str]  # judge_ids identified as outliers
    statistical_significance: Decimal  # p-value or similar
    consensus_method: str  # "weighted_average", "median", "trimmed_mean"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate consensus result."""
        if not (0 <= self.consensus_score <= 1):
            raise ValidationError("Consensus score must be between 0 and 1")

        if not (0 <= self.agreement_level <= 1):
            raise ValidationError("Agreement level must be between 0 and 1")

        if not (0 <= self.statistical_significance <= 1):
            raise ValidationError("Statistical significance must be between 0 and 1")

        if self.confidence_interval[0] > self.confidence_interval[1]:
            raise ValidationError("Invalid confidence interval bounds")

        if len(self.judge_scores) < 2:
            raise ValidationError("Consensus requires at least 2 judge scores")

        if set(self.judge_scores.keys()) != set(self.judge_weights.keys()):
            raise ValidationError("Judge scores and weights must have matching keys")

        # Validate weights sum to 1
        total_weight = sum(self.judge_weights.values())
        if abs(total_weight - 1) > Decimal("0.001"):
            raise ValidationError(f"Judge weights must sum to 1.0, got {total_weight}")

        # Validate consensus method
        valid_methods = ["weighted_average", "median", "trimmed_mean", "robust_average"]
        if self.consensus_method not in valid_methods:
            raise ValidationError(f"Invalid consensus method: {self.consensus_method}")

    @classmethod
    def create_simple_consensus(
        cls, judge_scores: Dict[str, Decimal], method: str = "weighted_average"
    ) -> "ConsensusResult":
        """Create simple consensus with equal weights."""
        if len(judge_scores) < 2:
            raise InsufficientDataError("Need at least 2 judges for consensus")

        # Equal weights for all judges
        equal_weight = Decimal("1.0") / len(judge_scores)
        judge_weights = {judge_id: equal_weight for judge_id in judge_scores.keys()}

        # Calculate consensus score
        if method == "weighted_average":
            consensus_score = sum(
                score * judge_weights[judge_id] for judge_id, score in judge_scores.items()
            )
        elif method == "median":
            scores_list = sorted(judge_scores.values())
            n = len(scores_list)
            if n % 2 == 0:
                consensus_score = (scores_list[n // 2 - 1] + scores_list[n // 2]) / 2
            else:
                consensus_score = scores_list[n // 2]
        else:
            # Default to weighted average
            consensus_score = sum(
                score * judge_weights[judge_id] for judge_id, score in judge_scores.items()
            )

        # Calculate basic statistics
        scores_list = list(judge_scores.values())
        mean_score = sum(scores_list) / len(scores_list)
        variance = sum((score - mean_score) ** 2 for score in scores_list) / len(scores_list)
        std_dev = variance.sqrt() if variance > 0 else Decimal("0")

        # Simple confidence interval (mean ± 1.96 * std_err)
        std_err = std_dev / Decimal(len(scores_list)).sqrt()
        margin = Decimal("1.96") * std_err
        confidence_interval = (
            max(Decimal("0"), consensus_score - margin),
            min(Decimal("1"), consensus_score + margin),
        )

        # Basic agreement level (inverse of coefficient of variation)
        if mean_score > 0:
            cv = std_dev / mean_score
            agreement_level = max(Decimal("0"), 1 - cv)
        else:
            agreement_level = Decimal("1") if std_dev == 0 else Decimal("0")

        return cls(
            consensus_score=consensus_score.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            confidence_interval=confidence_interval,
            agreement_level=agreement_level.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            judge_scores=judge_scores,
            judge_weights=judge_weights,
            outlier_judges=[],  # No outlier detection in simple method
            statistical_significance=Decimal("0.05"),  # Default significance level
            consensus_method=method,
        )

    def is_high_agreement(self, threshold: Decimal = Decimal("0.8")) -> bool:
        """Check if consensus has high agreement."""
        return self.agreement_level >= threshold

    def is_statistically_significant(self, alpha: Decimal = Decimal("0.05")) -> bool:
        """Check if consensus is statistically significant."""
        return self.statistical_significance <= alpha

    def has_outliers(self) -> bool:
        """Check if consensus detected outlier judges."""
        return len(self.outlier_judges) > 0

    def get_effective_judges_count(self) -> int:
        """Get count of judges excluding outliers."""
        return len(self.judge_scores) - len(self.outlier_judges)

    def get_consensus_strength(self) -> str:
        """Get qualitative assessment of consensus strength."""
        if not self.is_statistically_significant():
            return "WEAK"

        if self.agreement_level >= Decimal("0.9"):
            return "VERY_STRONG"
        elif self.agreement_level >= Decimal("0.8"):
            return "STRONG"
        elif self.agreement_level >= Decimal("0.6"):
            return "MODERATE"
        else:
            return "WEAK"

    def get_confidence_width(self) -> Decimal:
        """Get width of confidence interval."""
        return self.confidence_interval[1] - self.confidence_interval[0]

    def is_consensus_precise(self, max_width: Decimal = Decimal("0.2")) -> bool:
        """Check if confidence interval is narrow enough."""
        return self.get_confidence_width() <= max_width

    def get_judge_deviations(self) -> Dict[str, Decimal]:
        """Get absolute deviations of each judge from consensus."""
        return {
            judge_id: abs(score - self.consensus_score)
            for judge_id, score in self.judge_scores.items()
        }

    def get_most_deviant_judge(self) -> Optional[str]:
        """Get judge with largest deviation from consensus."""
        if not self.judge_scores:
            return None

        deviations = self.get_judge_deviations()
        return max(deviations.keys(), key=lambda k: deviations[k])

    def exclude_outlier_consensus(self) -> "ConsensusResult":
        """Create new consensus excluding outlier judges."""
        if not self.has_outliers():
            return self

        # Filter out outlier judges
        filtered_scores = {
            judge_id: score
            for judge_id, score in self.judge_scores.items()
            if judge_id not in self.outlier_judges
        }

        if len(filtered_scores) < 2:
            raise InsufficientDataError("Not enough judges after outlier removal")

        return ConsensusResult.create_simple_consensus(filtered_scores, self.consensus_method)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "consensus_score": str(self.consensus_score),
            "confidence_interval": [
                str(self.confidence_interval[0]),
                str(self.confidence_interval[1]),
            ],
            "agreement_level": str(self.agreement_level),
            "judge_scores": {k: str(v) for k, v in self.judge_scores.items()},
            "judge_weights": {k: str(v) for k, v in self.judge_weights.items()},
            "outlier_judges": self.outlier_judges,
            "statistical_significance": str(self.statistical_significance),
            "consensus_method": self.consensus_method,
            "metadata": self.metadata.copy(),
            "consensus_strength": self.get_consensus_strength(),
            "is_high_agreement": self.is_high_agreement(),
            "is_statistically_significant": self.is_statistically_significant(),
            "has_outliers": self.has_outliers(),
            "effective_judges_count": self.get_effective_judges_count(),
            "confidence_width": str(self.get_confidence_width()),
            "is_precise": self.is_consensus_precise(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsensusResult":
        """Create from dictionary representation."""
        return cls(
            consensus_score=Decimal(data["consensus_score"]),
            confidence_interval=(
                Decimal(data["confidence_interval"][0]),
                Decimal(data["confidence_interval"][1]),
            ),
            agreement_level=Decimal(data["agreement_level"]),
            judge_scores={k: Decimal(v) for k, v in data["judge_scores"].items()},
            judge_weights={k: Decimal(v) for k, v in data["judge_weights"].items()},
            outlier_judges=data["outlier_judges"],
            statistical_significance=Decimal(data["statistical_significance"]),
            consensus_method=data["consensus_method"],
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ConsensusResult(score={self.consensus_score}, "
            f"agreement={self.agreement_level}, "
            f"strength={self.get_consensus_strength()})"
        )
