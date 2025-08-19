"""Statistical test result value objects."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from ..exceptions import ValidationError


class EffectMagnitude(Enum):
    """Effect size magnitude classifications."""

    NEGLIGIBLE = "negligible"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True)
class TestInterpretation:
    """Statistical test result interpretation."""

    is_significant: bool
    significance_level: float
    effect_magnitude: EffectMagnitude
    practical_significance: bool
    recommendation: str

    def __post_init__(self):
        """Validate interpretation."""
        if not (0.0 < self.significance_level < 1.0):
            raise ValidationError("Significance level must be between 0 and 1")


@dataclass(frozen=True)
class TestResult:
    """Statistical test result value object."""

    test_type: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    degrees_of_freedom: Optional[int]
    interpretation: TestInterpretation
    sample_sizes: Dict[str, int]
    test_assumptions: Dict[str, bool]
    power: Optional[float] = None
    corrected_p_value: Optional[float] = None

    def __post_init__(self):
        """Validate test result."""
        if not (0.0 <= self.p_value <= 1.0):
            raise ValidationError(f"P-value must be between 0 and 1, got {self.p_value}")

        if self.corrected_p_value is not None and not (0.0 <= self.corrected_p_value <= 1.0):
            raise ValidationError(
                f"Corrected p-value must be between 0 and 1, got {self.corrected_p_value}"
            )

        if self.power is not None and not (0.0 <= self.power <= 1.0):
            raise ValidationError(f"Statistical power must be between 0 and 1, got {self.power}")

        if len(self.confidence_interval) != 2:
            raise ValidationError("Confidence interval must have exactly 2 values")

        if self.confidence_interval[0] > self.confidence_interval[1]:
            raise ValidationError("Confidence interval lower bound must be <= upper bound")

        if not all(size >= 0 for size in self.sample_sizes.values()):
            raise ValidationError("Sample sizes must be non-negative")

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        effective_p = self.corrected_p_value if self.corrected_p_value is not None else self.p_value
        return effective_p < alpha

    def has_practical_significance(self) -> bool:
        """Check if result has practical significance."""
        return self.interpretation.practical_significance

    def get_effect_size_interpretation(self) -> str:
        """Get human-readable effect size interpretation."""
        magnitude = self.interpretation.effect_magnitude
        direction = (
            "positive" if self.effect_size > 0 else "negative" if self.effect_size < 0 else "no"
        )

        if magnitude == EffectMagnitude.NEGLIGIBLE:
            return f"Negligible {direction} effect"
        elif magnitude == EffectMagnitude.SMALL:
            return f"Small {direction} effect"
        elif magnitude == EffectMagnitude.MEDIUM:
            return f"Medium {direction} effect"
        else:
            return f"Large {direction} effect"

    def get_confidence_width(self) -> float:
        """Get width of confidence interval."""
        return self.confidence_interval[1] - self.confidence_interval[0]

    def contains_zero(self) -> bool:
        """Check if confidence interval contains zero."""
        return self.confidence_interval[0] <= 0 <= self.confidence_interval[1]

    def total_sample_size(self) -> int:
        """Get total sample size across all groups."""
        return sum(self.sample_sizes.values())

    def meets_assumptions(self) -> bool:
        """Check if statistical test assumptions are met."""
        return all(self.test_assumptions.values())

    def get_violated_assumptions(self) -> list[str]:
        """Get list of violated assumptions."""
        return [assumption for assumption, met in self.test_assumptions.items() if not met]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_type": self.test_type,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval),
            "degrees_of_freedom": self.degrees_of_freedom,
            "interpretation": {
                "is_significant": self.interpretation.is_significant,
                "significance_level": self.interpretation.significance_level,
                "effect_magnitude": self.interpretation.effect_magnitude.value,
                "practical_significance": self.interpretation.practical_significance,
                "recommendation": self.interpretation.recommendation,
            },
            "sample_sizes": self.sample_sizes.copy(),
            "test_assumptions": self.test_assumptions.copy(),
            "power": self.power,
            "corrected_p_value": self.corrected_p_value,
            "effect_size_interpretation": self.get_effect_size_interpretation(),
            "confidence_width": self.get_confidence_width(),
            "contains_zero": self.contains_zero(),
            "total_sample_size": self.total_sample_size(),
            "meets_assumptions": self.meets_assumptions(),
            "violated_assumptions": self.get_violated_assumptions(),
        }

    def __str__(self) -> str:
        """String representation."""
        sig_status = "significant" if self.is_significant() else "not significant"
        return (
            f"{self.test_type}: statistic={self.statistic:.4f}, "
            f"p={self.p_value:.4f} ({sig_status}), "
            f"effect_size={self.effect_size:.4f} ({self.interpretation.effect_magnitude.value})"
        )
