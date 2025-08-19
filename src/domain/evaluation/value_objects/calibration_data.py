"""Calibration data value object."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..exceptions import ValidationError


@dataclass(frozen=True)
class CalibrationData:
    """Value object containing judge calibration information."""

    accuracy: Decimal
    consistency: Decimal  # Inter-rater reliability coefficient
    bias_score: Decimal  # Systematic bias measurement
    confidence_calibration: Decimal  # How well confidence matches actual accuracy
    sample_size: int
    calibrated_at: datetime
    golden_standard_scores: Dict[str, Decimal] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate calibration data."""
        if not (0 <= self.accuracy <= 1):
            raise ValidationError("Accuracy must be between 0 and 1")

        if not (0 <= self.consistency <= 1):
            raise ValidationError("Consistency must be between 0 and 1")

        if not (-1 <= self.bias_score <= 1):
            raise ValidationError("Bias score must be between -1 and 1")

        if not (0 <= self.confidence_calibration <= 1):
            raise ValidationError("Confidence calibration must be between 0 and 1")

        if self.sample_size < 10:
            raise ValidationError("Calibration requires minimum 10 samples")

    @classmethod
    def create_uncalibrated(cls) -> "CalibrationData":
        """Create uncalibrated data placeholder."""
        return cls(
            accuracy=Decimal("0.0"),
            consistency=Decimal("0.0"),
            bias_score=Decimal("0.0"),
            confidence_calibration=Decimal("0.0"),
            sample_size=10,  # Minimum required to pass validation
            calibrated_at=datetime.utcnow(),
            golden_standard_scores={},
            performance_metrics={"status": "uncalibrated"},
        )

    def is_production_ready(self) -> bool:
        """Check if calibration meets production standards."""
        return (
            self.accuracy >= Decimal("0.8")
            and self.consistency >= Decimal("0.75")
            and abs(self.bias_score) <= Decimal("0.2")
            and self.confidence_calibration >= Decimal("0.7")
            and self.sample_size >= 50
        )

    def has_significant_bias(self) -> bool:
        """Check if judge has significant systematic bias."""
        return abs(self.bias_score) > Decimal("0.3")

    def get_quality_grade(self) -> str:
        """Get quality grade based on calibration metrics."""
        if not self.is_production_ready():
            return "NEEDS_CALIBRATION"

        if (
            self.accuracy >= Decimal("0.95")
            and self.consistency >= Decimal("0.9")
            and abs(self.bias_score) <= Decimal("0.1")
        ):
            return "EXCELLENT"

        if (
            self.accuracy >= Decimal("0.9")
            and self.consistency >= Decimal("0.85")
            and abs(self.bias_score) <= Decimal("0.15")
        ):
            return "GOOD"

        return "ACCEPTABLE"

    def get_reliability_score(self) -> Decimal:
        """Calculate composite reliability score."""
        # Weighted combination of metrics
        reliability = (
            self.accuracy * Decimal("0.4")
            + self.consistency * Decimal("0.3")
            + (1 - abs(self.bias_score)) * Decimal("0.2")
            + self.confidence_calibration * Decimal("0.1")
        )
        return reliability.quantize(Decimal("0.001"))

    def needs_recalibration(self, days_threshold: int = 30) -> bool:
        """Check if recalibration is needed based on age."""
        age_days = (datetime.utcnow() - self.calibrated_at).days
        return age_days > days_threshold

    def get_drift_indicators(self) -> List[str]:
        """Get list of potential drift indicators."""
        indicators = []

        if self.accuracy < Decimal("0.8"):
            indicators.append("low_accuracy")

        if self.consistency < Decimal("0.75"):
            indicators.append("low_consistency")

        if abs(self.bias_score) > Decimal("0.2"):
            indicators.append("significant_bias")

        if self.confidence_calibration < Decimal("0.7"):
            indicators.append("poor_confidence_calibration")

        if self.needs_recalibration():
            indicators.append("stale_calibration")

        return indicators

    def update_performance_metric(self, key: str, value: Any) -> "CalibrationData":
        """Create new instance with updated performance metric."""
        new_metrics = self.performance_metrics.copy()
        new_metrics[key] = value

        return CalibrationData(
            accuracy=self.accuracy,
            consistency=self.consistency,
            bias_score=self.bias_score,
            confidence_calibration=self.confidence_calibration,
            sample_size=self.sample_size,
            calibrated_at=self.calibrated_at,
            golden_standard_scores=self.golden_standard_scores.copy(),
            performance_metrics=new_metrics,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "accuracy": str(self.accuracy),
            "consistency": str(self.consistency),
            "bias_score": str(self.bias_score),
            "confidence_calibration": str(self.confidence_calibration),
            "sample_size": self.sample_size,
            "calibrated_at": self.calibrated_at.isoformat(),
            "golden_standard_scores": {k: str(v) for k, v in self.golden_standard_scores.items()},
            "performance_metrics": self.performance_metrics.copy(),
            "is_production_ready": self.is_production_ready(),
            "quality_grade": self.get_quality_grade(),
            "reliability_score": str(self.get_reliability_score()),
            "drift_indicators": self.get_drift_indicators(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationData":
        """Create from dictionary representation."""
        return cls(
            accuracy=Decimal(data["accuracy"]),
            consistency=Decimal(data["consistency"]),
            bias_score=Decimal(data["bias_score"]),
            confidence_calibration=Decimal(data["confidence_calibration"]),
            sample_size=data["sample_size"],
            calibrated_at=datetime.fromisoformat(data["calibrated_at"]),
            golden_standard_scores={
                k: Decimal(v) for k, v in data.get("golden_standard_scores", {}).items()
            },
            performance_metrics=data.get("performance_metrics", {}),
        )

    def __str__(self) -> str:
        """String representation."""
        return (
            f"CalibrationData(accuracy={self.accuracy}, "
            f"consistency={self.consistency}, grade={self.get_quality_grade()})"
        )
