"""Insight value objects."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

from ..exceptions import ValidationError


class InsightType(Enum):
    """Types of insights."""

    PERFORMANCE = "performance"
    COST_EFFECTIVENESS = "cost_effectiveness"
    QUALITY_PATTERN = "quality_pattern"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    BIAS_DETECTION = "bias_detection"
    RECOMMENDATION = "recommendation"
    WARNING = "warning"


class InsightSeverity(Enum):
    """Insight severity levels."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Insight:
    """Automated insight value object."""

    insight_id: UUID
    insight_type: InsightType
    severity: InsightSeverity
    title: str
    description: str
    confidence_score: Decimal
    evidence: Dict[str, Any]
    recommendations: list[str]
    affected_models: Optional[list[str]] = None
    category: Optional[str] = None

    def __post_init__(self):
        """Validate insight."""
        if not self.title.strip():
            raise ValidationError("Insight title cannot be empty")

        if not self.description.strip():
            raise ValidationError("Insight description cannot be empty")

        if not (Decimal("0") <= self.confidence_score <= Decimal("1")):
            raise ValidationError(
                f"Confidence score must be between 0 and 1, got {self.confidence_score}"
            )

        if not self.recommendations:
            raise ValidationError("Insight must have at least one recommendation")

    def is_high_confidence(self, threshold: Decimal = Decimal("0.8")) -> bool:
        """Check if insight has high confidence."""
        return self.confidence_score >= threshold

    def is_actionable(self) -> bool:
        """Check if insight is actionable (has specific recommendations)."""
        return len(self.recommendations) > 0 and any(
            len(rec.strip()) > 10 for rec in self.recommendations
        )

    def requires_attention(self) -> bool:
        """Check if insight requires immediate attention."""
        return self.severity in [InsightSeverity.HIGH, InsightSeverity.CRITICAL] or (
            self.severity == InsightSeverity.MEDIUM and self.is_high_confidence()
        )

    def get_priority_score(self) -> int:
        """Get numeric priority score for sorting."""
        severity_scores = {
            InsightSeverity.INFO: 1,
            InsightSeverity.LOW: 2,
            InsightSeverity.MEDIUM: 3,
            InsightSeverity.HIGH: 4,
            InsightSeverity.CRITICAL: 5,
        }

        base_score = severity_scores[self.severity] * 10
        confidence_bonus = int(self.confidence_score * 5)  # 0-5 points

        return base_score + confidence_bonus

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "insight_id": str(self.insight_id),
            "insight_type": self.insight_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "confidence_score": str(self.confidence_score),
            "evidence": self.evidence,
            "recommendations": self.recommendations.copy(),
            "affected_models": self.affected_models.copy() if self.affected_models else None,
            "category": self.category,
            "is_high_confidence": self.is_high_confidence(),
            "is_actionable": self.is_actionable(),
            "requires_attention": self.requires_attention(),
            "priority_score": self.get_priority_score(),
        }

    def __str__(self) -> str:
        """String representation."""
        confidence_pct = int(self.confidence_score * 100)
        return f"{self.severity.value.upper()}: {self.title} (confidence: {confidence_pct}%)"
