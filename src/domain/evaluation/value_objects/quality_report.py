"""Quality report value object."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from ..exceptions import ValidationError


class QualityLevel(Enum):
    """Quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class QualityIssueType(Enum):
    """Types of quality issues."""

    REASONING_INCOMPLETE = "reasoning_incomplete"
    SCORE_INCONSISTENT = "score_inconsistent"
    LOW_CONFIDENCE = "low_confidence"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    BIAS_DETECTED = "bias_detected"
    OUTLIER_SCORE = "outlier_score"
    TEMPLATE_ERROR = "template_error"
    STATISTICAL_ANOMALY = "statistical_anomaly"


@dataclass(frozen=True)
class QualityIssue:
    """Individual quality issue."""

    issue_type: QualityIssueType
    severity: str  # "high", "medium", "low"
    description: str
    affected_component: Optional[str] = None
    suggested_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate quality issue."""
        if self.severity not in ["high", "medium", "low"]:
            raise ValidationError(f"Invalid severity level: {self.severity}")


@dataclass(frozen=True)
class QualityReport:
    """Value object representing evaluation quality assessment."""

    overall_quality: QualityLevel
    quality_score: Decimal  # 0-1 scale
    issues: List[QualityIssue]
    metrics: Dict[str, Decimal]  # Quality metrics
    recommendations: List[str]
    assessed_at: datetime
    evaluator_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate quality report."""
        if not (0 <= self.quality_score <= 1):
            raise ValidationError("Quality score must be between 0 and 1")

        if not self.evaluator_id:
            raise ValidationError("Evaluator ID is required")

    @classmethod
    def create_passed(
        cls, quality_score: Decimal, evaluator_id: str, metrics: Optional[Dict[str, Decimal]] = None
    ) -> "QualityReport":
        """Create passing quality report."""
        if quality_score < Decimal("0.7"):
            raise ValidationError("Passing quality score must be >= 0.7")

        # Determine quality level based on score
        if quality_score >= Decimal("0.95"):
            quality_level = QualityLevel.EXCELLENT
        elif quality_score >= Decimal("0.85"):
            quality_level = QualityLevel.GOOD
        else:
            quality_level = QualityLevel.ACCEPTABLE

        return cls(
            overall_quality=quality_level,
            quality_score=quality_score,
            issues=[],
            metrics=metrics or {},
            recommendations=[],
            assessed_at=datetime.utcnow(),
            evaluator_id=evaluator_id,
        )

    @classmethod
    def create_failed(
        cls,
        quality_score: Decimal,
        issues: List[QualityIssue],
        evaluator_id: str,
        metrics: Optional[Dict[str, Decimal]] = None,
    ) -> "QualityReport":
        """Create failing quality report."""
        if quality_score >= Decimal("0.7"):
            raise ValidationError("Failing quality score must be < 0.7")

        # Determine quality level based on score and issues
        if quality_score < Decimal("0.3") or any(issue.severity == "high" for issue in issues):
            quality_level = QualityLevel.UNACCEPTABLE
        else:
            quality_level = QualityLevel.POOR

        # Generate recommendations based on issues
        recommendations = cls._generate_recommendations(issues)

        return cls(
            overall_quality=quality_level,
            quality_score=quality_score,
            issues=issues,
            metrics=metrics or {},
            recommendations=recommendations,
            assessed_at=datetime.utcnow(),
            evaluator_id=evaluator_id,
        )

    @staticmethod
    def _generate_recommendations(issues: List[QualityIssue]) -> List[str]:
        """Generate recommendations based on quality issues."""
        recommendations = []

        issue_counts = {}
        for issue in issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1

        if QualityIssueType.REASONING_INCOMPLETE in issue_counts:
            recommendations.append("Improve reasoning completeness and detail")

        if QualityIssueType.SCORE_INCONSISTENT in issue_counts:
            recommendations.append("Review score consistency with reasoning")

        if QualityIssueType.LOW_CONFIDENCE in issue_counts:
            recommendations.append("Consider judge recalibration")

        if QualityIssueType.BIAS_DETECTED in issue_counts:
            recommendations.append("Investigate potential systematic bias")

        if QualityIssueType.OUTLIER_SCORE in issue_counts:
            recommendations.append("Review outlier scores with additional judges")

        return recommendations

    def is_passing(self) -> bool:
        """Check if quality assessment is passing."""
        return self.overall_quality in [
            QualityLevel.EXCELLENT,
            QualityLevel.GOOD,
            QualityLevel.ACCEPTABLE,
        ]

    def has_critical_issues(self) -> bool:
        """Check if report has critical (high severity) issues."""
        return any(issue.severity == "high" for issue in self.issues)

    def get_issues_by_severity(self, severity: str) -> List[QualityIssue]:
        """Get issues filtered by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_issues_by_type(self, issue_type: QualityIssueType) -> List[QualityIssue]:
        """Get issues filtered by type."""
        return [issue for issue in self.issues if issue.issue_type == issue_type]

    def get_issue_summary(self) -> Dict[str, int]:
        """Get summary of issues by type."""
        summary = {}
        for issue in self.issues:
            issue_key = issue.issue_type.value
            summary[issue_key] = summary.get(issue_key, 0) + 1
        return summary

    def get_severity_summary(self) -> Dict[str, int]:
        """Get summary of issues by severity."""
        summary = {"high": 0, "medium": 0, "low": 0}
        for issue in self.issues:
            summary[issue.severity] += 1
        return summary

    def add_metric(self, key: str, value: Decimal) -> "QualityReport":
        """Create new report with additional metric."""
        new_metrics = self.metrics.copy()
        new_metrics[key] = value

        return QualityReport(
            overall_quality=self.overall_quality,
            quality_score=self.quality_score,
            issues=self.issues,
            metrics=new_metrics,
            recommendations=self.recommendations,
            assessed_at=self.assessed_at,
            evaluator_id=self.evaluator_id,
            metadata=self.metadata,
        )

    def get_metric(self, key: str, default: Optional[Decimal] = None) -> Optional[Decimal]:
        """Get quality metric by key."""
        return self.metrics.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_quality": self.overall_quality.value,
            "quality_score": str(self.quality_score),
            "issues": [
                {
                    "type": issue.issue_type.value,
                    "severity": issue.severity,
                    "description": issue.description,
                    "affected_component": issue.affected_component,
                    "suggested_action": issue.suggested_action,
                    "metadata": issue.metadata,
                }
                for issue in self.issues
            ],
            "metrics": {k: str(v) for k, v in self.metrics.items()},
            "recommendations": self.recommendations,
            "assessed_at": self.assessed_at.isoformat(),
            "evaluator_id": self.evaluator_id,
            "metadata": self.metadata.copy(),
            "is_passing": self.is_passing(),
            "has_critical_issues": self.has_critical_issues(),
            "issue_summary": self.get_issue_summary(),
            "severity_summary": self.get_severity_summary(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityReport":
        """Create from dictionary representation."""
        issues = []
        for issue_data in data["issues"]:
            issue = QualityIssue(
                issue_type=QualityIssueType(issue_data["type"]),
                severity=issue_data["severity"],
                description=issue_data["description"],
                affected_component=issue_data.get("affected_component"),
                suggested_action=issue_data.get("suggested_action"),
                metadata=issue_data.get("metadata", {}),
            )
            issues.append(issue)

        return cls(
            overall_quality=QualityLevel(data["overall_quality"]),
            quality_score=Decimal(data["quality_score"]),
            issues=issues,
            metrics={k: Decimal(v) for k, v in data["metrics"].items()},
            recommendations=data["recommendations"],
            assessed_at=datetime.fromisoformat(data["assessed_at"]),
            evaluator_id=data["evaluator_id"],
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """String representation."""
        return (
            f"QualityReport(quality={self.overall_quality.value}, "
            f"score={self.quality_score}, "
            f"issues={len(self.issues)})"
        )
