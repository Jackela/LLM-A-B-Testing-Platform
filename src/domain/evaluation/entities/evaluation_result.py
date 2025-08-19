"""Evaluation result entity."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..exceptions import InvalidScore, ValidationError
from ..value_objects.quality_report import QualityReport
from .evaluation_template import EvaluationTemplate


@dataclass
class EvaluationResult:
    """Entity representing result from individual judge evaluation."""

    result_id: UUID
    judge_id: str
    template_id: UUID
    prompt: str
    response: str
    dimension_scores: Dict[str, int]  # dimension_name -> raw_score
    overall_score: Decimal
    confidence_score: Decimal  # Judge's confidence in evaluation (0-1)
    reasoning: str  # Judge's reasoning/explanation
    evaluation_time_ms: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    quality_report: Optional[QualityReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _domain_events: List[object] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate evaluation result after creation."""
        if not self.result_id:
            self.result_id = uuid4()

        self._validate_result()

    @classmethod
    def create_pending(
        cls, judge_id: str, template_id: UUID, prompt: str, response: str
    ) -> "EvaluationResult":
        """Factory method to create pending evaluation result."""
        if not judge_id.strip():
            raise ValidationError("Judge ID cannot be empty")

        if not prompt.strip():
            raise ValidationError("Prompt cannot be empty")

        if not response.strip():
            raise ValidationError("Response cannot be empty")

        return cls(
            result_id=uuid4(),
            judge_id=judge_id,
            template_id=template_id,
            prompt=prompt,
            response=response,
            dimension_scores={},
            overall_score=Decimal("0"),
            confidence_score=Decimal("0"),
            reasoning="",
            evaluation_time_ms=0,
            created_at=datetime.utcnow(),
        )

    def _validate_result(self) -> None:
        """Validate evaluation result properties."""
        if not self.judge_id.strip():
            raise ValidationError("Judge ID cannot be empty")

        if not (0 <= self.overall_score <= 1):
            raise ValidationError("Overall score must be between 0 and 1")

        if not (0 <= self.confidence_score <= 1):
            raise ValidationError("Confidence score must be between 0 and 1")

        if self.evaluation_time_ms < 0:
            raise ValidationError("Evaluation time cannot be negative")

        # Validate dimension scores
        for dimension_name, score in self.dimension_scores.items():
            if not isinstance(score, int):
                raise ValidationError(f"Score for dimension '{dimension_name}' must be an integer")

            if not (1 <= score <= 5):  # Assuming standard 1-5 scoring
                raise ValidationError(
                    f"Score for dimension '{dimension_name}' must be between 1 and 5"
                )

    def complete_evaluation(
        self,
        template: EvaluationTemplate,
        dimension_scores: Dict[str, int],
        confidence_score: Decimal,
        reasoning: str,
        evaluation_time_ms: int,
    ) -> None:
        """Complete the evaluation with results."""
        if self.is_completed():
            raise ValidationError("Evaluation is already completed")

        # Validate dimension scores against template
        for dimension in template.dimensions:
            if dimension.is_required and dimension.name not in dimension_scores:
                raise ValidationError(f"Missing score for required dimension: {dimension.name}")

            if dimension.name in dimension_scores:
                score = dimension_scores[dimension.name]
                if not dimension.is_valid_score(score):
                    raise InvalidScore(f"Invalid score {score} for dimension {dimension.name}")

        # Calculate overall score using template
        overall_score = template.calculate_weighted_score(dimension_scores)

        # Update result
        self.dimension_scores = dimension_scores.copy()
        self.overall_score = overall_score
        self.confidence_score = confidence_score.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        self.reasoning = reasoning.strip()
        self.evaluation_time_ms = evaluation_time_ms
        self.completed_at = datetime.utcnow()

        # Add domain event
        from ..events.evaluation_events import EvaluationCompleted

        event = EvaluationCompleted(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=self.result_id,
            judge_id=self.judge_id,
            template_id=template.template_id,
            overall_score=str(self.overall_score),
            dimension_scores=dimension_scores.copy(),
            confidence_score=str(self.confidence_score),
            evaluation_time_ms=evaluation_time_ms,
        )
        self._domain_events.append(event)

    def fail_evaluation(self, error_message: str, retry_count: int = 0) -> None:
        """Mark evaluation as failed."""
        if self.is_completed():
            raise ValidationError("Evaluation is already completed")

        self.completed_at = datetime.utcnow()
        self.metadata["error_message"] = error_message
        self.metadata["failed"] = True

        # Add domain event
        from ..events.evaluation_events import EvaluationFailed

        event = EvaluationFailed(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=self.result_id,
            judge_id=self.judge_id,
            template_id=self.template_id,
            error_type="evaluation_failed",
            error_message=error_message,
            retry_count=retry_count,
        )
        self._domain_events.append(event)

    def is_completed(self) -> bool:
        """Check if evaluation is completed."""
        return self.completed_at is not None

    def is_successful(self) -> bool:
        """Check if evaluation completed successfully."""
        return self.is_completed() and not self.has_error()

    def has_error(self) -> bool:
        """Check if evaluation has error."""
        return self.metadata.get("failed", False)

    def is_high_confidence(self, threshold: Decimal = Decimal("0.8")) -> bool:
        """Check if evaluation has high confidence."""
        return self.confidence_score >= threshold

    def is_low_confidence(self, threshold: Decimal = Decimal("0.5")) -> bool:
        """Check if evaluation has low confidence."""
        return self.confidence_score <= threshold

    def get_dimension_score(self, dimension_name: str) -> Optional[int]:
        """Get score for specific dimension."""
        return self.dimension_scores.get(dimension_name)

    def get_normalized_dimension_scores(self, template: EvaluationTemplate) -> Dict[str, Decimal]:
        """Get normalized dimension scores (0-1 range)."""
        normalized = {}

        for dimension_name, raw_score in self.dimension_scores.items():
            dimension = template.get_dimension(dimension_name)
            if dimension:
                normalized[dimension_name] = dimension.normalize_score(raw_score)

        return normalized

    def get_weighted_dimension_scores(self, template: EvaluationTemplate) -> Dict[str, Decimal]:
        """Get weighted dimension scores."""
        weighted = {}

        for dimension_name, raw_score in self.dimension_scores.items():
            dimension = template.get_dimension(dimension_name)
            if dimension:
                weighted[dimension_name] = dimension.calculate_weighted_score(raw_score)

        return weighted

    def add_quality_report(self, quality_report: QualityReport) -> None:
        """Add quality assessment report."""
        self.quality_report = quality_report

        # Add domain event if quality issues detected
        if quality_report.has_critical_issues():
            from ..events.evaluation_events import QualityIssueDetected

            for issue in quality_report.get_issues_by_severity("high"):
                event = QualityIssueDetected(
                    occurred_at=datetime.utcnow(),
                    event_id=uuid4(),
                    aggregate_id=self.result_id,
                    issue_type=issue.issue_type.value,
                    severity=issue.severity,
                    description=issue.description,
                    affected_component=issue.affected_component,
                    evaluation_id=self.result_id,
                    judge_id=self.judge_id,
                )
                self._domain_events.append(event)

    def has_quality_issues(self) -> bool:
        """Check if evaluation has quality issues."""
        return self.quality_report is not None and not self.quality_report.is_passing()

    def has_critical_quality_issues(self) -> bool:
        """Check if evaluation has critical quality issues."""
        return self.quality_report is not None and self.quality_report.has_critical_issues()

    def get_quality_score(self) -> Optional[Decimal]:
        """Get quality assessment score."""
        return self.quality_report.quality_score if self.quality_report else None

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for this evaluation."""
        if not self.is_completed():
            return {"status": "incomplete"}

        metrics = {
            "evaluation_time_ms": self.evaluation_time_ms,
            "confidence_score": str(self.confidence_score),
            "overall_score": str(self.overall_score),
            "dimension_count": len(self.dimension_scores),
            "reasoning_length": len(self.reasoning),
            "is_high_confidence": self.is_high_confidence(),
            "is_low_confidence": self.is_low_confidence(),
            "has_quality_issues": self.has_quality_issues(),
            "has_critical_quality_issues": self.has_critical_quality_issues(),
        }

        if self.quality_report:
            metrics["quality_score"] = str(self.quality_report.quality_score)
            metrics["quality_level"] = self.quality_report.overall_quality.value

        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "result_id": str(self.result_id),
            "judge_id": self.judge_id,
            "template_id": str(self.template_id),
            "prompt": self.prompt,
            "response": self.response,
            "dimension_scores": self.dimension_scores.copy(),
            "overall_score": str(self.overall_score),
            "confidence_score": str(self.confidence_score),
            "reasoning": self.reasoning,
            "evaluation_time_ms": self.evaluation_time_ms,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata.copy(),
            "is_completed": self.is_completed(),
            "is_successful": self.is_successful(),
            "has_error": self.has_error(),
        }

        if self.quality_report:
            result["quality_report"] = self.quality_report.to_dict()

        # Add performance metrics
        result["performance_metrics"] = self.calculate_performance_metrics()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary representation."""
        result = cls(
            result_id=UUID(data["result_id"]),
            judge_id=data["judge_id"],
            template_id=UUID(data["template_id"]),
            prompt=data["prompt"],
            response=data["response"],
            dimension_scores=data["dimension_scores"],
            overall_score=Decimal(data["overall_score"]),
            confidence_score=Decimal(data["confidence_score"]),
            reasoning=data["reasoning"],
            evaluation_time_ms=data["evaluation_time_ms"],
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            metadata=data.get("metadata", {}),
        )

        # Restore quality report if present
        if "quality_report" in data and data["quality_report"]:
            from ..value_objects.quality_report import QualityReport

            result.quality_report = QualityReport.from_dict(data["quality_report"])

        return result

    def __str__(self) -> str:
        """String representation."""
        status = "completed" if self.is_completed() else "pending"
        if self.has_error():
            status = "failed"

        return (
            f"EvaluationResult(id={str(self.result_id)[:8]}, "
            f"judge={self.judge_id}, "
            f"score={self.overall_score}, "
            f"status={status})"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison based on ID."""
        if not isinstance(other, EvaluationResult):
            return False
        return self.result_id == other.result_id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.result_id)
