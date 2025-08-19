"""Judge aggregate root entity."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..exceptions import (
    JudgeNotCalibratedError,
    QualityControlError,
    TemplateRenderError,
    ValidationError,
)
from ..value_objects.calibration_data import CalibrationData
from .evaluation_result import EvaluationResult
from .evaluation_template import EvaluationTemplate


@dataclass
class Judge:
    """Judge aggregate root for LLM-as-Judge evaluation."""

    judge_id: str
    name: str
    description: str
    model_provider_id: str  # Reference to model provider
    templates: List[EvaluationTemplate]
    calibration_data: Optional[CalibrationData] = None
    is_active: bool = True
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _domain_events: List[object] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate judge after creation."""
        self._validate_judge()

        # Initialize with uncalibrated data if none provided
        if self.calibration_data is None:
            self.calibration_data = CalibrationData.create_uncalibrated()

    @classmethod
    def create(
        cls,
        judge_id: str,
        name: str,
        description: str,
        model_provider_id: str,
        templates: Optional[List[EvaluationTemplate]] = None,
    ) -> "Judge":
        """Factory method to create judge."""
        judge = cls(
            judge_id=judge_id,
            name=name,
            description=description,
            model_provider_id=model_provider_id,
            templates=templates or [],
        )

        return judge

    def _validate_judge(self) -> None:
        """Validate judge properties."""
        if not self.judge_id.strip():
            raise ValidationError("Judge ID cannot be empty")

        if not self.name.strip():
            raise ValidationError("Judge name cannot be empty")

        if not self.description.strip():
            raise ValidationError("Judge description cannot be empty")

        if not self.model_provider_id.strip():
            raise ValidationError("Model provider ID cannot be empty")

    def is_calibrated(self) -> bool:
        """Check if judge is properly calibrated for production use."""
        if not self.calibration_data:
            return False

        return self.calibration_data.is_production_ready()

    def is_production_ready(self) -> bool:
        """Check if judge is ready for production evaluation."""
        return self.is_active and self.is_calibrated() and len(self.templates) > 0

    async def evaluate(
        self, prompt: str, response: str, template: EvaluationTemplate, **context
    ) -> EvaluationResult:
        """Perform evaluation using this judge."""
        if not self.is_active:
            raise ValidationError(f"Judge {self.name} is not active")

        if not self.is_calibrated():
            raise JudgeNotCalibratedError(
                f"Judge {self.name} requires calibration before production use. "
                f"Current calibration grade: {self.calibration_data.get_quality_grade()}"
            )

        if template not in self.templates:
            # Add template if not already present
            self.add_template(template)

        # Create pending evaluation result
        result = EvaluationResult.create_pending(
            judge_id=self.judge_id,
            template_id=template.template_id,
            prompt=prompt,
            response=response,
        )

        try:
            # Render evaluation prompt from template
            evaluation_prompt = template.render(
                prompt=prompt, response=response, judge_name=self.name, **context
            )

            start_time = datetime.utcnow()

            # Call underlying model provider (this would be implemented in application layer)
            # For now, we simulate the evaluation process
            model_response = await self._call_model_provider(
                template.judge_model_id, evaluation_prompt, **template.model_parameters
            )

            end_time = datetime.utcnow()
            evaluation_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Parse and validate evaluation response
            dimension_scores, confidence_score, reasoning = self._parse_evaluation_response(
                model_response, template
            )

            # Complete the evaluation result
            result.complete_evaluation(
                template=template,
                dimension_scores=dimension_scores,
                confidence_score=confidence_score,
                reasoning=reasoning,
                evaluation_time_ms=evaluation_time_ms,
            )

            # Record performance for calibration tracking
            self._record_evaluation_performance(result, template)

            return result

        except Exception as e:
            # Handle evaluation failure
            error_message = f"Evaluation failed: {str(e)}"
            result.fail_evaluation(error_message)

            # Record failure for performance tracking
            self._record_evaluation_failure(str(e), template)

            return result

    async def _call_model_provider(self, model_id: str, prompt: str, **parameters) -> str:
        """Call underlying model provider (to be implemented in application layer)."""
        # This is a placeholder - actual implementation would use dependency injection
        # to call the model provider service
        raise NotImplementedError(
            "Model provider integration must be implemented in application layer"
        )

    def _parse_evaluation_response(
        self, model_response: str, template: EvaluationTemplate
    ) -> tuple[Dict[str, int], Decimal, str]:
        """Parse model response into structured evaluation data."""
        # This is a simplified parser - in production, this would be more sophisticated
        # and handle various response formats, potentially using structured output or JSON

        lines = model_response.strip().split("\n")
        dimension_scores = {}
        confidence_score = Decimal("0.5")  # Default confidence
        reasoning = model_response  # Full response as reasoning for now

        # Extract dimension scores (simplified parsing)
        for line in lines:
            for dimension in template.dimensions:
                if dimension.name.lower() in line.lower():
                    # Try to extract score from line
                    import re

                    score_match = re.search(r"\b([1-5])\b", line)
                    if score_match:
                        score = int(score_match.group(1))
                        if dimension.is_valid_score(score):
                            dimension_scores[dimension.name] = score

        # Extract confidence if mentioned
        confidence_patterns = [
            r"confidence:?\s*([0-9.]+)",
            r"confident:?\s*([0-9.]+)",
            r"certainty:?\s*([0-9.]+)",
        ]

        for pattern in confidence_patterns:
            import re

            match = re.search(pattern, model_response.lower())
            if match:
                try:
                    conf_value = float(match.group(1))
                    if conf_value > 1:
                        conf_value = conf_value / 100  # Convert percentage
                    if 0 <= conf_value <= 1:
                        confidence_score = Decimal(str(conf_value))
                        break
                except (ValueError, TypeError):
                    continue

        # Validate all required dimensions have scores
        for dimension in template.get_required_dimensions():
            if dimension.name not in dimension_scores:
                # Assign default middle score if missing
                min_score, max_score = dimension.get_score_range()
                default_score = (min_score + max_score) // 2
                dimension_scores[dimension.name] = default_score

        return dimension_scores, confidence_score, reasoning

    def _record_evaluation_performance(
        self, result: EvaluationResult, template: EvaluationTemplate
    ) -> None:
        """Record evaluation performance for calibration tracking."""
        performance_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "result_id": str(result.result_id),
            "template_id": str(template.template_id),
            "overall_score": str(result.overall_score),
            "confidence_score": str(result.confidence_score),
            "evaluation_time_ms": result.evaluation_time_ms,
            "dimension_scores": result.dimension_scores.copy(),
            "success": True,
        }

        self.performance_history.append(performance_record)

        # Keep only recent history to prevent unbounded growth
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def _record_evaluation_failure(self, error_message: str, template: EvaluationTemplate) -> None:
        """Record evaluation failure for performance tracking."""
        failure_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "template_id": str(template.template_id),
            "error_message": error_message,
            "success": False,
        }

        self.performance_history.append(failure_record)

        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def add_template(self, template: EvaluationTemplate) -> None:
        """Add evaluation template to judge."""
        if any(t.template_id == template.template_id for t in self.templates):
            return  # Template already exists

        self.templates.append(template)

    def remove_template(self, template_id: UUID) -> None:
        """Remove evaluation template from judge."""
        self.templates = [t for t in self.templates if t.template_id != template_id]

    def get_template(self, template_id: UUID) -> Optional[EvaluationTemplate]:
        """Get template by ID."""
        for template in self.templates:
            if template.template_id == template_id:
                return template
        return None

    def has_template(self, template_id: UUID) -> bool:
        """Check if judge has specific template."""
        return self.get_template(template_id) is not None

    def calibrate(self, calibration_data: CalibrationData) -> None:
        """Update judge calibration data."""
        self.calibration_data = calibration_data

        # Add domain event
        from ..events.evaluation_events import JudgeCalibrated

        # Create a valid UUID for aggregate_id by hashing judge_id if needed
        try:
            aggregate_id = (
                UUID(self.judge_id)
                if isinstance(self.judge_id, str) and len(self.judge_id) == 36
                else uuid4()
            )
        except ValueError:
            aggregate_id = uuid4()

        event = JudgeCalibrated(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=aggregate_id,
            judge_id=self.judge_id,
            accuracy=str(calibration_data.accuracy),
            consistency=str(calibration_data.consistency),
            bias_score=str(calibration_data.bias_score),
            sample_size=calibration_data.sample_size,
            is_production_ready=calibration_data.is_production_ready(),
        )
        self._domain_events.append(event)

    def deactivate(self, reason: Optional[str] = None) -> None:
        """Deactivate judge."""
        self.is_active = False

        if reason:
            self.metadata["deactivation_reason"] = reason
            self.metadata["deactivated_at"] = datetime.utcnow().isoformat()

    def reactivate(self) -> None:
        """Reactivate judge if calibrated."""
        if not self.is_calibrated():
            raise JudgeNotCalibratedError(
                f"Cannot reactivate judge {self.name} without proper calibration"
            )

        self.is_active = True

        # Remove deactivation metadata
        self.metadata.pop("deactivation_reason", None)
        self.metadata.pop("deactivated_at", None)
        self.metadata["reactivated_at"] = datetime.utcnow().isoformat()

    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get performance statistics for recent period."""
        cutoff_date = datetime.utcnow() - datetime.timedelta(days=days)

        recent_records = [
            record
            for record in self.performance_history
            if datetime.fromisoformat(record["timestamp"]) >= cutoff_date
        ]

        if not recent_records:
            return {
                "period_days": days,
                "total_evaluations": 0,
                "success_rate": 0.0,
                "average_confidence": 0.0,
                "average_evaluation_time_ms": 0,
                "unique_templates": 0,
            }

        successful_records = [r for r in recent_records if r.get("success", False)]

        # Calculate statistics
        total_evaluations = len(recent_records)
        success_rate = len(successful_records) / total_evaluations if total_evaluations > 0 else 0

        if successful_records:
            avg_confidence = sum(float(r["confidence_score"]) for r in successful_records) / len(
                successful_records
            )

            avg_time = sum(r["evaluation_time_ms"] for r in successful_records) / len(
                successful_records
            )

            unique_templates = len(set(r["template_id"] for r in successful_records))
        else:
            avg_confidence = 0.0
            avg_time = 0
            unique_templates = 0

        return {
            "period_days": days,
            "total_evaluations": total_evaluations,
            "successful_evaluations": len(successful_records),
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "average_evaluation_time_ms": avg_time,
            "unique_templates": unique_templates,
        }

    def needs_recalibration(self) -> bool:
        """Check if judge needs recalibration."""
        if not self.calibration_data:
            return True

        return (
            self.calibration_data.needs_recalibration()
            or len(self.calibration_data.get_drift_indicators()) > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "judge_id": self.judge_id,
            "name": self.name,
            "description": self.description,
            "model_provider_id": self.model_provider_id,
            "templates": [t.to_dict() for t in self.templates],
            "is_active": self.is_active,
            "metadata": self.metadata.copy(),
            "is_calibrated": self.is_calibrated(),
            "is_production_ready": self.is_production_ready(),
            "template_count": len(self.templates),
            "performance_history_size": len(self.performance_history),
        }

        if self.calibration_data:
            result["calibration_data"] = self.calibration_data.to_dict()

        # Add recent performance stats
        result["recent_performance"] = self.get_performance_stats()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Judge":
        """Create from dictionary representation."""
        templates = [EvaluationTemplate.from_dict(t) for t in data.get("templates", [])]

        judge = cls(
            judge_id=data["judge_id"],
            name=data["name"],
            description=data["description"],
            model_provider_id=data["model_provider_id"],
            templates=templates,
            is_active=data.get("is_active", True),
            performance_history=data.get("performance_history", []),
            metadata=data.get("metadata", {}),
        )

        # Restore calibration data if present
        if "calibration_data" in data and data["calibration_data"]:
            judge.calibration_data = CalibrationData.from_dict(data["calibration_data"])

        return judge

    def __str__(self) -> str:
        """String representation."""
        status = "active" if self.is_active else "inactive"
        calibration = "calibrated" if self.is_calibrated() else "uncalibrated"

        return (
            f"Judge(id='{self.judge_id}', "
            f"name='{self.name}', "
            f"status={status}, "
            f"calibration={calibration}, "
            f"templates={len(self.templates)})"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison based on ID."""
        if not isinstance(other, Judge):
            return False
        return self.judge_id == other.judge_id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.judge_id)
