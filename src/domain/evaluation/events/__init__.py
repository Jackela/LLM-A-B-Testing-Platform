"""Events for evaluation domain."""

from .evaluation_events import (
    ConsensusCalculated,
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationTemplateCreated,
    EvaluationTemplateDeactivated,
    EvaluationTemplateModified,
    JudgeCalibrated,
    QualityIssueDetected,
)

__all__ = [
    "EvaluationTemplateCreated",
    "EvaluationTemplateModified",
    "EvaluationTemplateDeactivated",
    "EvaluationCompleted",
    "EvaluationFailed",
    "ConsensusCalculated",
    "JudgeCalibrated",
    "QualityIssueDetected",
]
