"""Domain events for evaluation domain."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass(frozen=True)
class EvaluationDomainEvent:
    """Base class for evaluation domain events."""

    occurred_at: datetime
    event_id: UUID
    aggregate_id: UUID


@dataclass(frozen=True)
class EvaluationTemplateCreated(EvaluationDomainEvent):
    """Event fired when evaluation template is created."""

    template_name: str
    dimensions_count: int
    created_by: Optional[str] = None


@dataclass(frozen=True)
class EvaluationTemplateModified(EvaluationDomainEvent):
    """Event fired when evaluation template is modified."""

    modification_type: str  # "dimension_added", "dimension_removed", "weight_updated", etc.
    details: Dict[str, Any]


@dataclass(frozen=True)
class EvaluationTemplateDeactivated(EvaluationDomainEvent):
    """Event fired when evaluation template is deactivated."""

    template_name: str
    version: int


@dataclass(frozen=True)
class EvaluationCompleted(EvaluationDomainEvent):
    """Event fired when evaluation is completed successfully."""

    judge_id: str
    template_id: UUID
    overall_score: str  # Decimal as string
    dimension_scores: Dict[str, int]
    confidence_score: str  # Decimal as string
    evaluation_time_ms: int


@dataclass(frozen=True)
class EvaluationFailed(EvaluationDomainEvent):
    """Event fired when evaluation fails."""

    judge_id: str
    template_id: UUID
    error_type: str
    error_message: str
    retry_count: int = 0


@dataclass(frozen=True)
class ConsensusCalculated(EvaluationDomainEvent):
    """Event fired when consensus is calculated from multiple judges."""

    consensus_score: str  # Decimal as string
    agreement_level: str  # Decimal as string
    judge_count: int
    outlier_count: int
    consensus_method: str


@dataclass(frozen=True)
class JudgeCalibrated(EvaluationDomainEvent):
    """Event fired when judge is calibrated."""

    judge_id: str
    accuracy: str  # Decimal as string
    consistency: str  # Decimal as string
    bias_score: str  # Decimal as string
    sample_size: int
    is_production_ready: bool


@dataclass(frozen=True)
class QualityIssueDetected(EvaluationDomainEvent):
    """Event fired when quality control detects an issue."""

    issue_type: str
    severity: str  # "high", "medium", "low"
    description: str
    affected_component: Optional[str] = None
    evaluation_id: Optional[UUID] = None
    judge_id: Optional[str] = None
