"""DTOs for evaluation requests and responses."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

if TYPE_CHECKING:
    from .consensus_result_dto import ConsensusResultDTO


@dataclass(frozen=True)
class EvaluationRequestDTO:
    """DTO for evaluation requests."""

    model_response_id: UUID
    test_sample_id: UUID
    judge_ids: List[str]
    evaluation_config: Dict[str, Any]
    template_id: UUID
    request_metadata: Optional[Dict[str, Any]] = None
    priority: str = "normal"  # low, normal, high
    timeout_seconds: int = 300


@dataclass(frozen=True)
class EvaluationConfigDTO:
    """DTO for evaluation configuration."""

    minimum_judges: int
    consensus_method: str  # weighted_average, median, trimmed_mean, robust_average
    confidence_weighting: bool
    exclude_outliers: bool
    consensus_threshold: Decimal
    quality_threshold: Decimal
    max_evaluation_time_seconds: int
    retry_failed_evaluations: bool
    cache_evaluations: bool
    batch_size: int = 5
    max_concurrent_evaluations: int = 10

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.minimum_judges < 2:
            raise ValueError("minimum_judges must be at least 2")

        if self.consensus_method not in [
            "weighted_average",
            "median",
            "trimmed_mean",
            "robust_average",
        ]:
            raise ValueError(f"Invalid consensus method: {self.consensus_method}")

        if not (0 <= self.consensus_threshold <= 1):
            raise ValueError("consensus_threshold must be between 0 and 1")

        if not (0 <= self.quality_threshold <= 1):
            raise ValueError("quality_threshold must be between 0 and 1")


@dataclass(frozen=True)
class QualityReportDTO:
    """DTO for evaluation quality reports."""

    overall_quality_score: Decimal
    reasoning_quality: Decimal
    consistency_score: Decimal
    confidence_calibration: Decimal
    bias_indicators: Dict[str, Decimal]
    quality_passed: bool
    quality_issues: List[str]
    recommendations: List[str]
    quality_metrics: Dict[str, Any]
    generated_at: datetime


@dataclass(frozen=True)
class TestEvaluationResultDTO:
    """DTO for complete test evaluation results."""

    test_id: UUID
    model_evaluations: Dict[str, List["ConsensusResultDTO"]]  # model_id -> evaluations
    evaluation_summary: Dict[str, Any]
    quality_metrics: QualityReportDTO
    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    average_evaluation_time_ms: int
    total_evaluation_time_ms: int
    created_at: datetime


@dataclass(frozen=True)
class EvaluationBatchDTO:
    """DTO for batch evaluation requests."""

    batch_id: UUID
    evaluation_requests: List[EvaluationRequestDTO]
    batch_config: EvaluationConfigDTO
    priority: str = "normal"
    estimated_completion_time: Optional[int] = None
    created_at: Optional[datetime] = None

    def __len__(self) -> int:
        """Return number of evaluations in batch."""
        return len(self.evaluation_requests)

    def is_empty(self) -> bool:
        """Check if batch is empty."""
        return len(self.evaluation_requests) == 0


@dataclass(frozen=True)
class EvaluationProgressDTO:
    """DTO for evaluation progress tracking."""

    batch_id: UUID
    total_evaluations: int
    completed_evaluations: int
    failed_evaluations: int
    in_progress_evaluations: int
    pending_evaluations: int
    average_time_per_evaluation_ms: int
    estimated_completion_time: Optional[datetime]
    last_update: datetime

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_evaluations == 0:
            return 0.0
        return (self.completed_evaluations / self.total_evaluations) * 100.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        processed = self.completed_evaluations + self.failed_evaluations
        if processed == 0:
            return 0.0
        return (self.completed_evaluations / processed) * 100.0


@dataclass(frozen=True)
class JudgePerformanceDTO:
    """DTO for judge performance metrics."""

    judge_id: str
    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    average_evaluation_time_ms: int
    average_confidence_score: Decimal
    consistency_score: Decimal
    bias_score: Decimal
    quality_score: Decimal
    is_production_ready: bool
    last_evaluation: Optional[datetime]
    performance_trend: str  # improving, stable, declining
