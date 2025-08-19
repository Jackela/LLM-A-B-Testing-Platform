"""DTOs for consensus results and related data structures."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

if TYPE_CHECKING:
    from .evaluation_request_dto import QualityReportDTO


@dataclass(frozen=True)
class IndividualEvaluationDTO:
    """DTO for individual judge evaluation results."""

    evaluation_id: UUID
    judge_id: str
    overall_score: Decimal
    dimension_scores: Dict[str, Decimal]
    confidence_score: Decimal
    reasoning: str
    evaluation_time_ms: int
    template_id: UUID
    is_successful: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class ConsensusResultDTO:
    """DTO for consensus evaluation results."""

    consensus_score: Decimal
    confidence_level: Decimal
    agreement_score: Decimal
    dimension_scores: Dict[str, Decimal]
    judge_count: int
    effective_judge_count: int
    outlier_judges: List[str]
    consensus_method: str
    statistical_significance: Decimal
    confidence_interval_lower: Decimal
    confidence_interval_upper: Decimal
    evaluation_metadata: Dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class ConsensusEvaluationResultDTO:
    """DTO for complete consensus evaluation result including all judge results."""

    consensus: ConsensusResultDTO
    individual_results: List[IndividualEvaluationDTO]
    quality_report: "QualityReportDTO"
    evaluation_metadata: Dict[str, Any]
    total_judges: int
    successful_judges: int
    failed_judges: int
    outlier_judges: List[str]
    consensus_reached: bool
    evaluation_duration_ms: int
    created_at: datetime

    @property
    def success_rate(self) -> float:
        """Calculate success rate of judge evaluations."""
        if self.total_judges == 0:
            return 0.0
        return (self.successful_judges / self.total_judges) * 100.0

    @property
    def effective_consensus(self) -> bool:
        """Check if consensus is based on sufficient judges."""
        effective_judges = self.successful_judges - len(self.outlier_judges)
        return effective_judges >= 2 and self.consensus_reached


@dataclass(frozen=True)
class StatisticalSignificanceDTO:
    """DTO for statistical significance analysis."""

    p_value: Decimal
    confidence_level: Decimal
    is_significant: bool
    test_statistic: Decimal
    degrees_of_freedom: int
    statistical_method: str
    effect_size: Optional[Decimal] = None
    power_analysis: Optional[Dict[str, Any]] = None

    @property
    def significance_level(self) -> str:
        """Return significance level as string."""
        p_val = float(self.p_value)
        if p_val < 0.001:
            return "highly_significant"
        elif p_val < 0.01:
            return "very_significant"
        elif p_val < 0.05:
            return "significant"
        elif p_val < 0.1:
            return "marginally_significant"
        else:
            return "not_significant"


@dataclass(frozen=True)
class AgreementAnalysisDTO:
    """DTO for inter-judge agreement analysis."""

    agreement_coefficient: Decimal
    agreement_level: str  # low, moderate, high, very_high
    pairwise_agreements: Dict[str, Decimal]  # judge_pair -> agreement
    outlier_analysis: Dict[str, Any]
    consistency_metrics: Dict[str, Decimal]
    reliability_score: Decimal

    @property
    def is_reliable(self) -> bool:
        """Check if agreement is reliable for decision making."""
        return float(self.agreement_coefficient) >= 0.6 and self.agreement_level in [
            "high",
            "very_high",
        ]


@dataclass(frozen=True)
class DimensionConsensusDTO:
    """DTO for dimension-specific consensus results."""

    dimension_name: str
    consensus_score: Decimal
    judge_scores: Dict[str, Decimal]  # judge_id -> score
    agreement_level: Decimal
    confidence_interval: tuple[Decimal, Decimal]
    outliers: List[str]
    statistical_significance: StatisticalSignificanceDTO

    @property
    def score_range(self) -> Decimal:
        """Calculate range of scores for this dimension."""
        if not self.judge_scores:
            return Decimal("0")
        scores = list(self.judge_scores.values())
        return max(scores) - min(scores)


@dataclass(frozen=True)
class EvaluationMetricsDTO:
    """DTO for comprehensive evaluation metrics."""

    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    consensus_evaluations: int
    high_agreement_evaluations: int
    outlier_rate: Decimal
    average_consensus_score: Decimal
    average_agreement_level: Decimal
    average_confidence: Decimal
    evaluation_time_statistics: Dict[str, int]  # min, max, avg, p95
    quality_distribution: Dict[str, int]  # quality_level -> count

    @property
    def success_rate(self) -> Decimal:
        """Calculate overall success rate."""
        if self.total_evaluations == 0:
            return Decimal("0")
        return Decimal(str(self.successful_evaluations / self.total_evaluations))

    @property
    def consensus_rate(self) -> Decimal:
        """Calculate rate of achieving consensus."""
        if self.successful_evaluations == 0:
            return Decimal("0")
        return Decimal(str(self.consensus_evaluations / self.successful_evaluations))

    @property
    def reliability_score(self) -> Decimal:
        """Calculate overall reliability score."""
        factors = [
            self.success_rate,
            self.consensus_rate,
            min(Decimal("1"), Decimal("1") - self.outlier_rate),
            self.average_agreement_level,
        ]
        return Decimal(sum(factors)) / Decimal(len(factors))


@dataclass(frozen=True)
class ConsensusConfigurationDTO:
    """DTO for consensus algorithm configuration."""

    algorithm: str  # weighted_average, median, trimmed_mean, robust_average
    minimum_judges: int
    outlier_detection: bool
    outlier_threshold: Decimal
    confidence_weighting: bool
    confidence_weight_factor: Decimal
    agreement_threshold: Decimal
    statistical_significance_threshold: Decimal

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []

        if self.algorithm not in ["weighted_average", "median", "trimmed_mean", "robust_average"]:
            errors.append(f"Invalid algorithm: {self.algorithm}")

        if self.minimum_judges < 2:
            errors.append("minimum_judges must be at least 2")

        if not (0 <= self.outlier_threshold <= 5):
            errors.append("outlier_threshold must be between 0 and 5")

        if not (0 <= self.confidence_weight_factor <= 1):
            errors.append("confidence_weight_factor must be between 0 and 1")

        if not (0 <= self.agreement_threshold <= 1):
            errors.append("agreement_threshold must be between 0 and 1")

        if not (0 <= self.statistical_significance_threshold <= 1):
            errors.append("statistical_significance_threshold must be between 0 and 1")

        return errors

    @classmethod
    def default(cls) -> "ConsensusConfigurationDTO":
        """Create default consensus configuration."""
        return cls(
            algorithm="weighted_average",
            minimum_judges=3,
            outlier_detection=True,
            outlier_threshold=Decimal("2.0"),
            confidence_weighting=True,
            confidence_weight_factor=Decimal("0.3"),
            agreement_threshold=Decimal("0.6"),
            statistical_significance_threshold=Decimal("0.05"),
        )
