"""Data Transfer Objects for performance metrics."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID


@dataclass
class MetricConfigurationDTO:
    """DTO for metric calculation configuration."""

    include_confidence_intervals: bool = True
    confidence_level: float = 0.95
    outlier_detection: bool = True
    outlier_threshold: float = 2.0  # Standard deviations
    weight_by_sample_size: bool = True
    normalize_scores: bool = True
    include_trend_analysis: bool = True
    custom_weights: Optional[Dict[str, float]] = None
    aggregation_methods: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate DTO after initialization."""
        if not (0.5 <= self.confidence_level <= 0.99):
            raise ValueError("Confidence level must be between 0.5 and 0.99")

        if self.outlier_threshold <= 0:
            raise ValueError("Outlier threshold must be positive")

        if self.custom_weights is None:
            self.custom_weights = {}

        if self.aggregation_methods is None:
            self.aggregation_methods = ["mean"]

        # Validate custom weights if provided
        if self.custom_weights:
            total_weight = sum(self.custom_weights.values())
            if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
                raise ValueError("Custom weights must sum to 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "include_confidence_intervals": self.include_confidence_intervals,
            "confidence_level": self.confidence_level,
            "outlier_detection": self.outlier_detection,
            "outlier_threshold": self.outlier_threshold,
            "weight_by_sample_size": self.weight_by_sample_size,
            "normalize_scores": self.normalize_scores,
            "include_trend_analysis": self.include_trend_analysis,
            "custom_weights": self.custom_weights,
            "aggregation_methods": self.aggregation_methods,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricConfigurationDTO":
        """Create DTO from dictionary."""
        return cls(
            include_confidence_intervals=data.get("include_confidence_intervals", True),
            confidence_level=data.get("confidence_level", 0.95),
            outlier_detection=data.get("outlier_detection", True),
            outlier_threshold=data.get("outlier_threshold", 2.0),
            weight_by_sample_size=data.get("weight_by_sample_size", True),
            normalize_scores=data.get("normalize_scores", True),
            include_trend_analysis=data.get("include_trend_analysis", True),
            custom_weights=data.get("custom_weights", {}),
            aggregation_methods=data.get("aggregation_methods", ["mean"]),
        )


@dataclass
class CalculatedMetricDTO:
    """DTO for a calculated metric."""

    metric_type: str
    value: float
    confidence_interval: Optional[Tuple[float, float]]
    sample_count: int
    calculation_method: str
    metadata: Dict[str, Any]
    calculated_at: str  # ISO format datetime
    outliers_detected: int = 0
    trend_direction: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_type": self.metric_type,
            "value": self.value,
            "confidence_interval": (
                list(self.confidence_interval) if self.confidence_interval else None
            ),
            "sample_count": self.sample_count,
            "calculation_method": self.calculation_method,
            "metadata": self.metadata,
            "calculated_at": self.calculated_at,
            "outliers_detected": self.outliers_detected,
            "trend_direction": self.trend_direction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalculatedMetricDTO":
        """Create DTO from dictionary."""
        ci = None
        if data.get("confidence_interval"):
            ci = tuple(data["confidence_interval"])

        return cls(
            metric_type=data["metric_type"],
            value=data["value"],
            confidence_interval=ci,
            sample_count=data["sample_count"],
            calculation_method=data["calculation_method"],
            metadata=data["metadata"],
            calculated_at=data["calculated_at"],
            outliers_detected=data.get("outliers_detected", 0),
            trend_direction=data.get("trend_direction"),
        )


@dataclass
class ModelMetricsSummaryDTO:
    """DTO for model metrics summary."""

    model_id: str
    model_name: str
    overall_score: float
    individual_metrics: Dict[str, CalculatedMetricDTO]
    composite_scores: Dict[str, float]
    ranking_position: Optional[int]
    percentile_rank: Optional[float]
    quality_grade: str
    recommendations: List[str]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate DTO after initialization."""
        if not (0.0 <= self.overall_score <= 1.0):
            raise ValueError("Overall score must be between 0.0 and 1.0")

        if self.percentile_rank is not None and not (0.0 <= self.percentile_rank <= 100.0):
            raise ValueError("Percentile rank must be between 0.0 and 100.0")

        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "overall_score": self.overall_score,
            "individual_metrics": {
                name: metric.to_dict() for name, metric in self.individual_metrics.items()
            },
            "composite_scores": self.composite_scores,
            "ranking_position": self.ranking_position,
            "percentile_rank": self.percentile_rank,
            "quality_grade": self.quality_grade,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetricsSummaryDTO":
        """Create DTO from dictionary."""
        individual_metrics = {}
        for name, metric_data in data.get("individual_metrics", {}).items():
            individual_metrics[name] = CalculatedMetricDTO.from_dict(metric_data)

        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            overall_score=data["overall_score"],
            individual_metrics=individual_metrics,
            composite_scores=data.get("composite_scores", {}),
            ranking_position=data.get("ranking_position"),
            percentile_rank=data.get("percentile_rank"),
            quality_grade=data["quality_grade"],
            recommendations=data.get("recommendations", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CostEfficiencyMetricDTO:
    """DTO for cost efficiency metrics."""

    model_id: str
    model_name: str
    efficiency_ratio: float
    cost_adjusted_score: float
    value_score: float
    cost_per_sample: float
    performance_score: float
    cost_percentile: Optional[float]
    efficiency_rank: Optional[int]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "efficiency_ratio": self.efficiency_ratio,
            "cost_adjusted_score": self.cost_adjusted_score,
            "value_score": self.value_score,
            "cost_per_sample": self.cost_per_sample,
            "performance_score": self.performance_score,
            "cost_percentile": self.cost_percentile,
            "efficiency_rank": self.efficiency_rank,
            "recommendations": self.recommendations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostEfficiencyMetricDTO":
        """Create DTO from dictionary."""
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            efficiency_ratio=data["efficiency_ratio"],
            cost_adjusted_score=data["cost_adjusted_score"],
            value_score=data["value_score"],
            cost_per_sample=data["cost_per_sample"],
            performance_score=data["performance_score"],
            cost_percentile=data.get("cost_percentile"),
            efficiency_rank=data.get("efficiency_rank"),
            recommendations=data.get("recommendations", []),
        )


@dataclass
class TrendMetricDTO:
    """DTO for trend analysis metrics."""

    trend_direction: str  # "improving", "declining", "stable"
    slope: float
    data_points: int
    score_range: Tuple[float, float]
    latest_score: float
    score_change: float
    time_window_hours: Optional[int] = None
    confidence_level: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trend_direction": self.trend_direction,
            "slope": self.slope,
            "data_points": self.data_points,
            "score_range": list(self.score_range),
            "latest_score": self.latest_score,
            "score_change": self.score_change,
            "time_window_hours": self.time_window_hours,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrendMetricDTO":
        """Create DTO from dictionary."""
        return cls(
            trend_direction=data["trend_direction"],
            slope=data["slope"],
            data_points=data["data_points"],
            score_range=tuple(data["score_range"]),
            latest_score=data["latest_score"],
            score_change=data["score_change"],
            time_window_hours=data.get("time_window_hours"),
            confidence_level=data.get("confidence_level"),
        )


@dataclass
class ReliabilityMetricDTO:
    """DTO for reliability metrics."""

    completion_rate: float
    error_rate: float
    score_coefficient_of_variation: float
    response_time_coefficient_of_variation: float
    reliability_score: float
    total_evaluations: int
    completed_evaluations: int
    failed_evaluations: int
    consistency_grade: str

    def __post_init__(self) -> None:
        """Validate DTO after initialization."""
        if not (0.0 <= self.completion_rate <= 1.0):
            raise ValueError("Completion rate must be between 0.0 and 1.0")

        if not (0.0 <= self.error_rate <= 1.0):
            raise ValueError("Error rate must be between 0.0 and 1.0")

        if not (0.0 <= self.reliability_score <= 1.0):
            raise ValueError("Reliability score must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "completion_rate": self.completion_rate,
            "error_rate": self.error_rate,
            "score_coefficient_of_variation": self.score_coefficient_of_variation,
            "response_time_coefficient_of_variation": self.response_time_coefficient_of_variation,
            "reliability_score": self.reliability_score,
            "total_evaluations": self.total_evaluations,
            "completed_evaluations": self.completed_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "consistency_grade": self.consistency_grade,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReliabilityMetricDTO":
        """Create DTO from dictionary."""
        return cls(
            completion_rate=data["completion_rate"],
            error_rate=data["error_rate"],
            score_coefficient_of_variation=data["score_coefficient_of_variation"],
            response_time_coefficient_of_variation=data["response_time_coefficient_of_variation"],
            reliability_score=data["reliability_score"],
            total_evaluations=data["total_evaluations"],
            completed_evaluations=data["completed_evaluations"],
            failed_evaluations=data["failed_evaluations"],
            consistency_grade=data["consistency_grade"],
        )


@dataclass
class MetricsRequestDTO:
    """DTO for metrics calculation request."""

    test_id: UUID
    configuration: MetricConfigurationDTO
    include_cost_efficiency: bool = True
    include_trend_analysis: bool = True
    include_reliability_analysis: bool = True
    time_window_hours: Optional[int] = None
    requested_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_id": str(self.test_id),
            "configuration": self.configuration.to_dict(),
            "include_cost_efficiency": self.include_cost_efficiency,
            "include_trend_analysis": self.include_trend_analysis,
            "include_reliability_analysis": self.include_reliability_analysis,
            "time_window_hours": self.time_window_hours,
            "requested_by": self.requested_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsRequestDTO":
        """Create DTO from dictionary."""
        config = MetricConfigurationDTO.from_dict(data["configuration"])

        return cls(
            test_id=UUID(data["test_id"]),
            configuration=config,
            include_cost_efficiency=data.get("include_cost_efficiency", True),
            include_trend_analysis=data.get("include_trend_analysis", True),
            include_reliability_analysis=data.get("include_reliability_analysis", True),
            time_window_hours=data.get("time_window_hours"),
            requested_by=data.get("requested_by"),
        )


@dataclass
class MetricsResponseDTO:
    """DTO for metrics calculation response."""

    test_id: UUID
    model_summaries: Dict[str, ModelMetricsSummaryDTO]
    cost_efficiency_metrics: Optional[Dict[str, CostEfficiencyMetricDTO]]
    trend_metrics: Optional[Dict[str, TrendMetricDTO]]
    reliability_metrics: Optional[Dict[str, ReliabilityMetricDTO]]
    overall_insights: List[str]
    calculation_metadata: Dict[str, Any]
    created_at: str  # ISO format datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_id": str(self.test_id),
            "model_summaries": {
                model_id: summary.to_dict() for model_id, summary in self.model_summaries.items()
            },
            "cost_efficiency_metrics": (
                {
                    model_id: metric.to_dict()
                    for model_id, metric in self.cost_efficiency_metrics.items()
                }
                if self.cost_efficiency_metrics
                else None
            ),
            "trend_metrics": (
                {model_id: metric.to_dict() for model_id, metric in self.trend_metrics.items()}
                if self.trend_metrics
                else None
            ),
            "reliability_metrics": (
                {
                    model_id: metric.to_dict()
                    for model_id, metric in self.reliability_metrics.items()
                }
                if self.reliability_metrics
                else None
            ),
            "overall_insights": self.overall_insights,
            "calculation_metadata": self.calculation_metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsResponseDTO":
        """Create DTO from dictionary."""
        model_summaries = {}
        for model_id, summary_data in data["model_summaries"].items():
            model_summaries[model_id] = ModelMetricsSummaryDTO.from_dict(summary_data)

        cost_efficiency_metrics = None
        if data.get("cost_efficiency_metrics"):
            cost_efficiency_metrics = {}
            for model_id, metric_data in data["cost_efficiency_metrics"].items():
                cost_efficiency_metrics[model_id] = CostEfficiencyMetricDTO.from_dict(metric_data)

        trend_metrics = None
        if data.get("trend_metrics"):
            trend_metrics = {}
            for model_id, metric_data in data["trend_metrics"].items():
                trend_metrics[model_id] = TrendMetricDTO.from_dict(metric_data)

        reliability_metrics = None
        if data.get("reliability_metrics"):
            reliability_metrics = {}
            for model_id, metric_data in data["reliability_metrics"].items():
                reliability_metrics[model_id] = ReliabilityMetricDTO.from_dict(metric_data)

        return cls(
            test_id=UUID(data["test_id"]),
            model_summaries=model_summaries,
            cost_efficiency_metrics=cost_efficiency_metrics,
            trend_metrics=trend_metrics,
            reliability_metrics=reliability_metrics,
            overall_insights=data.get("overall_insights", []),
            calculation_metadata=data.get("calculation_metadata", {}),
            created_at=data["created_at"],
        )
