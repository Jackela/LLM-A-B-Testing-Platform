"""Data Transfer Objects for analytics analysis requests."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...domain.analytics.entities.statistical_test import CorrectionMethod


@dataclass
class AnalysisRequestDTO:
    """DTO for analysis request parameters."""

    test_id: UUID
    confidence_level: float = 0.95
    correction_method: CorrectionMethod = CorrectionMethod.BONFERRONI
    include_effect_sizes: bool = True
    include_power_analysis: bool = True
    enable_dimension_analysis: bool = True
    enable_cost_analysis: bool = True
    minimum_sample_size: int = 30
    requested_by: Optional[str] = None
    analysis_name: Optional[str] = None
    custom_parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate DTO after initialization."""
        if not (0.5 <= self.confidence_level <= 0.99):
            raise ValueError("Confidence level must be between 0.5 and 0.99")

        if self.minimum_sample_size < 1:
            raise ValueError("Minimum sample size must be at least 1")

        if self.custom_parameters is None:
            self.custom_parameters = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_id": str(self.test_id),
            "confidence_level": self.confidence_level,
            "correction_method": self.correction_method.value,
            "include_effect_sizes": self.include_effect_sizes,
            "include_power_analysis": self.include_power_analysis,
            "enable_dimension_analysis": self.enable_dimension_analysis,
            "enable_cost_analysis": self.enable_cost_analysis,
            "minimum_sample_size": self.minimum_sample_size,
            "requested_by": self.requested_by,
            "analysis_name": self.analysis_name,
            "custom_parameters": self.custom_parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisRequestDTO":
        """Create DTO from dictionary."""
        return cls(
            test_id=UUID(data["test_id"]),
            confidence_level=data.get("confidence_level", 0.95),
            correction_method=CorrectionMethod(data.get("correction_method", "bonferroni")),
            include_effect_sizes=data.get("include_effect_sizes", True),
            include_power_analysis=data.get("include_power_analysis", True),
            enable_dimension_analysis=data.get("enable_dimension_analysis", True),
            enable_cost_analysis=data.get("enable_cost_analysis", True),
            minimum_sample_size=data.get("minimum_sample_size", 30),
            requested_by=data.get("requested_by"),
            analysis_name=data.get("analysis_name"),
            custom_parameters=data.get("custom_parameters", {}),
        )


@dataclass
class ModelComparisonRequestDTO:
    """DTO for model comparison analysis request."""

    test_id: UUID
    model_ids: List[str]
    comparison_dimensions: Optional[List[str]] = None
    confidence_level: float = 0.95
    correction_method: CorrectionMethod = CorrectionMethod.BONFERRONI
    include_pairwise_comparisons: bool = True
    include_effect_sizes: bool = True

    def __post_init__(self) -> None:
        """Validate DTO after initialization."""
        if len(self.model_ids) < 2:
            raise ValueError("At least 2 models required for comparison")

        if not (0.5 <= self.confidence_level <= 0.99):
            raise ValueError("Confidence level must be between 0.5 and 0.99")

        if self.comparison_dimensions is None:
            self.comparison_dimensions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_id": str(self.test_id),
            "model_ids": self.model_ids,
            "comparison_dimensions": self.comparison_dimensions,
            "confidence_level": self.confidence_level,
            "correction_method": self.correction_method.value,
            "include_pairwise_comparisons": self.include_pairwise_comparisons,
            "include_effect_sizes": self.include_effect_sizes,
        }


@dataclass
class DimensionAnalysisRequestDTO:
    """DTO for dimension performance analysis request."""

    test_id: UUID
    dimensions: List[str]
    confidence_level: float = 0.95
    include_correlations: bool = True
    include_dimension_weights: bool = False
    custom_weights: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        """Validate DTO after initialization."""
        if not self.dimensions:
            raise ValueError("At least one dimension required")

        if not (0.5 <= self.confidence_level <= 0.99):
            raise ValueError("Confidence level must be between 0.5 and 0.99")

        if self.custom_weights is None:
            self.custom_weights = {}

        # Validate weights sum to 1 if provided
        if self.custom_weights:
            total_weight = sum(self.custom_weights.values())
            if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
                raise ValueError("Custom weights must sum to 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_id": str(self.test_id),
            "dimensions": self.dimensions,
            "confidence_level": self.confidence_level,
            "include_correlations": self.include_correlations,
            "include_dimension_weights": self.include_dimension_weights,
            "custom_weights": self.custom_weights,
        }


@dataclass
class PowerAnalysisRequestDTO:
    """DTO for statistical power analysis request."""

    effect_sizes: Dict[str, float]
    desired_power: float = 0.8
    significance_level: float = 0.05
    test_types: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate DTO after initialization."""
        if not self.effect_sizes:
            raise ValueError("At least one effect size required")

        if not (0.5 <= self.desired_power <= 0.99):
            raise ValueError("Desired power must be between 0.5 and 0.99")

        if not (0.01 <= self.significance_level <= 0.1):
            raise ValueError("Significance level must be between 0.01 and 0.1")

        for test_name, effect_size in self.effect_sizes.items():
            if effect_size <= 0:
                raise ValueError(f"Effect size for {test_name} must be positive")

        if self.test_types is None:
            self.test_types = ["ttest_independent"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "effect_sizes": self.effect_sizes,
            "desired_power": self.desired_power,
            "significance_level": self.significance_level,
            "test_types": self.test_types,
        }


@dataclass
class AnalysisResponseDTO:
    """DTO for analysis response."""

    analysis_id: UUID
    test_id: UUID
    status: str  # "completed", "in_progress", "failed"
    created_at: str  # ISO format datetime
    completed_at: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    results_summary: Optional[Dict[str, Any]] = None
    download_urls: Optional[Dict[str, str]] = None  # Format -> URL mapping

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "analysis_id": str(self.analysis_id),
            "test_id": str(self.test_id),
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "processing_time_ms": self.processing_time_ms,
            "error_message": self.error_message,
            "results_summary": self.results_summary,
            "download_urls": self.download_urls,
        }

    @property
    def is_completed(self) -> bool:
        """Check if analysis is completed."""
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        """Check if analysis failed."""
        return self.status == "failed"

    @property
    def is_in_progress(self) -> bool:
        """Check if analysis is in progress."""
        return self.status == "in_progress"
