"""Analytics domain events."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...test_management.events.test_events import DomainEvent


@dataclass(frozen=True)
class AnalyticsDomainEvent(DomainEvent):
    """Base analytics domain event."""

    pass


@dataclass(frozen=True)
class StatisticalTestCompleted(AnalyticsDomainEvent):
    """Event fired when statistical test completes."""

    test_id: UUID
    test_type: str
    p_value: float
    effect_size: float
    is_significant: bool
    sample_sizes: Dict[str, int]
    confidence_level: float

    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class AnalysisCompleted(AnalyticsDomainEvent):
    """Event fired when comprehensive analysis completes."""

    analysis_id: UUID
    test_id: UUID
    models_analyzed: List[str]
    statistical_tests_count: int
    insights_generated: int
    total_samples: int
    analysis_duration_ms: int

    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class InsightGenerated(AnalyticsDomainEvent):
    """Event fired when new insight is generated."""

    insight_id: UUID
    insight_type: str
    severity: str
    confidence_score: float
    affected_models: Optional[List[str]]
    category: Optional[str]

    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class ModelComparisonCompleted(AnalyticsDomainEvent):
    """Event fired when model comparison completes."""

    comparison_id: UUID
    model_a_id: str
    model_b_id: str
    performance_difference: float
    cost_difference: float
    statistical_significance: bool
    recommendation: str

    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class AggregationCompleted(AnalyticsDomainEvent):
    """Event fired when data aggregation completes."""

    aggregation_id: UUID
    test_id: UUID
    aggregation_rules_applied: int
    samples_processed: int
    dimensions_analyzed: int
    processing_time_ms: int

    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class QualityThresholdViolated(AnalyticsDomainEvent):
    """Event fired when analysis quality threshold is violated."""

    test_id: UUID
    analysis_id: UUID
    violation_type: str
    threshold_value: float
    actual_value: float
    severity: str
    recommendation: str

    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class InsufficientDataDetected(AnalyticsDomainEvent):
    """Event fired when insufficient data is detected for analysis."""

    test_id: UUID
    analysis_type: str
    required_sample_size: int
    actual_sample_size: int
    recommendation: str

    def __post_init__(self):
        super().__post_init__()
