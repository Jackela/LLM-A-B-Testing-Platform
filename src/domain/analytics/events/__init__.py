"""Analytics domain events."""

from .analytics_events import (
    AggregationCompleted,
    AnalysisCompleted,
    AnalyticsDomainEvent,
    InsightGenerated,
    ModelComparisonCompleted,
    StatisticalTestCompleted,
)

__all__ = [
    "AnalyticsDomainEvent",
    "AnalysisCompleted",
    "StatisticalTestCompleted",
    "InsightGenerated",
    "ModelComparisonCompleted",
    "AggregationCompleted",
]
