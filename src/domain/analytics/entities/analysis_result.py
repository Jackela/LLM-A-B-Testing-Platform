"""Analysis result aggregate for comprehensive test analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..exceptions import ValidationError
from ..value_objects.cost_data import CostData
from ..value_objects.insight import Insight
from ..value_objects.test_result import TestResult


@dataclass
class AggregatedData:
    """Aggregated data for specific grouping."""

    group_key: tuple
    group_labels: Dict[str, str]
    sample_count: int
    aggregated_value: Decimal
    confidence_interval: Optional[tuple[Decimal, Decimal]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate aggregated data."""
        if self.sample_count < 0:
            raise ValidationError("Sample count cannot be negative")

        if (
            self.confidence_interval
            and len(self.confidence_interval) == 2
            and self.confidence_interval[0] > self.confidence_interval[1]
        ):
            raise ValidationError("Confidence interval lower bound must be <= upper bound")


@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics from aggregation."""

    model_id: str
    model_name: str
    overall_score: Decimal
    dimension_scores: Dict[str, Decimal]
    sample_count: int
    confidence_score: Decimal
    cost_metrics: Optional[CostData] = None
    quality_indicators: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate model performance metrics."""
        if not self.model_id.strip():
            raise ValidationError("Model ID cannot be empty")

        if self.sample_count < 0:
            raise ValidationError("Sample count cannot be negative")

        if not (Decimal("0") <= self.overall_score <= Decimal("1")):
            raise ValidationError("Overall score must be between 0 and 1")

        if not (Decimal("0") <= self.confidence_score <= Decimal("1")):
            raise ValidationError("Confidence score must be between 0 and 1")


@dataclass
class AnalysisResult:
    """Analysis result aggregate containing comprehensive test analysis."""

    analysis_id: UUID
    test_id: UUID
    name: str
    description: str
    created_at: datetime
    completed_at: Optional[datetime]

    # Statistical analysis results
    statistical_tests: Dict[str, TestResult] = field(default_factory=dict)

    # Aggregated data by different groupings
    aggregated_data: Dict[str, List[AggregatedData]] = field(default_factory=dict)

    # Model performance metrics
    model_performances: Dict[str, ModelPerformanceMetrics] = field(default_factory=dict)

    # Generated insights
    insights: List[Insight] = field(default_factory=list)

    # Analysis metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Domain events
    _domain_events: List[object] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate analysis result after creation."""
        if not self.analysis_id:
            self.analysis_id = uuid4()

        if not self.name.strip():
            raise ValidationError("Analysis name cannot be empty")

    @classmethod
    def create(cls, test_id: UUID, name: str, description: str) -> "AnalysisResult":
        """Factory method to create new analysis result."""

        return cls(
            analysis_id=uuid4(),
            test_id=test_id,
            name=name,
            description=description,
            created_at=datetime.utcnow(),
            completed_at=None,
        )

    def add_statistical_test(self, test_name: str, test_result: TestResult) -> None:
        """Add statistical test result."""
        if not test_name.strip():
            raise ValidationError("Test name cannot be empty")

        self.statistical_tests[test_name] = test_result

        # Add domain event
        from ..events.analytics_events import StatisticalTestCompleted

        event = StatisticalTestCompleted(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            test_id=self.test_id,
            test_type=test_result.test_type,
            p_value=test_result.p_value,
            effect_size=test_result.effect_size,
            is_significant=test_result.is_significant(),
            sample_sizes=test_result.sample_sizes,
            confidence_level=1.0 - test_result.interpretation.significance_level,
        )

        self._domain_events.append(event)

    def add_aggregated_data(self, aggregation_name: str, data: List[AggregatedData]) -> None:
        """Add aggregated data for specific aggregation rule."""
        if not aggregation_name.strip():
            raise ValidationError("Aggregation name cannot be empty")

        if not data:
            raise ValidationError("Aggregated data cannot be empty")

        self.aggregated_data[aggregation_name] = data

    def add_model_performance(
        self, model_id: str, performance_metrics: ModelPerformanceMetrics
    ) -> None:
        """Add model performance metrics."""
        if not model_id.strip():
            raise ValidationError("Model ID cannot be empty")

        self.model_performances[model_id] = performance_metrics

    def add_insight(self, insight: Insight) -> None:
        """Add generated insight."""
        self.insights.append(insight)

        # Add domain event
        from ..events.analytics_events import InsightGenerated

        event = InsightGenerated(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            insight_id=insight.insight_id,
            insight_type=insight.insight_type.value,
            severity=insight.severity.value,
            confidence_score=float(insight.confidence_score),
            affected_models=insight.affected_models,
            category=insight.category,
        )

        self._domain_events.append(event)

    def complete_analysis(self, processing_time_ms: int) -> None:
        """Mark analysis as completed."""
        if self.is_completed():
            raise ValidationError("Analysis is already completed")

        self.completed_at = datetime.utcnow()
        self.metadata["processing_time_ms"] = processing_time_ms

        # Add completion event
        from ..events.analytics_events import AnalysisCompleted

        event = AnalysisCompleted(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            analysis_id=self.analysis_id,
            test_id=self.test_id,
            models_analyzed=list(self.model_performances.keys()),
            statistical_tests_count=len(self.statistical_tests),
            insights_generated=len(self.insights),
            total_samples=self.get_total_sample_count(),
            analysis_duration_ms=processing_time_ms,
        )

        self._domain_events.append(event)

    def is_completed(self) -> bool:
        """Check if analysis is completed."""
        return self.completed_at is not None

    def get_total_sample_count(self) -> int:
        """Get total sample count across all model performances."""
        return sum(perf.sample_count for perf in self.model_performances.values())

    def get_significant_tests(self, alpha: float = 0.05) -> Dict[str, TestResult]:
        """Get statistically significant test results."""
        return {
            name: result
            for name, result in self.statistical_tests.items()
            if result.is_significant(alpha)
        }

    def get_high_confidence_insights(self, threshold: Decimal = Decimal("0.8")) -> List[Insight]:
        """Get high confidence insights."""
        return [insight for insight in self.insights if insight.is_high_confidence(threshold)]

    def get_actionable_insights(self) -> List[Insight]:
        """Get actionable insights."""
        return [insight for insight in self.insights if insight.is_actionable()]

    def get_insights_by_severity(self, severity: str) -> List[Insight]:
        """Get insights filtered by severity level."""
        return [insight for insight in self.insights if insight.severity.value == severity]

    def get_best_performing_model(self) -> Optional[ModelPerformanceMetrics]:
        """Get the best performing model based on overall score."""
        if not self.model_performances:
            return None

        return max(self.model_performances.values(), key=lambda perf: perf.overall_score)

    def get_most_cost_effective_model(self) -> Optional[ModelPerformanceMetrics]:
        """Get the most cost-effective model."""
        cost_effective_models = [
            perf for perf in self.model_performances.values() if perf.cost_metrics is not None
        ]

        if not cost_effective_models:
            return None

        # Calculate cost-effectiveness score (performance per dollar)
        def cost_effectiveness_score(perf: ModelPerformanceMetrics) -> Decimal:
            if perf.cost_metrics.total_cost.amount <= 0:
                return Decimal("inf")  # Free is maximally cost-effective

            return perf.overall_score / perf.cost_metrics.total_cost.amount

        return max(cost_effective_models, key=cost_effectiveness_score)

    def get_model_comparison_summary(self) -> Dict[str, Any]:
        """Get summary of model comparisons."""
        if len(self.model_performances) < 2:
            return {"error": "At least 2 models required for comparison"}

        models = list(self.model_performances.values())
        models.sort(key=lambda m: m.overall_score, reverse=True)

        best_model = models[0]
        worst_model = models[-1]

        performance_spread = best_model.overall_score - worst_model.overall_score

        # Count significant differences
        significant_differences = len(self.get_significant_tests())

        return {
            "total_models": len(models),
            "best_model": {
                "id": best_model.model_id,
                "name": best_model.model_name,
                "score": str(best_model.overall_score),
            },
            "worst_model": {
                "id": worst_model.model_id,
                "name": worst_model.model_name,
                "score": str(worst_model.overall_score),
            },
            "performance_spread": str(performance_spread),
            "significant_differences": significant_differences,
            "total_statistical_tests": len(self.statistical_tests),
            "actionable_insights": len(self.get_actionable_insights()),
            "high_confidence_insights": len(self.get_high_confidence_insights()),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "analysis_id": str(self.analysis_id),
            "test_id": str(self.test_id),
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "is_completed": self.is_completed(),
            "statistical_tests": {
                name: result.to_dict() for name, result in self.statistical_tests.items()
            },
            "aggregated_data": {
                name: [
                    {
                        "group_key": list(data.group_key),
                        "group_labels": data.group_labels,
                        "sample_count": data.sample_count,
                        "aggregated_value": str(data.aggregated_value),
                        "confidence_interval": (
                            [str(data.confidence_interval[0]), str(data.confidence_interval[1])]
                            if data.confidence_interval
                            else None
                        ),
                        "metadata": data.metadata,
                    }
                    for data in data_list
                ]
                for name, data_list in self.aggregated_data.items()
            },
            "model_performances": {
                model_id: {
                    "model_id": perf.model_id,
                    "model_name": perf.model_name,
                    "overall_score": str(perf.overall_score),
                    "dimension_scores": {
                        dim: str(score) for dim, score in perf.dimension_scores.items()
                    },
                    "sample_count": perf.sample_count,
                    "confidence_score": str(perf.confidence_score),
                    "cost_metrics": perf.cost_metrics.to_dict() if perf.cost_metrics else None,
                    "quality_indicators": perf.quality_indicators,
                }
                for model_id, perf in self.model_performances.items()
            },
            "insights": [insight.to_dict() for insight in self.insights],
            "metadata": self.metadata.copy(),
            "summary": self.get_model_comparison_summary(),
        }

    def __str__(self) -> str:
        """String representation."""
        status = "completed" if self.is_completed() else "in_progress"
        models_count = len(self.model_performances)
        insights_count = len(self.insights)

        return (
            f"AnalysisResult(name='{self.name}', "
            f"status={status}, "
            f"models={models_count}, "
            f"insights={insights_count})"
        )
