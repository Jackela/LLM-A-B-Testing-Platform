"""Performance metrics calculator for comprehensive model evaluation."""

import asyncio
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from ....domain.analytics.entities.analysis_result import (
    AggregatedData,
    AnalysisResult,
    ModelPerformanceMetrics,
)
from ....domain.analytics.exceptions import CalculationError, ValidationError
from ....domain.analytics.repositories.analytics_repository import AnalyticsRepository
from ....domain.analytics.value_objects.cost_data import CostData
from ....domain.analytics.value_objects.performance_score import PerformanceScore
from ....domain.analytics.value_objects.test_result import TestResult
from ....domain.evaluation.entities.evaluation_result import EvaluationResult
from ...dto.performance_metrics_dto import MetricsResponseDTO

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST_EFFICIENCY = "cost_efficiency"
    CONSISTENCY = "consistency"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"


class AggregationMethod(Enum):
    """Methods for aggregating metrics."""

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    WEIGHTED_MEAN = "weighted_mean"
    HARMONIC_MEAN = "harmonic_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    PERCENTILE_95 = "percentile_95"
    PERCENTILE_99 = "percentile_99"


@dataclass
class MetricCalculationConfig:
    """Configuration for metric calculations."""

    include_confidence_intervals: bool = True
    confidence_level: float = 0.95
    outlier_detection: bool = True
    outlier_threshold: float = 2.0  # Standard deviations
    weight_by_sample_size: bool = True
    normalize_scores: bool = True
    include_trend_analysis: bool = True


@dataclass
class CalculatedMetric:
    """A calculated performance metric with metadata."""

    metric_type: MetricType
    value: Decimal
    confidence_interval: Optional[Tuple[Decimal, Decimal]]
    sample_count: int
    calculation_method: AggregationMethod
    metadata: Dict[str, Any]
    calculated_at: datetime
    outliers_detected: int = 0
    trend_direction: Optional[str] = None


@dataclass
class ModelMetricsSummary:
    """Summary of all metrics for a model."""

    model_id: str
    model_name: str
    overall_score: Decimal
    individual_metrics: Dict[MetricType, CalculatedMetric]
    composite_scores: Dict[str, Decimal]
    ranking_position: Optional[int]
    percentile_rank: Optional[Decimal]
    quality_grade: str
    recommendations: List[str]


class MetricsCalculator:
    """Service for calculating comprehensive performance metrics."""

    def __init__(self, analytics_repository: AnalyticsRepository):
        self.analytics_repository = analytics_repository
        self._logger = logger.getChild(self.__class__.__name__)

        # Metric weights for composite scoring
        self._default_metric_weights = {
            MetricType.ACCURACY: 0.25,
            MetricType.PRECISION: 0.15,
            MetricType.RECALL: 0.15,
            MetricType.F1_SCORE: 0.20,
            MetricType.LATENCY: 0.10,
            MetricType.COST_EFFICIENCY: 0.10,
            MetricType.CONSISTENCY: 0.05,
        }

    async def calculate_comprehensive_metrics(
        self, test_id: UUID, config: Optional[MetricCalculationConfig] = None
    ) -> Dict[str, ModelMetricsSummary]:
        """
        Calculate comprehensive performance metrics for all models in a test.

        Args:
            test_id: ID of the test to analyze
            config: Configuration for metric calculations

        Returns:
            Dictionary of model metrics summaries keyed by model ID

        Raises:
            ValidationError: If test_id is invalid
            CalculationError: If metric calculation fails
        """
        if config is None:
            config = MetricCalculationConfig()

        try:
            self._logger.info(f"Calculating comprehensive metrics for test {test_id}")

            # Load evaluation results
            evaluation_results = await self.analytics_repository.get_evaluation_results(test_id)

            if not evaluation_results:
                raise ValidationError(f"No evaluation results found for test {test_id}")

            # Group results by model
            model_results = self._group_results_by_model(evaluation_results)

            # Calculate metrics for each model
            model_summaries = {}
            for model_id, results in model_results.items():
                summary = await self._calculate_model_metrics(model_id, results, config)
                model_summaries[model_id] = summary

            # Calculate rankings and percentiles
            await self._calculate_rankings(model_summaries)

            # Generate recommendations
            await self._generate_metric_recommendations(model_summaries)

            self._logger.info(f"Metrics calculated for {len(model_summaries)} models")
            return model_summaries

        except Exception as e:
            self._logger.error(f"Comprehensive metrics calculation failed: {str(e)}")
            raise CalculationError(f"Failed to calculate metrics: {str(e)}")

    async def calculate_model_performance_score(
        self,
        evaluation_results: List[EvaluationResult],
        weights: Optional[Dict[str, float]] = None,
        config: Optional[MetricCalculationConfig] = None,
    ) -> PerformanceScore:
        """
        Calculate overall performance score for a model.

        Args:
            evaluation_results: List of evaluation results for the model
            weights: Custom weights for different dimensions
            config: Configuration for calculations

        Returns:
            PerformanceScore with detailed breakdown
        """
        if config is None:
            config = MetricCalculationConfig()

        if not evaluation_results:
            raise ValidationError("Evaluation results cannot be empty")

        try:
            # Extract dimension scores
            dimension_scores = self._extract_dimension_scores(evaluation_results)

            # Calculate weighted average if weights provided
            if weights:
                overall_score = self._calculate_weighted_score(dimension_scores, weights)
            else:
                overall_score = self._calculate_simple_average(dimension_scores)

            # Calculate confidence interval
            confidence_interval = None
            if config.include_confidence_intervals:
                confidence_interval = self._calculate_confidence_interval(
                    [float(r.overall_score) for r in evaluation_results], config.confidence_level
                )

            # Calculate consistency score
            consistency_score = self._calculate_consistency_score(evaluation_results)

            return PerformanceScore(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                confidence_interval=confidence_interval,
                sample_count=len(evaluation_results),
                consistency_score=consistency_score,
                metadata={
                    "calculation_method": "weighted_average" if weights else "simple_average",
                    "outliers_removed": 0,  # Would be set by outlier detection
                    "weights_used": weights or {},
                },
            )

        except Exception as e:
            self._logger.error(f"Performance score calculation failed: {str(e)}")
            raise CalculationError(f"Failed to calculate performance score: {str(e)}")

    async def calculate_cost_efficiency_metrics(
        self, model_performances: Dict[str, ModelPerformanceMetrics], cost_weight: float = 0.3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate cost efficiency metrics across models.

        Args:
            model_performances: Dictionary of model performance data
            cost_weight: Weight to assign to cost in efficiency calculation

        Returns:
            Dictionary of cost efficiency metrics by model
        """
        try:
            efficiency_metrics = {}

            models_with_cost = [
                (model_id, perf)
                for model_id, perf in model_performances.items()
                if perf.cost_metrics is not None
            ]

            if not models_with_cost:
                return {}

            for model_id, performance in models_with_cost:
                cost_per_sample = float(performance.cost_metrics.cost_per_sample.amount)
                performance_score = float(performance.overall_score)

                # Calculate different efficiency metrics
                efficiency_ratio = (
                    performance_score / cost_per_sample if cost_per_sample > 0 else float("inf")
                )

                # Cost-adjusted performance score
                cost_penalty = min(1.0, cost_per_sample / 0.01)  # Penalty for high cost
                cost_adjusted_score = performance_score * (1 - cost_weight * cost_penalty)

                # Value score (performance per dollar)
                value_score = (
                    performance_score / cost_per_sample if cost_per_sample > 0 else float("inf")
                )

                efficiency_metrics[model_id] = {
                    "efficiency_ratio": efficiency_ratio,
                    "cost_adjusted_score": cost_adjusted_score,
                    "value_score": value_score,
                    "cost_per_sample": cost_per_sample,
                    "performance_score": performance_score,
                    "cost_percentile": None,  # Will be calculated below
                    "efficiency_rank": None,  # Will be calculated below
                }

            # Calculate percentiles and rankings
            costs = [metrics["cost_per_sample"] for metrics in efficiency_metrics.values()]
            efficiencies = [metrics["efficiency_ratio"] for metrics in efficiency_metrics.values()]

            for model_id, metrics in efficiency_metrics.items():
                cost_percentile = self._calculate_percentile(metrics["cost_per_sample"], costs)
                efficiency_rank = (
                    sorted(efficiencies, reverse=True).index(metrics["efficiency_ratio"]) + 1
                )

                metrics["cost_percentile"] = cost_percentile
                metrics["efficiency_rank"] = efficiency_rank

            return efficiency_metrics

        except Exception as e:
            self._logger.error(f"Cost efficiency calculation failed: {str(e)}")
            raise CalculationError(f"Failed to calculate cost efficiency: {str(e)}")

    async def calculate_trend_metrics(
        self, test_id: UUID, time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Calculate performance trend metrics over time.

        Args:
            test_id: ID of the test
            time_window_hours: Time window for trend analysis

        Returns:
            Dictionary of trend metrics
        """
        try:
            # Load evaluation results with timestamps
            evaluation_results = await self.analytics_repository.get_evaluation_results(test_id)

            if not evaluation_results:
                return {}

            # Filter results within time window
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            recent_results = [
                result
                for result in evaluation_results
                if result.completed_at and result.completed_at >= cutoff_time
            ]

            if len(recent_results) < 2:
                return {"error": "Insufficient data for trend analysis"}

            # Group by model and calculate trends
            model_trends = {}
            model_results = self._group_results_by_model(recent_results)

            for model_id, results in model_results.items():
                trend = self._calculate_model_trend(results)
                model_trends[model_id] = trend

            # Calculate overall trends
            overall_trend = self._calculate_overall_trend(recent_results)

            return {
                "overall_trend": overall_trend,
                "model_trends": model_trends,
                "time_window_hours": time_window_hours,
                "total_evaluations": len(recent_results),
                "trend_calculation_time": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self._logger.error(f"Trend metrics calculation failed: {str(e)}")
            raise CalculationError(f"Failed to calculate trend metrics: {str(e)}")

    async def calculate_reliability_metrics(
        self, evaluation_results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Calculate reliability and consistency metrics.

        Args:
            evaluation_results: List of evaluation results

        Returns:
            Dictionary of reliability metrics
        """
        try:
            if not evaluation_results:
                return {}

            # Calculate completion rate
            completed_count = len([r for r in evaluation_results if r.is_completed()])
            completion_rate = completed_count / len(evaluation_results)

            # Calculate error rate
            error_count = len([r for r in evaluation_results if r.has_error()])
            error_rate = error_count / len(evaluation_results)

            # Calculate score consistency (coefficient of variation)
            scores = [float(r.overall_score) for r in evaluation_results if r.is_completed()]

            if len(scores) > 1:
                mean_score = statistics.mean(scores)
                score_std = statistics.stdev(scores)
                coefficient_of_variation = score_std / mean_score if mean_score > 0 else 0
            else:
                coefficient_of_variation = 0

            # Calculate response time consistency
            response_times = []
            for result in evaluation_results:
                if result.metadata.get("response_time_ms"):
                    response_times.append(result.metadata["response_time_ms"])

            response_time_cv = 0
            if len(response_times) > 1:
                mean_time = statistics.mean(response_times)
                time_std = statistics.stdev(response_times)
                response_time_cv = time_std / mean_time if mean_time > 0 else 0

            # Calculate reliability score (0-1)
            reliability_score = (
                completion_rate * 0.4  # 40% weight to completion rate
                + (1 - error_rate) * 0.3  # 30% weight to error rate
                + (1 - min(1, coefficient_of_variation)) * 0.2  # 20% weight to score consistency
                + (1 - min(1, response_time_cv)) * 0.1  # 10% weight to time consistency
            )

            return {
                "completion_rate": completion_rate,
                "error_rate": error_rate,
                "score_coefficient_of_variation": coefficient_of_variation,
                "response_time_coefficient_of_variation": response_time_cv,
                "reliability_score": reliability_score,
                "total_evaluations": len(evaluation_results),
                "completed_evaluations": completed_count,
                "failed_evaluations": error_count,
                "consistency_grade": self._grade_consistency(coefficient_of_variation),
            }

        except Exception as e:
            self._logger.error(f"Reliability metrics calculation failed: {str(e)}")
            raise CalculationError(f"Failed to calculate reliability metrics: {str(e)}")

    def _group_results_by_model(
        self, evaluation_results: List[EvaluationResult]
    ) -> Dict[str, List[EvaluationResult]]:
        """Group evaluation results by model ID."""
        model_results = {}
        for result in evaluation_results:
            model_id = result.model_id
            if model_id not in model_results:
                model_results[model_id] = []
            model_results[model_id].append(result)
        return model_results

    async def _calculate_model_metrics(
        self, model_id: str, results: List[EvaluationResult], config: MetricCalculationConfig
    ) -> ModelMetricsSummary:
        """Calculate comprehensive metrics for a single model."""

        # Extract basic information
        model_name = results[0].metadata.get("model_name", model_id)

        # Calculate individual metrics
        individual_metrics = {}

        # Accuracy-based metrics
        if self._has_dimension_scores(results):
            individual_metrics.update(await self._calculate_accuracy_metrics(results, config))

        # Performance metrics
        individual_metrics.update(await self._calculate_performance_metrics(results, config))

        # Reliability metrics
        reliability_data = await self.calculate_reliability_metrics(results)
        if reliability_data:
            individual_metrics[MetricType.RELIABILITY] = CalculatedMetric(
                metric_type=MetricType.RELIABILITY,
                value=Decimal(str(reliability_data["reliability_score"])),
                confidence_interval=None,
                sample_count=len(results),
                calculation_method=AggregationMethod.WEIGHTED_MEAN,
                metadata=reliability_data,
                calculated_at=datetime.utcnow(),
            )

        # Calculate overall score
        overall_score = self._calculate_composite_score(individual_metrics)

        # Calculate composite scores
        composite_scores = {
            "quality_score": self._calculate_quality_score(individual_metrics),
            "efficiency_score": self._calculate_efficiency_score(individual_metrics),
            "reliability_score": individual_metrics.get(
                MetricType.RELIABILITY,
                CalculatedMetric(
                    MetricType.RELIABILITY,
                    Decimal("0"),
                    None,
                    0,
                    AggregationMethod.MEAN,
                    {},
                    datetime.utcnow(),
                ),
            ).value,
        }

        # Assign quality grade
        quality_grade = self._assign_quality_grade(overall_score)

        return ModelMetricsSummary(
            model_id=model_id,
            model_name=model_name,
            overall_score=overall_score,
            individual_metrics=individual_metrics,
            composite_scores=composite_scores,
            ranking_position=None,  # Will be set later
            percentile_rank=None,  # Will be set later
            quality_grade=quality_grade,
            recommendations=[],  # Will be populated later
        )

    async def _calculate_accuracy_metrics(
        self, results: List[EvaluationResult], config: MetricCalculationConfig
    ) -> Dict[MetricType, CalculatedMetric]:
        """Calculate accuracy-based metrics."""
        metrics = {}

        # Extract overall scores
        scores = [float(r.overall_score) for r in results if r.is_completed()]

        if scores:
            # Remove outliers if configured
            if config.outlier_detection:
                cleaned_scores, outliers_count = self._remove_outliers(
                    scores, config.outlier_threshold
                )
            else:
                cleaned_scores, outliers_count = scores, 0

            # Calculate accuracy metric
            mean_accuracy = statistics.mean(cleaned_scores)

            confidence_interval = None
            if config.include_confidence_intervals and len(cleaned_scores) > 1:
                confidence_interval = self._calculate_confidence_interval(
                    cleaned_scores, config.confidence_level
                )

            metrics[MetricType.ACCURACY] = CalculatedMetric(
                metric_type=MetricType.ACCURACY,
                value=Decimal(str(mean_accuracy)),
                confidence_interval=confidence_interval,
                sample_count=len(cleaned_scores),
                calculation_method=AggregationMethod.MEAN,
                metadata={
                    "original_sample_count": len(scores),
                    "std_deviation": (
                        statistics.stdev(cleaned_scores) if len(cleaned_scores) > 1 else 0
                    ),
                },
                calculated_at=datetime.utcnow(),
                outliers_detected=outliers_count,
            )

        return metrics

    async def _calculate_performance_metrics(
        self, results: List[EvaluationResult], config: MetricCalculationConfig
    ) -> Dict[MetricType, CalculatedMetric]:
        """Calculate performance-related metrics."""
        metrics = {}

        # Extract response times
        response_times = []
        for result in results:
            if result.metadata.get("response_time_ms"):
                response_times.append(result.metadata["response_time_ms"])

        if response_times:
            # Calculate latency metric
            mean_latency = statistics.mean(response_times)

            metrics[MetricType.LATENCY] = CalculatedMetric(
                metric_type=MetricType.LATENCY,
                value=Decimal(str(mean_latency)),
                confidence_interval=None,
                sample_count=len(response_times),
                calculation_method=AggregationMethod.MEAN,
                metadata={
                    "min_latency": min(response_times),
                    "max_latency": max(response_times),
                    "median_latency": statistics.median(response_times),
                    "p95_latency": self._calculate_percentile_value(response_times, 95),
                },
                calculated_at=datetime.utcnow(),
            )

        # Calculate consistency metric
        scores = [float(r.overall_score) for r in results if r.is_completed()]
        if len(scores) > 1:
            cv = (
                statistics.stdev(scores) / statistics.mean(scores)
                if statistics.mean(scores) > 0
                else 0
            )
            consistency_score = max(0, 1 - cv)  # Higher consistency = lower CV

            metrics[MetricType.CONSISTENCY] = CalculatedMetric(
                metric_type=MetricType.CONSISTENCY,
                value=Decimal(str(consistency_score)),
                confidence_interval=None,
                sample_count=len(scores),
                calculation_method=AggregationMethod.MEAN,
                metadata={"coefficient_of_variation": cv, "score_range": max(scores) - min(scores)},
                calculated_at=datetime.utcnow(),
            )

        return metrics

    def _calculate_composite_score(
        self, individual_metrics: Dict[MetricType, CalculatedMetric]
    ) -> Decimal:
        """Calculate composite score from individual metrics."""

        weighted_sum = Decimal("0")
        total_weight = Decimal("0")

        for metric_type, weight in self._default_metric_weights.items():
            if metric_type in individual_metrics:
                weighted_sum += individual_metrics[metric_type].value * Decimal(str(weight))
                total_weight += Decimal(str(weight))

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return Decimal("0")

    def _calculate_quality_score(
        self, individual_metrics: Dict[MetricType, CalculatedMetric]
    ) -> Decimal:
        """Calculate quality-focused composite score."""
        quality_weights = {
            MetricType.ACCURACY: 0.4,
            MetricType.PRECISION: 0.25,
            MetricType.RECALL: 0.25,
            MetricType.CONSISTENCY: 0.1,
        }

        weighted_sum = Decimal("0")
        total_weight = Decimal("0")

        for metric_type, weight in quality_weights.items():
            if metric_type in individual_metrics:
                weighted_sum += individual_metrics[metric_type].value * Decimal(str(weight))
                total_weight += Decimal(str(weight))

        return weighted_sum / total_weight if total_weight > 0 else Decimal("0")

    def _calculate_efficiency_score(
        self, individual_metrics: Dict[MetricType, CalculatedMetric]
    ) -> Decimal:
        """Calculate efficiency-focused composite score."""
        efficiency_weights = {
            MetricType.LATENCY: 0.4,  # Lower is better
            MetricType.COST_EFFICIENCY: 0.4,
            MetricType.THROUGHPUT: 0.2,
        }

        weighted_sum = Decimal("0")
        total_weight = Decimal("0")

        for metric_type, weight in efficiency_weights.items():
            if metric_type in individual_metrics:
                metric_value = individual_metrics[metric_type].value

                # For latency, invert the score (lower latency = higher efficiency)
                if metric_type == MetricType.LATENCY:
                    # Normalize latency to 0-1 scale and invert
                    normalized_value = max(
                        Decimal("0"), Decimal("1") - (metric_value / Decimal("1000"))
                    )
                    weighted_sum += normalized_value * Decimal(str(weight))
                else:
                    weighted_sum += metric_value * Decimal(str(weight))

                total_weight += Decimal(str(weight))

        return weighted_sum / total_weight if total_weight > 0 else Decimal("0")

    def _assign_quality_grade(self, overall_score: Decimal) -> str:
        """Assign quality grade based on overall score."""
        score = float(overall_score)

        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.5:
            return "C"
        else:
            return "D"

    async def _calculate_rankings(self, model_summaries: Dict[str, ModelMetricsSummary]):
        """Calculate rankings and percentiles for all models."""

        # Sort models by overall score
        sorted_models = sorted(
            model_summaries.values(), key=lambda m: m.overall_score, reverse=True
        )

        # Assign rankings and percentiles
        total_models = len(sorted_models)

        for rank, model in enumerate(sorted_models, 1):
            model.ranking_position = rank
            model.percentile_rank = Decimal(str((total_models - rank + 1) / total_models * 100))

    async def _generate_metric_recommendations(
        self, model_summaries: Dict[str, ModelMetricsSummary]
    ):
        """Generate recommendations for each model based on metrics."""

        for model_id, summary in model_summaries.items():
            recommendations = []

            # Check individual metrics for recommendations
            for metric_type, metric in summary.individual_metrics.items():

                if metric_type == MetricType.ACCURACY and float(metric.value) < 0.7:
                    recommendations.append(
                        "Consider improving model accuracy through better training data or hyperparameter tuning"
                    )

                if metric_type == MetricType.CONSISTENCY and float(metric.value) < 0.8:
                    recommendations.append(
                        "Model shows inconsistent performance - consider ensemble methods or better regularization"
                    )

                if metric_type == MetricType.LATENCY and float(metric.value) > 1000:  # >1 second
                    recommendations.append(
                        "High latency detected - consider model optimization or caching strategies"
                    )

                if metric_type == MetricType.RELIABILITY and float(metric.value) < 0.9:
                    recommendations.append(
                        "Reliability concerns detected - investigate error patterns and improve error handling"
                    )

            # Overall performance recommendations
            if float(summary.overall_score) < 0.6:
                recommendations.append(
                    "Overall performance is below acceptable threshold - comprehensive review recommended"
                )

            if summary.ranking_position and summary.ranking_position > len(model_summaries) * 0.8:
                recommendations.append(
                    "Performance lags behind other models - consider alternative approaches"
                )

            summary.recommendations = recommendations

    def _extract_dimension_scores(
        self, evaluation_results: List[EvaluationResult]
    ) -> Dict[str, Decimal]:
        """Extract and average dimension scores."""
        dimension_totals = {}
        dimension_counts = {}

        for result in evaluation_results:
            if result.is_completed():
                for dimension, score in result.dimension_scores.items():
                    if dimension not in dimension_totals:
                        dimension_totals[dimension] = Decimal("0")
                        dimension_counts[dimension] = 0

                    dimension_totals[dimension] += score
                    dimension_counts[dimension] += 1

        # Calculate averages
        dimension_averages = {}
        for dimension in dimension_totals:
            if dimension_counts[dimension] > 0:
                dimension_averages[dimension] = (
                    dimension_totals[dimension] / dimension_counts[dimension]
                )

        return dimension_averages

    def _calculate_weighted_score(
        self, dimension_scores: Dict[str, Decimal], weights: Dict[str, float]
    ) -> Decimal:
        """Calculate weighted average score."""
        weighted_sum = Decimal("0")
        total_weight = Decimal("0")

        for dimension, score in dimension_scores.items():
            weight = weights.get(dimension, 0)
            if weight > 0:
                weighted_sum += score * Decimal(str(weight))
                total_weight += Decimal(str(weight))

        return weighted_sum / total_weight if total_weight > 0 else Decimal("0")

    def _calculate_simple_average(self, dimension_scores: Dict[str, Decimal]) -> Decimal:
        """Calculate simple average of dimension scores."""
        if not dimension_scores:
            return Decimal("0")

        total = sum(dimension_scores.values())
        return total / len(dimension_scores)

    def _calculate_confidence_interval(
        self, values: List[float], confidence_level: float
    ) -> Tuple[Decimal, Decimal]:
        """Calculate confidence interval for a list of values."""
        if len(values) < 2:
            mean_val = values[0] if values else 0
            return (Decimal(str(mean_val)), Decimal(str(mean_val)))

        mean = statistics.mean(values)
        std_error = statistics.stdev(values) / math.sqrt(len(values))

        # Use t-distribution for small samples
        from scipy.stats import t

        alpha = 1 - confidence_level
        t_critical = t.ppf(1 - alpha / 2, len(values) - 1)

        margin_of_error = t_critical * std_error

        lower = mean - margin_of_error
        upper = mean + margin_of_error

        return (Decimal(str(lower)), Decimal(str(upper)))

    def _calculate_consistency_score(self, evaluation_results: List[EvaluationResult]) -> Decimal:
        """Calculate consistency score based on score variance."""
        scores = [float(r.overall_score) for r in evaluation_results if r.is_completed()]

        if len(scores) < 2:
            return Decimal("1")  # Perfect consistency for single data point

        mean_score = statistics.mean(scores)
        cv = statistics.stdev(scores) / mean_score if mean_score > 0 else 0

        # Convert CV to consistency score (0-1, higher is better)
        consistency = max(0, 1 - cv)
        return Decimal(str(consistency))

    def _remove_outliers(self, values: List[float], threshold: float) -> Tuple[List[float], int]:
        """Remove outliers using z-score method."""
        if len(values) < 3:
            return values, 0

        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)

        if std_dev == 0:
            return values, 0

        cleaned_values = []
        outliers_count = 0

        for value in values:
            z_score = abs((value - mean) / std_dev)
            if z_score <= threshold:
                cleaned_values.append(value)
            else:
                outliers_count += 1

        return cleaned_values, outliers_count

    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calculate percentile rank of a value in a list."""
        if not values:
            return 0

        sorted_values = sorted(values)
        rank = sorted_values.index(value) if value in sorted_values else 0

        return (rank / len(sorted_values)) * 100

    def _calculate_percentile_value(self, values: List[float], percentile: float) -> float:
        """Calculate the value at a given percentile."""
        if not values:
            return 0

        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))

        if index >= len(sorted_values):
            return sorted_values[-1]

        return sorted_values[index]

    def _has_dimension_scores(self, results: List[EvaluationResult]) -> bool:
        """Check if results contain dimension scores."""
        return any(result.dimension_scores for result in results)

    def _calculate_model_trend(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate trend metrics for a model."""
        # Sort by completion time
        sorted_results = sorted(
            [r for r in results if r.completed_at], key=lambda r: r.completed_at
        )

        if len(sorted_results) < 2:
            return {"trend": "insufficient_data"}

        # Calculate trend in scores
        scores = [float(r.overall_score) for r in sorted_results]
        times = [
            (r.completed_at - sorted_results[0].completed_at).total_seconds()
            for r in sorted_results
        ]

        # Simple linear regression for trend
        n = len(scores)
        sum_x = sum(times)
        sum_y = sum(scores)
        sum_xy = sum(x * y for x, y in zip(times, scores))
        sum_x2 = sum(x * x for x in times)

        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        else:
            slope = 0

        trend_direction = (
            "improving" if slope > 0.001 else "declining" if slope < -0.001 else "stable"
        )

        return {
            "trend": trend_direction,
            "slope": slope,
            "data_points": n,
            "score_range": (min(scores), max(scores)),
            "latest_score": scores[-1],
            "score_change": scores[-1] - scores[0] if len(scores) > 1 else 0,
        }

    def _calculate_overall_trend(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate overall trend across all models."""
        # Group by time periods and calculate average scores
        time_periods = {}

        for result in results:
            if result.completed_at and result.is_completed():
                # Group by hour
                hour_key = result.completed_at.replace(minute=0, second=0, microsecond=0)
                if hour_key not in time_periods:
                    time_periods[hour_key] = []
                time_periods[hour_key].append(float(result.overall_score))

        if len(time_periods) < 2:
            return {"trend": "insufficient_data"}

        # Calculate average score per time period
        sorted_periods = sorted(time_periods.keys())
        period_averages = [statistics.mean(time_periods[period]) for period in sorted_periods]

        # Calculate trend
        first_half = period_averages[: len(period_averages) // 2]
        second_half = period_averages[len(period_averages) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        trend_direction = (
            "improving"
            if second_avg > first_avg
            else "declining" if second_avg < first_avg else "stable"
        )

        return {
            "trend": trend_direction,
            "first_half_average": first_avg,
            "second_half_average": second_avg,
            "change": second_avg - first_avg,
            "time_periods_analyzed": len(time_periods),
        }

    def _grade_consistency(self, coefficient_of_variation: float) -> str:
        """Grade consistency based on coefficient of variation."""
        if coefficient_of_variation < 0.05:
            return "Excellent"
        elif coefficient_of_variation < 0.1:
            return "Good"
        elif coefficient_of_variation < 0.2:
            return "Fair"
        else:
            return "Poor"
