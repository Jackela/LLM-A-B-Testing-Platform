"""Insight service for generating actionable insights from analytics data."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from ....domain.analytics.entities.analysis_result import AnalysisResult, ModelPerformanceMetrics
from ....domain.analytics.exceptions import InsightGenerationError, ValidationError
from ....domain.analytics.services.insight_generator import InsightGenerator
from ....domain.analytics.value_objects.insight import Insight, InsightSeverity, InsightType
from ....domain.analytics.value_objects.test_result import TestResult
from .metrics_calculator import CalculatedMetric, MetricType, ModelMetricsSummary

logger = logging.getLogger(__name__)


class InsightCategory(Enum):
    """Categories of insights that can be generated."""

    PERFORMANCE = "performance"
    COST = "cost"
    RELIABILITY = "reliability"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    COMPARISON = "comparison"
    TREND = "trend"
    RECOMMENDATION = "recommendation"


@dataclass
class InsightConfig:
    """Configuration for insight generation."""

    min_confidence_threshold: float = 0.7
    include_statistical_insights: bool = True
    include_cost_insights: bool = True
    include_performance_insights: bool = True
    include_trend_insights: bool = True
    include_recommendations: bool = True
    max_insights_per_category: int = 5
    prioritize_actionable: bool = True


@dataclass
class InsightContext:
    """Context information for insight generation."""

    test_id: UUID
    test_name: str
    total_models: int
    total_samples: int
    analysis_duration_ms: int
    significant_differences: int
    business_context: Optional[Dict[str, Any]] = None


class InsightService:
    """Service for generating comprehensive insights from analytics data."""

    def __init__(self, insight_generator: InsightGenerator):
        self.insight_generator = insight_generator
        self._logger = logger.getChild(self.__class__.__name__)

    async def generate_comprehensive_insights(
        self,
        analysis_result: AnalysisResult,
        model_summaries: Optional[Dict[str, ModelMetricsSummary]] = None,
        config: Optional[InsightConfig] = None,
        context: Optional[InsightContext] = None,
    ) -> List[Insight]:
        """
        Generate comprehensive insights from analysis results.

        Args:
            analysis_result: Complete analysis results
            model_summaries: Optional model metrics summaries
            config: Configuration for insight generation
            context: Additional context for insights

        Returns:
            List of generated insights

        Raises:
            ValidationError: If inputs are invalid
            InsightGenerationError: If insight generation fails
        """
        if config is None:
            config = InsightConfig()

        if context is None:
            context = self._create_default_context(analysis_result)

        try:
            self._logger.info(f"Generating comprehensive insights for test {context.test_id}")

            # Validate inputs
            await self._validate_inputs(analysis_result, config)

            # Generate insights by category
            all_insights = []

            # Performance insights
            if config.include_performance_insights:
                performance_insights = await self._generate_performance_insights(
                    analysis_result, model_summaries, config, context
                )
                all_insights.extend(performance_insights)

            # Cost insights
            if config.include_cost_insights and self._has_cost_data(analysis_result):
                cost_insights = await self._generate_cost_insights(
                    analysis_result, model_summaries, config, context
                )
                all_insights.extend(cost_insights)

            # Statistical insights
            if config.include_statistical_insights:
                statistical_insights = await self._generate_statistical_insights(
                    analysis_result, config, context
                )
                all_insights.extend(statistical_insights)

            # Quality insights
            quality_insights = await self._generate_quality_insights(
                analysis_result, model_summaries, config, context
            )
            all_insights.extend(quality_insights)

            # Comparison insights
            comparison_insights = await self._generate_comparison_insights(
                analysis_result, model_summaries, config, context
            )
            all_insights.extend(comparison_insights)

            # Trend insights
            if config.include_trend_insights:
                trend_insights = await self._generate_trend_insights(
                    analysis_result, model_summaries, config, context
                )
                all_insights.extend(trend_insights)

            # Recommendations
            if config.include_recommendations:
                recommendations = await self._generate_recommendations(
                    analysis_result, model_summaries, config, context
                )
                all_insights.extend(recommendations)

            # Filter and prioritize insights
            filtered_insights = await self._filter_and_prioritize_insights(all_insights, config)

            self._logger.info(
                f"Generated {len(filtered_insights)} insights for test {context.test_id}"
            )
            return filtered_insights

        except Exception as e:
            self._logger.error(f"Insight generation failed: {str(e)}")
            raise InsightGenerationError(f"Failed to generate insights: {str(e)}")

    async def generate_model_specific_insights(
        self,
        model_id: str,
        model_summary: ModelMetricsSummary,
        analysis_result: AnalysisResult,
        config: Optional[InsightConfig] = None,
    ) -> List[Insight]:
        """Generate insights specific to a single model."""
        if config is None:
            config = InsightConfig()

        try:
            insights = []

            # Performance insights for the model
            if float(model_summary.overall_score) < 0.6:
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.PERFORMANCE_ISSUE,
                        title=f"Low Performance Detected for {model_summary.model_name}",
                        description=f"Model {model_summary.model_name} shows below-average performance with an overall score of {float(model_summary.overall_score):.3f}",
                        category=InsightCategory.PERFORMANCE.value,
                        severity=InsightSeverity.HIGH,
                        confidence_score=Decimal("0.9"),
                        affected_models=[model_id],
                        recommendation=f"Consider retraining {model_summary.model_name} with improved data or different hyperparameters",
                        metadata={
                            "overall_score": float(model_summary.overall_score),
                            "quality_grade": model_summary.quality_grade,
                            "ranking_position": model_summary.ranking_position,
                        },
                    )
                )

            # Consistency insights
            if MetricType.CONSISTENCY in model_summary.individual_metrics:
                consistency_metric = model_summary.individual_metrics[MetricType.CONSISTENCY]
                if float(consistency_metric.value) < 0.8:
                    insights.append(
                        Insight(
                            insight_id=uuid4(),
                            insight_type=InsightType.CONSISTENCY_ISSUE,
                            title=f"Inconsistent Performance in {model_summary.model_name}",
                            description=f"Model shows high variance in performance with consistency score of {float(consistency_metric.value):.3f}",
                            category=InsightCategory.RELIABILITY.value,
                            severity=InsightSeverity.MEDIUM,
                            confidence_score=Decimal("0.85"),
                            affected_models=[model_id],
                            recommendation="Consider ensemble methods or regularization techniques to improve consistency",
                        )
                    )

            # Cost efficiency insights
            if model_summary.ranking_position and model_summary.ranking_position == 1:
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.POSITIVE_FINDING,
                        title=f"{model_summary.model_name} is Top Performer",
                        description=f"Model {model_summary.model_name} ranks #1 with excellent performance ({float(model_summary.overall_score):.3f})",
                        category=InsightCategory.PERFORMANCE.value,
                        severity=InsightSeverity.LOW,
                        confidence_score=Decimal("0.95"),
                        affected_models=[model_id],
                        recommendation=f"Consider deploying {model_summary.model_name} as the primary model for production use",
                    )
                )

            return insights

        except Exception as e:
            self._logger.error(f"Model-specific insight generation failed: {str(e)}")
            return []

    async def generate_alert_insights(
        self, analysis_result: AnalysisResult, thresholds: Optional[Dict[str, float]] = None
    ) -> List[Insight]:
        """Generate alert-level insights for critical issues."""
        if thresholds is None:
            thresholds = {
                "min_performance": 0.5,
                "max_error_rate": 0.1,
                "min_reliability": 0.8,
                "max_cost_per_sample": 0.1,
            }

        try:
            alert_insights = []

            # Check for critically low performance
            for model_id, performance in analysis_result.model_performances.items():
                if float(performance.overall_score) < thresholds["min_performance"]:
                    alert_insights.append(
                        Insight(
                            insight_id=uuid4(),
                            insight_type=InsightType.PERFORMANCE_ISSUE,
                            title=f"Critical Performance Issue: {performance.model_name}",
                            description=f"Model performance ({float(performance.overall_score):.3f}) is below critical threshold ({thresholds['min_performance']})",
                            category=InsightCategory.PERFORMANCE.value,
                            severity=InsightSeverity.CRITICAL,
                            confidence_score=Decimal("0.95"),
                            affected_models=[model_id],
                            recommendation="Immediate attention required - consider removing from production",
                            is_actionable=True,
                        )
                    )

            # Check for high cost models
            for model_id, performance in analysis_result.model_performances.items():
                if (
                    performance.cost_metrics
                    and float(performance.cost_metrics.cost_per_sample.amount)
                    > thresholds["max_cost_per_sample"]
                ):

                    alert_insights.append(
                        Insight(
                            insight_id=uuid4(),
                            insight_type=InsightType.COST_ISSUE,
                            title=f"High Cost Alert: {performance.model_name}",
                            description=f"Cost per sample (${float(performance.cost_metrics.cost_per_sample.amount):.4f}) exceeds threshold",
                            category=InsightCategory.COST.value,
                            severity=InsightSeverity.HIGH,
                            confidence_score=Decimal("0.9"),
                            affected_models=[model_id],
                            recommendation="Review cost optimization strategies or consider alternative models",
                            is_actionable=True,
                        )
                    )

            return alert_insights

        except Exception as e:
            self._logger.error(f"Alert insight generation failed: {str(e)}")
            return []

    async def _generate_performance_insights(
        self,
        analysis_result: AnalysisResult,
        model_summaries: Optional[Dict[str, ModelMetricsSummary]],
        config: InsightConfig,
        context: InsightContext,
    ) -> List[Insight]:
        """Generate performance-related insights."""
        insights = []

        try:
            # Best performer insight
            best_model = analysis_result.get_best_performing_model()
            if best_model:
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.POSITIVE_FINDING,
                        title=f"Top Performer Identified: {best_model.model_name}",
                        description=f"{best_model.model_name} achieves the highest performance score of {float(best_model.overall_score):.3f}",
                        category=InsightCategory.PERFORMANCE.value,
                        severity=InsightSeverity.LOW,
                        confidence_score=Decimal("0.95"),
                        affected_models=[best_model.model_id],
                        recommendation=f"Consider {best_model.model_name} for primary deployment",
                        is_actionable=True,
                    )
                )

            # Performance spread analysis
            summary = analysis_result.get_model_comparison_summary()
            spread = float(summary.get("performance_spread", 0))

            if spread > 0.2:  # Significant performance differences
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.STATISTICAL_FINDING,
                        title="Significant Performance Differences Detected",
                        description=f"Large performance spread ({spread:.3f}) indicates substantial model differences",
                        category=InsightCategory.COMPARISON.value,
                        severity=InsightSeverity.MEDIUM,
                        confidence_score=Decimal("0.85"),
                        affected_models=list(analysis_result.model_performances.keys()),
                        recommendation="Focus on understanding what drives performance differences between models",
                    )
                )

            # Low-performing models
            low_performers = [
                model
                for model in analysis_result.model_performances.values()
                if float(model.overall_score) < 0.6
            ]

            if low_performers:
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.PERFORMANCE_ISSUE,
                        title=f"{len(low_performers)} Model(s) Show Poor Performance",
                        description=f"Models with scores below 0.6: {', '.join([m.model_name for m in low_performers])}",
                        category=InsightCategory.PERFORMANCE.value,
                        severity=InsightSeverity.HIGH,
                        confidence_score=Decimal("0.9"),
                        affected_models=[m.model_id for m in low_performers],
                        recommendation="Review and potentially retrain underperforming models",
                        is_actionable=True,
                    )
                )

            return insights[: config.max_insights_per_category]

        except Exception as e:
            self._logger.warning(f"Performance insight generation failed: {str(e)}")
            return []

    async def _generate_cost_insights(
        self,
        analysis_result: AnalysisResult,
        model_summaries: Optional[Dict[str, ModelMetricsSummary]],
        config: InsightConfig,
        context: InsightContext,
    ) -> List[Insight]:
        """Generate cost-related insights."""
        insights = []

        try:
            # Most cost-effective model
            most_cost_effective = analysis_result.get_most_cost_effective_model()
            if most_cost_effective:
                cost_efficiency = float(most_cost_effective.overall_score) / float(
                    most_cost_effective.cost_metrics.cost_per_sample.amount
                )

                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.COST_OPTIMIZATION,
                        title=f"Most Cost-Effective Model: {most_cost_effective.model_name}",
                        description=f"Best cost-performance ratio with efficiency score of {cost_efficiency:.2f}",
                        category=InsightCategory.COST.value,
                        severity=InsightSeverity.LOW,
                        confidence_score=Decimal("0.9"),
                        affected_models=[most_cost_effective.model_id],
                        recommendation=f"Consider {most_cost_effective.model_name} for cost-sensitive deployments",
                        is_actionable=True,
                    )
                )

            # High-cost models
            models_with_cost = [
                model
                for model in analysis_result.model_performances.values()
                if model.cost_metrics is not None
            ]

            if models_with_cost:
                avg_cost = sum(
                    float(m.cost_metrics.cost_per_sample.amount) for m in models_with_cost
                ) / len(models_with_cost)
                high_cost_models = [
                    model
                    for model in models_with_cost
                    if float(model.cost_metrics.cost_per_sample.amount) > avg_cost * 1.5
                ]

                if high_cost_models:
                    insights.append(
                        Insight(
                            insight_id=uuid4(),
                            insight_type=InsightType.COST_ISSUE,
                            title=f"{len(high_cost_models)} High-Cost Model(s) Identified",
                            description=f"Models with above-average costs: {', '.join([m.model_name for m in high_cost_models])}",
                            category=InsightCategory.COST.value,
                            severity=InsightSeverity.MEDIUM,
                            confidence_score=Decimal("0.8"),
                            affected_models=[m.model_id for m in high_cost_models],
                            recommendation="Evaluate if high costs are justified by performance benefits",
                            is_actionable=True,
                        )
                    )

            return insights[: config.max_insights_per_category]

        except Exception as e:
            self._logger.warning(f"Cost insight generation failed: {str(e)}")
            return []

    async def _generate_statistical_insights(
        self, analysis_result: AnalysisResult, config: InsightConfig, context: InsightContext
    ) -> List[Insight]:
        """Generate statistical significance insights."""
        insights = []

        try:
            significant_tests = analysis_result.get_significant_tests()

            if significant_tests:
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.STATISTICAL_FINDING,
                        title=f"{len(significant_tests)} Statistically Significant Difference(s) Found",
                        description=f"Significant differences detected in: {', '.join(significant_tests.keys())}",
                        category=InsightCategory.COMPARISON.value,
                        severity=InsightSeverity.MEDIUM,
                        confidence_score=Decimal("0.95"),
                        affected_models=list(analysis_result.model_performances.keys()),
                        recommendation="Results provide strong evidence for model performance differences",
                    )
                )
            else:
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.STATISTICAL_FINDING,
                        title="No Statistically Significant Differences Found",
                        description="Models show similar performance levels with no significant statistical differences",
                        category=InsightCategory.COMPARISON.value,
                        severity=InsightSeverity.LOW,
                        confidence_score=Decimal("0.8"),
                        affected_models=list(analysis_result.model_performances.keys()),
                        recommendation="Consider other factors like cost or latency for model selection",
                    )
                )

            # Effect size insights
            large_effects = []
            for test_name, test_result in analysis_result.statistical_tests.items():
                if test_result.effect_size and abs(float(test_result.effect_size)) > 0.8:
                    large_effects.append(test_name)

            if large_effects:
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.STATISTICAL_FINDING,
                        title=f"Large Effect Size(s) Detected",
                        description=f"Substantial practical differences found in: {', '.join(large_effects)}",
                        category=InsightCategory.COMPARISON.value,
                        severity=InsightSeverity.MEDIUM,
                        confidence_score=Decimal("0.9"),
                        affected_models=list(analysis_result.model_performances.keys()),
                        recommendation="Large effect sizes indicate meaningful practical differences",
                    )
                )

            return insights[: config.max_insights_per_category]

        except Exception as e:
            self._logger.warning(f"Statistical insight generation failed: {str(e)}")
            return []

    async def _generate_quality_insights(
        self,
        analysis_result: AnalysisResult,
        model_summaries: Optional[Dict[str, ModelMetricsSummary]],
        config: InsightConfig,
        context: InsightContext,
    ) -> List[Insight]:
        """Generate quality-related insights."""
        insights = []

        try:
            # Overall quality assessment
            high_quality_models = [
                model
                for model in analysis_result.model_performances.values()
                if float(model.overall_score) > 0.8
            ]

            if high_quality_models:
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.POSITIVE_FINDING,
                        title=f"{len(high_quality_models)} High-Quality Model(s) Identified",
                        description=f"Models with scores above 0.8: {', '.join([m.model_name for m in high_quality_models])}",
                        category=InsightCategory.QUALITY.value,
                        severity=InsightSeverity.LOW,
                        confidence_score=Decimal("0.9"),
                        affected_models=[m.model_id for m in high_quality_models],
                        recommendation="These models meet high-quality standards for production use",
                    )
                )

            # Quality consistency
            if model_summaries:
                quality_grades = [summary.quality_grade for summary in model_summaries.values()]
                if len(set(quality_grades)) == 1:  # All same grade
                    insights.append(
                        Insight(
                            insight_id=uuid4(),
                            insight_type=InsightType.QUALITY_FINDING,
                            title="Consistent Quality Across Models",
                            description=f"All models achieve similar quality grade: {quality_grades[0]}",
                            category=InsightCategory.QUALITY.value,
                            severity=InsightSeverity.LOW,
                            confidence_score=Decimal("0.85"),
                            affected_models=list(model_summaries.keys()),
                            recommendation="Models show consistent quality levels",
                        )
                    )

            return insights[: config.max_insights_per_category]

        except Exception as e:
            self._logger.warning(f"Quality insight generation failed: {str(e)}")
            return []

    async def _generate_comparison_insights(
        self,
        analysis_result: AnalysisResult,
        model_summaries: Optional[Dict[str, ModelMetricsSummary]],
        config: InsightConfig,
        context: InsightContext,
    ) -> List[Insight]:
        """Generate model comparison insights."""
        insights = []

        try:
            if len(analysis_result.model_performances) < 2:
                return insights

            # Performance ranking insights
            sorted_models = sorted(
                analysis_result.model_performances.values(),
                key=lambda m: m.overall_score,
                reverse=True,
            )

            top_model = sorted_models[0]
            bottom_model = sorted_models[-1]

            performance_gap = float(top_model.overall_score) - float(bottom_model.overall_score)

            insights.append(
                Insight(
                    insight_id=uuid4(),
                    insight_type=InsightType.COMPARISON_FINDING,
                    title=f"Performance Gap: {performance_gap:.3f} Between Best and Worst",
                    description=f"{top_model.model_name} outperforms {bottom_model.model_name} by {performance_gap:.3f} points",
                    category=InsightCategory.COMPARISON.value,
                    severity=(
                        InsightSeverity.MEDIUM if performance_gap > 0.2 else InsightSeverity.LOW
                    ),
                    confidence_score=Decimal("0.9"),
                    affected_models=[top_model.model_id, bottom_model.model_id],
                    recommendation=f"Focus resources on understanding why {top_model.model_name} performs better",
                )
            )

            # Dimension-specific leader insights
            if analysis_result.model_performances:
                dimension_leaders = self._find_dimension_leaders(analysis_result.model_performances)

                for dimension, leader in dimension_leaders.items():
                    insights.append(
                        Insight(
                            insight_id=uuid4(),
                            insight_type=InsightType.COMPARISON_FINDING,
                            title=f"Dimension Leader: {leader['model_name']} in {dimension}",
                            description=f"{leader['model_name']} leads in {dimension} with score {leader['score']:.3f}",
                            category=InsightCategory.COMPARISON.value,
                            severity=InsightSeverity.LOW,
                            confidence_score=Decimal("0.85"),
                            affected_models=[leader["model_id"]],
                            recommendation=f"Consider leveraging {leader['model_name']}'s strength in {dimension}",
                        )
                    )

            return insights[: config.max_insights_per_category]

        except Exception as e:
            self._logger.warning(f"Comparison insight generation failed: {str(e)}")
            return []

    async def _generate_trend_insights(
        self,
        analysis_result: AnalysisResult,
        model_summaries: Optional[Dict[str, ModelMetricsSummary]],
        config: InsightConfig,
        context: InsightContext,
    ) -> List[Insight]:
        """Generate trend-related insights."""
        # This would require historical data - placeholder implementation
        insights = []

        try:
            # Sample trend insight based on available data
            if context.analysis_duration_ms > 60000:  # Long analysis
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.TREND_ANALYSIS,
                        title="Extended Analysis Duration Detected",
                        description=f"Analysis took {context.analysis_duration_ms/1000:.1f} seconds to complete",
                        category=InsightCategory.TREND.value,
                        severity=InsightSeverity.LOW,
                        confidence_score=Decimal("0.8"),
                        affected_models=list(analysis_result.model_performances.keys()),
                        recommendation="Consider optimizing analysis pipeline for faster results",
                    )
                )

            return insights[: config.max_insights_per_category]

        except Exception as e:
            self._logger.warning(f"Trend insight generation failed: {str(e)}")
            return []

    async def _generate_recommendations(
        self,
        analysis_result: AnalysisResult,
        model_summaries: Optional[Dict[str, ModelMetricsSummary]],
        config: InsightConfig,
        context: InsightContext,
    ) -> List[Insight]:
        """Generate actionable recommendations."""
        insights = []

        try:
            # Deployment recommendation
            best_model = analysis_result.get_best_performing_model()
            most_cost_effective = analysis_result.get_most_cost_effective_model()

            if (
                best_model
                and most_cost_effective
                and best_model.model_id != most_cost_effective.model_id
            ):
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.RECOMMENDATION,
                        title="Consider A/B Testing Between Top Performers",
                        description=f"Deploy both {best_model.model_name} (best performance) and {most_cost_effective.model_name} (best cost-efficiency)",
                        category=InsightCategory.RECOMMENDATION.value,
                        severity=InsightSeverity.MEDIUM,
                        confidence_score=Decimal("0.85"),
                        affected_models=[best_model.model_id, most_cost_effective.model_id],
                        recommendation="Run production A/B test to validate real-world performance vs cost trade-offs",
                        is_actionable=True,
                    )
                )

            # Sample size recommendation
            if context.total_samples < 100:
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.RECOMMENDATION,
                        title="Increase Sample Size for Better Statistical Power",
                        description=f"Current sample size ({context.total_samples}) may limit statistical confidence",
                        category=InsightCategory.RECOMMENDATION.value,
                        severity=InsightSeverity.MEDIUM,
                        confidence_score=Decimal("0.8"),
                        affected_models=list(analysis_result.model_performances.keys()),
                        recommendation="Collect additional evaluation samples to improve statistical reliability",
                        is_actionable=True,
                    )
                )

            return insights[: config.max_insights_per_category]

        except Exception as e:
            self._logger.warning(f"Recommendation generation failed: {str(e)}")
            return []

    async def _filter_and_prioritize_insights(
        self, insights: List[Insight], config: InsightConfig
    ) -> List[Insight]:
        """Filter and prioritize insights based on configuration."""

        # Filter by confidence threshold
        filtered_insights = [
            insight
            for insight in insights
            if float(insight.confidence_score) >= config.min_confidence_threshold
        ]

        # Prioritize actionable insights if configured
        if config.prioritize_actionable:
            actionable_insights = [
                insight for insight in filtered_insights if insight.is_actionable
            ]
            non_actionable_insights = [
                insight for insight in filtered_insights if not insight.is_actionable
            ]

            # Sort actionable by severity, then non-actionable
            actionable_insights.sort(key=lambda x: x.severity.value, reverse=True)
            non_actionable_insights.sort(key=lambda x: x.severity.value, reverse=True)

            filtered_insights = actionable_insights + non_actionable_insights
        else:
            # Sort by severity and confidence
            filtered_insights.sort(
                key=lambda x: (x.severity.value, float(x.confidence_score)), reverse=True
            )

        return filtered_insights

    def _create_default_context(self, analysis_result: AnalysisResult) -> InsightContext:
        """Create default context from analysis result."""
        return InsightContext(
            test_id=analysis_result.test_id,
            test_name=analysis_result.name,
            total_models=len(analysis_result.model_performances),
            total_samples=analysis_result.get_total_sample_count(),
            analysis_duration_ms=analysis_result.metadata.get("processing_time_ms", 0),
            significant_differences=len(analysis_result.get_significant_tests()),
        )

    async def _validate_inputs(self, analysis_result: AnalysisResult, config: InsightConfig):
        """Validate inputs for insight generation."""
        if not analysis_result:
            raise ValidationError("Analysis result is required")

        if not analysis_result.is_completed():
            raise ValidationError("Analysis must be completed before generating insights")

        if config.min_confidence_threshold < 0 or config.min_confidence_threshold > 1:
            raise ValidationError("Confidence threshold must be between 0 and 1")

    def _has_cost_data(self, analysis_result: AnalysisResult) -> bool:
        """Check if analysis result contains cost data."""
        return any(
            model.cost_metrics is not None for model in analysis_result.model_performances.values()
        )

    def _find_dimension_leaders(
        self, model_performances: Dict[str, ModelPerformanceMetrics]
    ) -> Dict[str, Dict[str, Any]]:
        """Find the leading model for each dimension."""
        dimension_leaders = {}

        # Collect all dimensions
        all_dimensions = set()
        for performance in model_performances.values():
            all_dimensions.update(performance.dimension_scores.keys())

        # Find leader for each dimension
        for dimension in all_dimensions:
            best_model = None
            best_score = Decimal("-1")

            for model_id, performance in model_performances.items():
                if dimension in performance.dimension_scores:
                    score = performance.dimension_scores[dimension]
                    if score > best_score:
                        best_score = score
                        best_model = performance

            if best_model:
                dimension_leaders[dimension] = {
                    "model_id": best_model.model_id,
                    "model_name": best_model.model_name,
                    "score": float(best_score),
                }

        return dimension_leaders
