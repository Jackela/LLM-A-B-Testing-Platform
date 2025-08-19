"""Domain-model mapper for Analytics domain."""

from typing import Any, Dict, List
from uuid import UUID

from .....domain.analytics.entities.analysis_result import AnalysisResult
from .....domain.analytics.entities.model_performance import ModelPerformance
from .....domain.analytics.entities.statistical_test import StatisticalTest
from .....domain.analytics.value_objects.insight import Insight
from ...models.analytics_models import (
    AnalysisResultModel,
    InsightModel,
    ModelPerformanceModel,
    StatisticalTestModel,
)


class AnalyticsMapper:
    """Mapper between Analytics domain entities and database models."""

    def analysis_result_to_model(self, result: AnalysisResult) -> AnalysisResultModel:
        """Convert AnalysisResult domain entity to database model."""
        result_model = AnalysisResultModel(
            id=result.id,
            test_id=result.test_id,
            analysis_type=result.analysis_type,
            status=result.status,
            configuration=result.configuration.copy(),
            parameters=result.parameters.copy() if result.parameters else {},
            summary=result.summary.copy() if result.summary else {},
            detailed_results=result.detailed_results.copy() if result.detailed_results else {},
            statistical_significance=result.statistical_significance,
            confidence_level=result.confidence_level,
            effect_size=result.effect_size,
            error_message=result.error_message,
            error_details=result.error_details.copy() if result.error_details else {},
            execution_time_ms=result.execution_time_ms,
            data_points_analyzed=result.data_points_analyzed,
            created_at=result.created_at,
            started_at=result.started_at,
            completed_at=result.completed_at,
            metadata=result.metadata.copy() if result.metadata else {},
        )

        # Convert model performances if they exist
        if hasattr(result, "model_performances") and result.model_performances:
            result_model.model_performances = [
                self.model_performance_to_model(perf) for perf in result.model_performances
            ]

        # Convert insights if they exist
        if hasattr(result, "insights") and result.insights:
            result_model.insights = [self.insight_to_model(insight) for insight in result.insights]

        return result_model

    def model_to_analysis_result(self, result_model: AnalysisResultModel) -> AnalysisResult:
        """Convert database model to AnalysisResult domain entity."""
        result = AnalysisResult(
            id=result_model.id,
            test_id=result_model.test_id,
            analysis_type=result_model.analysis_type,
            status=result_model.status,
            configuration=result_model.configuration.copy(),
            parameters=result_model.parameters.copy() if result_model.parameters else {},
            summary=result_model.summary.copy() if result_model.summary else {},
            detailed_results=(
                result_model.detailed_results.copy() if result_model.detailed_results else {}
            ),
            statistical_significance=result_model.statistical_significance,
            confidence_level=result_model.confidence_level,
            effect_size=result_model.effect_size,
            error_message=result_model.error_message,
            error_details=result_model.error_details.copy() if result_model.error_details else {},
            execution_time_ms=result_model.execution_time_ms,
            data_points_analyzed=result_model.data_points_analyzed,
            created_at=result_model.created_at,
            started_at=result_model.started_at,
            completed_at=result_model.completed_at,
            metadata=result_model.metadata.copy() if result_model.metadata else {},
        )

        return result

    def model_performance_to_model(self, performance: ModelPerformance) -> ModelPerformanceModel:
        """Convert ModelPerformance domain entity to database model."""
        return ModelPerformanceModel(
            id=performance.id,
            analysis_result_id=performance.analysis_result_id,
            model_id=performance.model_id,
            provider_name=performance.provider_name,
            overall_score=performance.overall_score,
            dimension_scores=performance.dimension_scores.copy(),
            sample_count=performance.sample_count,
            success_rate=performance.success_rate,
            mean_score=performance.mean_score,
            median_score=performance.median_score,
            std_deviation=performance.std_deviation,
            confidence_interval=(
                performance.confidence_interval.copy() if performance.confidence_interval else {}
            ),
            percentiles=performance.percentiles.copy() if performance.percentiles else {},
            total_cost=performance.total_cost,
            average_latency_ms=performance.average_latency_ms,
            input_tokens_total=performance.input_tokens_total,
            output_tokens_total=performance.output_tokens_total,
            cost_per_sample=performance.cost_per_sample,
            error_rate=performance.error_rate,
            timeout_rate=performance.timeout_rate,
            quality_score=performance.quality_score,
            rank=performance.rank,
            rank_percentile=performance.rank_percentile,
            relative_performance=(
                performance.relative_performance.copy() if performance.relative_performance else {}
            ),
            calculated_at=performance.calculated_at,
            metadata=performance.metadata.copy() if performance.metadata else {},
        )

    def model_to_model_performance(self, perf_model: ModelPerformanceModel) -> ModelPerformance:
        """Convert database model to ModelPerformance domain entity."""
        return ModelPerformance(
            id=perf_model.id,
            analysis_result_id=perf_model.analysis_result_id,
            model_id=perf_model.model_id,
            provider_name=perf_model.provider_name,
            overall_score=perf_model.overall_score,
            dimension_scores=perf_model.dimension_scores.copy(),
            sample_count=perf_model.sample_count,
            success_rate=perf_model.success_rate,
            mean_score=perf_model.mean_score,
            median_score=perf_model.median_score,
            std_deviation=perf_model.std_deviation,
            confidence_interval=(
                perf_model.confidence_interval.copy() if perf_model.confidence_interval else {}
            ),
            percentiles=perf_model.percentiles.copy() if perf_model.percentiles else {},
            total_cost=perf_model.total_cost,
            average_latency_ms=perf_model.average_latency_ms,
            input_tokens_total=perf_model.input_tokens_total,
            output_tokens_total=perf_model.output_tokens_total,
            cost_per_sample=perf_model.cost_per_sample,
            error_rate=perf_model.error_rate,
            timeout_rate=perf_model.timeout_rate,
            quality_score=perf_model.quality_score,
            rank=perf_model.rank,
            rank_percentile=perf_model.rank_percentile,
            relative_performance=(
                perf_model.relative_performance.copy() if perf_model.relative_performance else {}
            ),
            calculated_at=perf_model.calculated_at,
            metadata=perf_model.metadata.copy() if perf_model.metadata else {},
        )

    def statistical_test_to_model(self, stat_test: StatisticalTest) -> StatisticalTestModel:
        """Convert StatisticalTest domain entity to database model."""
        return StatisticalTestModel(
            id=stat_test.id,
            analysis_result_id=stat_test.analysis_result_id,
            test_name=stat_test.test_name,
            test_type=stat_test.test_type,
            null_hypothesis=stat_test.null_hypothesis,
            alternative_hypothesis=stat_test.alternative_hypothesis,
            significance_level=stat_test.significance_level,
            power=stat_test.power,
            effect_size_threshold=stat_test.effect_size_threshold,
            sample_sizes=stat_test.sample_sizes.copy(),
            group_data=stat_test.group_data.copy(),
            assumptions_checked=(
                stat_test.assumptions_checked.copy() if stat_test.assumptions_checked else {}
            ),
            test_statistic=stat_test.test_statistic,
            p_value=stat_test.p_value,
            effect_size=stat_test.effect_size,
            confidence_interval=(
                stat_test.confidence_interval.copy() if stat_test.confidence_interval else {}
            ),
            is_significant=stat_test.is_significant,
            degrees_of_freedom=stat_test.degrees_of_freedom,
            critical_value=stat_test.critical_value,
            observed_power=stat_test.observed_power,
            assumptions_met=stat_test.assumptions_met,
            warnings=stat_test.warnings.copy() if stat_test.warnings else {},
            recommendations=stat_test.recommendations.copy() if stat_test.recommendations else {},
            executed_at=stat_test.executed_at,
            metadata=stat_test.metadata.copy() if stat_test.metadata else {},
        )

    def model_to_statistical_test(self, stat_model: StatisticalTestModel) -> StatisticalTest:
        """Convert database model to StatisticalTest domain entity."""
        return StatisticalTest(
            id=stat_model.id,
            analysis_result_id=stat_model.analysis_result_id,
            test_name=stat_model.test_name,
            test_type=stat_model.test_type,
            null_hypothesis=stat_model.null_hypothesis,
            alternative_hypothesis=stat_model.alternative_hypothesis,
            significance_level=stat_model.significance_level,
            power=stat_model.power,
            effect_size_threshold=stat_model.effect_size_threshold,
            sample_sizes=stat_model.sample_sizes.copy(),
            group_data=stat_model.group_data.copy(),
            assumptions_checked=(
                stat_model.assumptions_checked.copy() if stat_model.assumptions_checked else {}
            ),
            test_statistic=stat_model.test_statistic,
            p_value=stat_model.p_value,
            effect_size=stat_model.effect_size,
            confidence_interval=(
                stat_model.confidence_interval.copy() if stat_model.confidence_interval else {}
            ),
            is_significant=stat_model.is_significant,
            degrees_of_freedom=stat_model.degrees_of_freedom,
            critical_value=stat_model.critical_value,
            observed_power=stat_model.observed_power,
            assumptions_met=stat_model.assumptions_met,
            warnings=stat_model.warnings.copy() if stat_model.warnings else {},
            recommendations=stat_model.recommendations.copy() if stat_model.recommendations else {},
            executed_at=stat_model.executed_at,
            metadata=stat_model.metadata.copy() if stat_model.metadata else {},
        )

    def insight_to_model(self, insight: Insight) -> InsightModel:
        """Convert Insight domain entity to database model."""
        return InsightModel(
            id=insight.id,
            analysis_result_id=insight.analysis_result_id,
            insight_type=insight.insight_type,
            category=insight.category,
            severity=insight.severity,
            title=insight.title,
            description=insight.description,
            recommendation=insight.recommendation,
            supporting_data=insight.supporting_data.copy() if insight.supporting_data else {},
            evidence_strength=insight.evidence_strength,
            confidence_score=insight.confidence_score,
            potential_impact=insight.potential_impact,
            implementation_difficulty=insight.implementation_difficulty,
            priority_score=insight.priority_score,
            affected_models=insight.affected_models.copy() if insight.affected_models else [],
            affected_dimensions=(
                insight.affected_dimensions.copy() if insight.affected_dimensions else []
            ),
            context_tags=insight.context_tags.copy() if insight.context_tags else [],
            generated_at=insight.generated_at,
            expires_at=insight.expires_at,
            metadata=insight.metadata.copy() if insight.metadata else {},
        )

    def model_to_insight(self, insight_model: InsightModel) -> Insight:
        """Convert database model to Insight domain entity."""
        return Insight(
            id=insight_model.id,
            analysis_result_id=insight_model.analysis_result_id,
            insight_type=insight_model.insight_type,
            category=insight_model.category,
            severity=insight_model.severity,
            title=insight_model.title,
            description=insight_model.description,
            recommendation=insight_model.recommendation,
            supporting_data=(
                insight_model.supporting_data.copy() if insight_model.supporting_data else {}
            ),
            evidence_strength=insight_model.evidence_strength,
            confidence_score=insight_model.confidence_score,
            potential_impact=insight_model.potential_impact,
            implementation_difficulty=insight_model.implementation_difficulty,
            priority_score=insight_model.priority_score,
            affected_models=(
                insight_model.affected_models.copy() if insight_model.affected_models else []
            ),
            affected_dimensions=(
                insight_model.affected_dimensions.copy()
                if insight_model.affected_dimensions
                else []
            ),
            context_tags=insight_model.context_tags.copy() if insight_model.context_tags else [],
            generated_at=insight_model.generated_at,
            expires_at=insight_model.expires_at,
            metadata=insight_model.metadata.copy() if insight_model.metadata else {},
        )
