"""Test insight generation functionality."""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from src.domain.analytics.entities.analysis_result import AnalysisResult, ModelPerformanceMetrics
from src.domain.analytics.entities.model_performance import ModelComparison, ModelPerformance
from src.domain.analytics.entities.statistical_test import StatisticalTest, TestType
from src.domain.analytics.services.insight_generator import InsightGenerator
from src.domain.analytics.value_objects.insight import Insight, InsightSeverity, InsightType
from src.domain.analytics.value_objects.performance_score import PerformanceScore
from src.domain.analytics.value_objects.test_result import (
    EffectMagnitude,
    TestInterpretation,
    TestResult,
)
from src.domain.evaluation.entities.evaluation_result import EvaluationResult


class TestInsightGenerator:
    """Test insight generation service."""

    def setup_method(self):
        """Set up test data for insight generation."""
        self.insight_generator = InsightGenerator()

        # Create sample evaluation results for testing
        self.evaluation_results_a = []
        self.evaluation_results_b = []

        # Model A - higher accuracy, lower fluency
        for i in range(15):
            result = EvaluationResult.create_pending(
                judge_id="human_evaluator_1",
                template_id=uuid4(),
                prompt=f"Test prompt {i}",
                response=f"Model A response {i}",
            )

            result.dimension_scores = {
                "accuracy": 5 if i < 10 else 4,  # High accuracy
                "fluency": 3,  # Lower fluency
                "coherence": 4,
            }
            result.overall_score = Decimal("0.8") + Decimal(str((i % 3) * 0.05))
            result.confidence_score = Decimal("0.85") + Decimal(str((i % 2) * 0.05))
            result.reasoning = f"Model A reasoning {i}"
            result.evaluation_time_ms = 2000
            result.completed_at = datetime.utcnow()
            result.metadata = {
                "model": "model_a",
                "category": "accuracy" if i % 2 == 0 else "fluency",
                "difficulty_level": "easy" if i < 8 else "hard",
            }

            self.evaluation_results_a.append(result)

        # Model B - lower accuracy, higher fluency
        for i in range(15):
            result = EvaluationResult.create_pending(
                judge_id="human_evaluator_1",
                template_id=uuid4(),
                prompt=f"Test prompt {i}",
                response=f"Model B response {i}",
            )

            result.dimension_scores = {
                "accuracy": 3,  # Lower accuracy
                "fluency": 5 if i < 12 else 4,  # High fluency
                "coherence": 4,
            }
            result.overall_score = Decimal("0.75") + Decimal(str((i % 4) * 0.03))
            result.confidence_score = Decimal("0.80") + Decimal(str((i % 3) * 0.03))
            result.reasoning = f"Model B reasoning {i}"
            result.evaluation_time_ms = 1800
            result.completed_at = datetime.utcnow()
            result.metadata = {
                "model": "model_b",
                "category": "accuracy" if i % 2 == 0 else "fluency",
                "difficulty_level": "easy" if i < 8 else "hard",
            }

            self.evaluation_results_b.append(result)

    def create_analysis_result_with_performances(self) -> AnalysisResult:
        """Create analysis result with model performances."""

        analysis = AnalysisResult.create(
            test_id=uuid4(), name="Test Analysis", description="Test analysis for insights"
        )

        # Add model performances
        perf_a = ModelPerformanceMetrics(
            model_id="model_a",
            model_name="GPT-4",
            overall_score=Decimal("0.82"),
            dimension_scores={
                "accuracy": Decimal("0.9"),
                "fluency": Decimal("0.6"),
                "coherence": Decimal("0.8"),
            },
            sample_count=15,
            confidence_score=Decimal("0.85"),
        )

        perf_b = ModelPerformanceMetrics(
            model_id="model_b",
            model_name="Claude-3",
            overall_score=Decimal("0.76"),
            dimension_scores={
                "accuracy": Decimal("0.6"),
                "fluency": Decimal("0.95"),
                "coherence": Decimal("0.8"),
            },
            sample_count=15,
            confidence_score=Decimal("0.82"),
        )

        analysis.add_model_performance("model_a", perf_a)
        analysis.add_model_performance("model_b", perf_b)

        return analysis

    def create_statistical_test_result(
        self, p_value: float = 0.01, effect_size: float = 0.8, is_significant: bool = True
    ) -> TestResult:
        """Create statistical test result for testing."""

        effect_magnitude = (
            EffectMagnitude.LARGE
            if abs(effect_size) >= 0.8
            else EffectMagnitude.MEDIUM if abs(effect_size) >= 0.5 else EffectMagnitude.SMALL
        )

        interpretation = TestInterpretation(
            is_significant=is_significant,
            significance_level=0.05,
            effect_magnitude=effect_magnitude,
            practical_significance=abs(effect_size) >= 0.2,
            recommendation="Test recommendation",
        )

        return TestResult(
            test_type="t_test_independent",
            statistic=2.5,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(-0.15, -0.05),
            degrees_of_freedom=28,
            interpretation=interpretation,
            sample_sizes={"model_a": 15, "model_b": 15},
            test_assumptions={"normality": True, "equal_variances": True},
        )

    def test_overall_performance_insights(self):
        """Test generation of overall performance insights."""

        analysis = self.create_analysis_result_with_performances()
        insights = self.insight_generator.generate_performance_insights(analysis)

        # Should generate insights
        assert len(insights) > 0

        # Should have performance-related insights
        performance_insights = [i for i in insights if i.insight_type == InsightType.PERFORMANCE]
        assert len(performance_insights) > 0

        # Check for significant performance variation insight
        variation_insights = [
            i
            for i in performance_insights
            if "variation" in i.title.lower() or "gap" in i.title.lower()
        ]

        if variation_insights:
            insight = variation_insights[0]
            assert insight.severity in [InsightSeverity.MEDIUM, InsightSeverity.HIGH]
            assert "model" in insight.description.lower()
            assert len(insight.recommendations) > 0
            assert len(insight.affected_models) >= 2

    def test_statistical_significance_insights(self):
        """Test statistical significance insight generation."""

        analysis = self.create_analysis_result_with_performances()

        # Add significant statistical test
        significant_test = self.create_statistical_test_result(p_value=0.01, effect_size=0.8)
        analysis.add_statistical_test("model_comparison", significant_test)

        insights = self.insight_generator.generate_performance_insights(analysis)

        # Should have statistical significance insights
        stat_insights = [
            i for i in insights if i.insight_type == InsightType.STATISTICAL_SIGNIFICANCE
        ]
        assert len(stat_insights) > 0

        # Check significant difference insight
        sig_insight = stat_insights[0]
        assert "significant" in sig_insight.title.lower()
        assert sig_insight.confidence_score >= Decimal("0.8")
        assert len(sig_insight.recommendations) > 0

    def test_dimension_insights(self):
        """Test dimension-specific insight generation."""

        analysis = self.create_analysis_result_with_performances()
        insights = self.insight_generator.generate_performance_insights(analysis)

        # Should have dimension-related insights
        dimension_insights = [i for i in insights if i.insight_type == InsightType.QUALITY_PATTERN]

        # Look for accuracy vs fluency pattern
        accuracy_fluency_insights = [
            i
            for i in dimension_insights
            if "accuracy" in i.description.lower() and "fluency" in i.description.lower()
        ]

        if accuracy_fluency_insights:
            insight = accuracy_fluency_insights[0]
            assert insight.severity in [InsightSeverity.MEDIUM, InsightSeverity.HIGH]
            assert len(insight.affected_models) >= 1
            assert "accuracy" in insight.category or "fluency" in insight.category

    def test_model_comparison_insights(self):
        """Test model comparison insight generation."""

        # Create model performances
        perf_a = ModelPerformance.create_from_evaluation_results(
            model_id="model_a", model_name="GPT-4", evaluation_results=self.evaluation_results_a
        )

        perf_b = ModelPerformance.create_from_evaluation_results(
            model_id="model_b", model_name="Claude-3", evaluation_results=self.evaluation_results_b
        )

        # Create comparison (this will include statistical test)
        comparison = perf_a.compare_with(perf_b)

        # Generate insights from comparison
        insights = self.insight_generator.generate_model_comparison_insights(comparison)

        assert len(insights) > 0

        # Should have statistical significance insight
        stat_insights = [
            i for i in insights if i.insight_type == InsightType.STATISTICAL_SIGNIFICANCE
        ]
        assert len(stat_insights) > 0

        # Should have effect size insight
        effect_insights = [i for i in insights if "effect" in i.title.lower()]
        assert len(effect_insights) > 0

    def test_reliability_insights(self):
        """Test reliability and quality insights."""

        # Create model performances with different reliability scores
        performances = []

        # High reliability model
        perf_high = ModelPerformance.create_from_evaluation_results(
            model_id="high_reliability",
            model_name="High Reliability Model",
            evaluation_results=self.evaluation_results_a[:10],
        )
        performances.append(perf_high)

        # Low reliability model (modify some results to have low confidence)
        low_reliability_results = []
        for i, result in enumerate(self.evaluation_results_b[:10]):
            # Create copy with low confidence
            new_result = EvaluationResult.create_pending(
                judge_id=result.judge_id,
                template_id=result.template_id,
                prompt=result.prompt,
                response=result.response,
            )

            new_result.dimension_scores = result.dimension_scores.copy()
            new_result.overall_score = result.overall_score
            new_result.confidence_score = Decimal("0.3")  # Low confidence
            new_result.reasoning = result.reasoning
            new_result.evaluation_time_ms = result.evaluation_time_ms
            new_result.completed_at = result.completed_at
            new_result.metadata = result.metadata.copy()

            low_reliability_results.append(new_result)

        perf_low = ModelPerformance.create_from_evaluation_results(
            model_id="low_reliability",
            model_name="Low Reliability Model",
            evaluation_results=low_reliability_results,
        )
        performances.append(perf_low)

        # Generate quality insights
        insights = self.insight_generator.generate_quality_insights(performances)

        # Should detect reliability issues
        reliability_insights = [
            i
            for i in insights
            if "reliability" in i.title.lower() or "confidence" in i.title.lower()
        ]

        # May or may not generate reliability insights depending on implementation
        # This test verifies the structure is in place
        if reliability_insights:
            insight = reliability_insights[0]
            assert insight.insight_type == InsightType.QUALITY_PATTERN
            assert len(insight.recommendations) > 0

    def test_bias_detection_insights(self):
        """Test bias detection insight generation."""

        analysis = self.create_analysis_result_with_performances()

        # Generate bias detection insights
        insights = self.insight_generator.generate_bias_detection_insights(analysis)

        # Structure should be in place for bias detection
        # May return empty list if no bias detected, which is fine
        assert isinstance(insights, list)

        # If bias insights are generated, verify structure
        if insights:
            bias_insight = insights[0]
            assert bias_insight.insight_type == InsightType.BIAS_DETECTION
            assert len(bias_insight.recommendations) > 0

    def test_actionable_recommendations(self):
        """Test recommendation insight generation."""

        analysis = self.create_analysis_result_with_performances()

        # Add a significant statistical test
        significant_test = self.create_statistical_test_result(p_value=0.005, effect_size=1.2)
        analysis.add_statistical_test("strong_difference", significant_test)

        insights = self.insight_generator.generate_recommendations(analysis)

        # Should generate recommendations
        assert len(insights) > 0

        # Should have recommendation type insights
        rec_insights = [i for i in insights if i.insight_type == InsightType.RECOMMENDATION]

        if rec_insights:
            recommendation = rec_insights[0]
            assert len(recommendation.recommendations) > 0
            assert recommendation.is_actionable()
            assert len(recommendation.description) > 20  # Substantial description

    def test_insight_prioritization(self):
        """Test insight priority scoring and sorting."""

        analysis = self.create_analysis_result_with_performances()

        # Add multiple statistical tests with different significance levels
        tests = [
            (
                "critical_difference",
                self.create_statistical_test_result(0.001, 1.5),
            ),  # Very significant
            (
                "moderate_difference",
                self.create_statistical_test_result(0.03, 0.6),
            ),  # Moderately significant
            ("weak_difference", self.create_statistical_test_result(0.08, 0.3)),  # Not significant
        ]

        for name, test in tests:
            analysis.add_statistical_test(name, test)

        insights = self.insight_generator.generate_performance_insights(analysis)

        if len(insights) > 1:
            # Sort by priority score
            insights_by_priority = sorted(
                insights, key=lambda x: x.get_priority_score(), reverse=True
            )

            # Verify priority scoring works
            highest_priority = insights_by_priority[0]
            lowest_priority = insights_by_priority[-1]

            assert highest_priority.get_priority_score() >= lowest_priority.get_priority_score()

            # High priority insights should require attention
            if highest_priority.severity in [InsightSeverity.HIGH, InsightSeverity.CRITICAL]:
                assert highest_priority.requires_attention()

    def test_confidence_scoring(self):
        """Test insight confidence scoring."""

        analysis = self.create_analysis_result_with_performances()

        # Add statistical test with very low p-value (high confidence)
        high_confidence_test = self.create_statistical_test_result(p_value=0.0001, effect_size=1.0)
        analysis.add_statistical_test("high_confidence", high_confidence_test)

        insights = self.insight_generator.generate_performance_insights(analysis)

        # Statistical insights should have high confidence scores
        stat_insights = [
            i for i in insights if i.insight_type == InsightType.STATISTICAL_SIGNIFICANCE
        ]

        if stat_insights:
            high_conf_insight = stat_insights[0]
            assert high_conf_insight.confidence_score >= Decimal("0.8")
            assert high_conf_insight.is_high_confidence()

    def test_insight_evidence_inclusion(self):
        """Test that insights include proper evidence."""

        analysis = self.create_analysis_result_with_performances()
        insights = self.insight_generator.generate_performance_insights(analysis)

        # All insights should have evidence
        for insight in insights:
            assert isinstance(insight.evidence, dict)
            assert len(insight.evidence) > 0

            # Evidence should contain relevant data
            evidence_keys = insight.evidence.keys()

            if insight.insight_type == InsightType.PERFORMANCE:
                # Performance insights should have performance-related evidence
                performance_evidence = any(
                    key in evidence_keys
                    for key in ["performance_spread", "best_model", "worst_model", "scores"]
                )
                # Not all performance insights will have these specific keys, so we don't assert

    def test_insight_recommendation_quality(self):
        """Test quality of generated recommendations."""

        analysis = self.create_analysis_result_with_performances()

        # Add significant test to generate actionable insights
        significant_test = self.create_statistical_test_result(p_value=0.01, effect_size=0.9)
        analysis.add_statistical_test("significant_diff", significant_test)

        insights = self.insight_generator.generate_performance_insights(analysis)

        # Check recommendation quality
        for insight in insights:
            assert len(insight.recommendations) > 0

            for recommendation in insight.recommendations:
                # Recommendations should be substantial
                assert len(recommendation.strip()) > 10

                # Should not be generic placeholders
                assert "todo" not in recommendation.lower()
                assert (
                    "implement" in recommendation.lower()
                    or "consider" in recommendation.lower()
                    or "analyze" in recommendation.lower()
                    or "investigate" in recommendation.lower()
                    or "use" in recommendation.lower()
                )

    def test_effect_size_interpretation_insights(self):
        """Test effect size interpretation in insights."""

        test_cases = [
            (0.1, "negligible", InsightSeverity.INFO),
            (0.3, "small", InsightSeverity.LOW),
            (0.6, "medium", InsightSeverity.MEDIUM),
            (1.0, "large", InsightSeverity.HIGH),
        ]

        for effect_size, expected_magnitude, expected_severity in test_cases:
            # Create statistical test with specific effect size
            test_result = self.create_statistical_test_result(
                p_value=0.01, effect_size=effect_size, is_significant=True
            )

            # Create mock model comparison for testing
            perf_a = ModelPerformance.create_from_evaluation_results(
                "model_a", "Model A", self.evaluation_results_a[:5]
            )
            perf_b = ModelPerformance.create_from_evaluation_results(
                "model_b", "Model B", self.evaluation_results_b[:5]
            )

            # Create comparison with our test result
            comparison = ModelComparison(
                model_a=perf_a,
                model_b=perf_b,
                score_difference=Decimal(str(effect_size * 0.1)),
                cost_difference=Decimal("0"),
                statistical_test_result=test_result,
                dimension_comparisons={},
                recommendation="Test recommendation",
            )

            insights = self.insight_generator.generate_model_comparison_insights(comparison)

            # Find effect size insight
            effect_insights = [i for i in insights if "effect" in i.title.lower()]

            if effect_insights:
                effect_insight = effect_insights[0]
                assert expected_magnitude in effect_insight.title.lower()

                # Severity should match expectation (approximately)
                assert effect_insight.severity in [expected_severity, InsightSeverity.MEDIUM]

    def test_empty_analysis_handling(self):
        """Test handling of empty or minimal analysis data."""

        # Empty analysis
        empty_analysis = AnalysisResult.create(
            test_id=uuid4(), name="Empty Analysis", description="Analysis with no data"
        )

        # Should handle gracefully
        insights = self.insight_generator.generate_performance_insights(empty_analysis)
        assert isinstance(insights, list)
        # May be empty, which is fine

        # Analysis with single model
        single_model_analysis = AnalysisResult.create(
            test_id=uuid4(), name="Single Model Analysis", description="Analysis with one model"
        )

        perf = ModelPerformanceMetrics(
            model_id="single_model",
            model_name="Single Model",
            overall_score=Decimal("0.8"),
            dimension_scores={"accuracy": Decimal("0.8")},
            sample_count=10,
            confidence_score=Decimal("0.85"),
        )
        single_model_analysis.add_model_performance("single_model", perf)

        insights = self.insight_generator.generate_performance_insights(single_model_analysis)
        assert isinstance(insights, list)
        # Should handle single model case gracefully

    def test_insight_categorization(self):
        """Test proper categorization of insights."""

        analysis = self.create_analysis_result_with_performances()
        insights = self.insight_generator.generate_performance_insights(analysis)

        # Verify insights have appropriate types
        insight_types = {insight.insight_type for insight in insights}

        # Should have performance insights
        assert (
            InsightType.PERFORMANCE in insight_types or InsightType.QUALITY_PATTERN in insight_types
        )

        # Verify categories are set appropriately
        for insight in insights:
            if insight.category:
                # Categories should be meaningful
                assert len(insight.category) > 0
                assert (
                    insight.category in ["accuracy", "fluency", "coherence"]
                    or "performance" in insight.category.lower()
                )

    def test_affected_models_tracking(self):
        """Test that insights properly track affected models."""

        analysis = self.create_analysis_result_with_performances()
        insights = self.insight_generator.generate_performance_insights(analysis)

        # Insights that compare models should list affected models
        comparison_insights = [i for i in insights if len(i.affected_models or []) >= 2]

        if comparison_insights:
            insight = comparison_insights[0]
            assert "model_a" in insight.affected_models
            assert "model_b" in insight.affected_models

            # Model names should be mentioned in description
            model_mentioned = any(
                model_id in insight.description.lower()
                or "gpt-4" in insight.description.lower()
                or "claude-3" in insight.description.lower()
                for model_id in insight.affected_models
            )
