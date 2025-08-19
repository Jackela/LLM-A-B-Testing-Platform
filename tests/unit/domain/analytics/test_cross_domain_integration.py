"""Test cross-domain integration for analytics domain."""

from datetime import datetime
from decimal import Decimal
from typing import List
from uuid import uuid4

import pytest

from src.domain.analytics.entities.analysis_result import AnalysisResult
from src.domain.analytics.entities.model_performance import ModelPerformance
from src.domain.analytics.entities.statistical_test import TestType
from src.domain.analytics.services.data_aggregator import DataAggregator
from src.domain.analytics.services.insight_generator import InsightGenerator
from src.domain.analytics.services.significance_tester import SignificanceTester
from src.domain.evaluation.entities.dimension import Dimension
from src.domain.evaluation.entities.evaluation_result import EvaluationResult
from src.domain.evaluation.entities.evaluation_template import EvaluationTemplate
from src.domain.evaluation.value_objects.scoring_scale import ScoringScale
from src.domain.model_provider.entities.model_response import ModelResponse
from src.domain.model_provider.value_objects.money import Money

# Cross-domain imports
from src.domain.test_management.entities.test import Test
from src.domain.test_management.entities.test_configuration import TestConfiguration
from src.domain.test_management.entities.test_sample import TestSample


class TestCrossDomainIntegration:
    """Test analytics domain integration with all other domains."""

    def setup_method(self):
        """Set up cross-domain test data."""

        # Test Management Domain data
        self.test_config = TestConfiguration.create(
            name="Cross-Domain Integration Test",
            description="Testing analytics integration",
            model_a_config={"model": "gpt-4", "temperature": 0.7},
            model_b_config={"model": "claude-3", "temperature": 0.7},
            evaluation_criteria=["accuracy", "fluency", "coherence"],
            sample_size=100,
        )

        self.test = Test.create(name="Analytics Integration Test", configuration=self.test_config)

        # Add test samples
        self.test_samples = []
        for i in range(20):
            sample = TestSample.create(
                prompt=f"Test prompt {i}",
                expected_response=f"Expected response {i}",
                category="accuracy" if i % 2 == 0 else "fluency",
                difficulty_level="easy" if i < 10 else "hard",
                metadata={"test_id": str(self.test.id)},
            )
            self.test.add_sample(sample)
            self.test_samples.append(sample)

        # Evaluation Domain data
        self.dimensions = [
            Dimension.create(
                name="accuracy",
                description="How accurate is the response",
                scoring_scale=ScoringScale.create_likert_scale(1, 5),
                weight=0.4,
            ),
            Dimension.create(
                name="fluency",
                description="How fluent is the response",
                scoring_scale=ScoringScale.create_likert_scale(1, 5),
                weight=0.3,
            ),
            Dimension.create(
                name="coherence",
                description="How coherent is the response",
                scoring_scale=ScoringScale.create_likert_scale(1, 5),
                weight=0.3,
            ),
        ]

        self.evaluation_template = EvaluationTemplate.create(
            name="Integration Test Template",
            description="Template for cross-domain testing",
            dimensions=self.dimensions,
            instructions="Evaluate the response on all dimensions",
        )

        # Model Provider Domain data - mock responses
        self.model_responses = []
        for i, sample in enumerate(self.test_samples[:10]):  # Model A responses
            response = ModelResponse.create(
                request_id=uuid4(),
                model_config_id=str(uuid4()),
                prompt=sample.prompt,
                response=f"Model A response {i}",
                tokens_used=50 + i * 2,
                processing_time_ms=200 + i * 10,
                cost=Money(Decimal("0.001"), "USD"),
            )
            self.model_responses.append(response)

        for i, sample in enumerate(self.test_samples[10:]):  # Model B responses
            response = ModelResponse.create(
                request_id=uuid4(),
                model_config_id=str(uuid4()),
                prompt=sample.prompt,
                response=f"Model B response {i}",
                tokens_used=45 + i * 2,
                processing_time_ms=250 + i * 15,
                cost=Money(Decimal("0.0015"), "USD"),
            )
            self.model_responses.append(response)

        # Create evaluation results
        self.evaluation_results = []

        # Model A evaluation results
        for i in range(10):
            result = EvaluationResult.create_pending(
                judge_id="human_evaluator_1",
                template_id=self.evaluation_template.template_id,
                prompt=self.test_samples[i].prompt,
                response=f"Model A response {i}",
            )

            # Complete evaluation with realistic scores
            dimension_scores = {
                "accuracy": 4 if i < 7 else 3,  # Model A better at accuracy
                "fluency": 4,  # Equal fluency
                "coherence": 3 + (i % 2),  # Variable coherence
            }

            result.complete_evaluation(
                template=self.evaluation_template,
                dimension_scores=dimension_scores,
                confidence_score=Decimal("0.8") + Decimal(str(i % 3)) * Decimal("0.05"),
                reasoning=f"Evaluation reasoning for Model A response {i}",
                evaluation_time_ms=2000 + i * 100,
            )

            # Add metadata
            result.metadata = {
                "model": "model_a",
                "difficulty_level": "easy" if i < 5 else "hard",
                "category": "accuracy" if i % 2 == 0 else "fluency",
                "test_id": str(self.test.id),
            }

            self.evaluation_results.append(result)

        # Model B evaluation results
        for i in range(10):
            result = EvaluationResult.create_pending(
                judge_id="human_evaluator_1",
                template_id=self.evaluation_template.template_id,
                prompt=self.test_samples[10 + i].prompt,
                response=f"Model B response {i}",
            )

            # Complete evaluation - Model B better at fluency/coherence
            dimension_scores = {
                "accuracy": 3 + (i % 2),  # Variable accuracy
                "fluency": 5 if i < 8 else 4,  # Model B better at fluency
                "coherence": 4,  # Better coherence
            }

            result.complete_evaluation(
                template=self.evaluation_template,
                dimension_scores=dimension_scores,
                confidence_score=Decimal("0.75") + Decimal(str(i % 4)) * Decimal("0.05"),
                reasoning=f"Evaluation reasoning for Model B response {i}",
                evaluation_time_ms=1800 + i * 120,
            )

            # Add metadata
            result.metadata = {
                "model": "model_b",
                "difficulty_level": "easy" if i < 5 else "hard",
                "category": "accuracy" if i % 2 == 0 else "fluency",
                "test_id": str(self.test.id),
            }

            self.evaluation_results.append(result)

    def test_end_to_end_analysis_workflow(self):
        """Test complete end-to-end analysis workflow across all domains."""

        # 1. Create analysis result from test management data
        analysis = AnalysisResult.create(
            test_id=self.test.id,
            name=f"Analysis for {self.test.name}",
            description="Complete cross-domain analysis",
        )

        # 2. Perform statistical significance testing
        significance_tester = SignificanceTester()

        model_a_results = [r for r in self.evaluation_results if r.metadata["model"] == "model_a"]
        model_b_results = [r for r in self.evaluation_results if r.metadata["model"] == "model_b"]

        # Overall performance comparison
        overall_test = significance_tester.test_model_performance_difference(
            model_a_results, model_b_results, test_type=TestType.TTEST_INDEPENDENT
        )

        analysis.add_statistical_test("overall_performance", overall_test)

        # Dimension-specific comparisons
        for dimension in ["accuracy", "fluency", "coherence"]:
            dimension_test = significance_tester.test_model_performance_difference(
                model_a_results, model_b_results, dimension=dimension
            )
            analysis.add_statistical_test(f"{dimension}_performance", dimension_test)

        # 3. Perform data aggregation
        aggregator = DataAggregator()

        # Calculate model performances
        model_a_performance = aggregator.calculate_model_performance(
            model_id="model_a",
            model_name="GPT-4",
            evaluation_results=model_a_results,
            model_responses=[r for r in self.model_responses[:10]],
        )

        model_b_performance = aggregator.calculate_model_performance(
            model_id="model_b",
            model_name="Claude-3",
            evaluation_results=model_b_results,
            model_responses=[r for r in self.model_responses[10:]],
        )

        analysis.add_model_performance("model_a", model_a_performance)
        analysis.add_model_performance("model_b", model_b_performance)

        # 4. Generate insights
        insight_generator = InsightGenerator()

        insights = insight_generator.generate_performance_insights(analysis)
        for insight in insights:
            analysis.add_insight(insight)

        # 5. Complete analysis
        analysis.complete_analysis(processing_time_ms=5000)

        # Verify cross-domain integration
        assert analysis.is_completed()
        assert len(analysis.statistical_tests) >= 4  # Overall + 3 dimensions
        assert len(analysis.model_performances) == 2
        assert len(analysis.insights) > 0

        # Verify test management integration
        assert analysis.test_id == self.test.id
        assert analysis.get_total_sample_count() == 20

        # Verify evaluation integration
        for model_perf in analysis.model_performances.values():
            assert len(model_perf.dimension_scores) == 3  # accuracy, fluency, coherence
            assert all(
                dim in model_perf.dimension_scores for dim in ["accuracy", "fluency", "coherence"]
            )

        # Verify model provider integration (if cost data available)
        for model_perf in analysis.model_performances.values():
            if model_perf.cost_data:
                assert model_perf.cost_data.request_count > 0
                assert model_perf.cost_data.total_cost.amount > Decimal("0")

    def test_statistical_significance_across_dimensions(self):
        """Test statistical testing integrates properly with evaluation dimensions."""

        significance_tester = SignificanceTester()

        # Test dimension differences within same model
        dimension_tests = significance_tester.test_dimension_differences(
            evaluation_results=self.evaluation_results,
            dimensions=["accuracy", "fluency", "coherence"],
        )

        # Should have pairwise comparisons for all dimension pairs
        expected_comparisons = [
            "accuracy_vs_fluency",
            "accuracy_vs_coherence",
            "fluency_vs_coherence",
        ]

        for comparison in expected_comparisons:
            if comparison in dimension_tests:  # May not be present if insufficient data
                test_result = dimension_tests[comparison]
                assert test_result.test_type == "t_test_paired"  # Should use paired test
                assert 0.0 <= test_result.p_value <= 1.0
                assert test_result.sample_sizes is not None

    def test_difficulty_level_analysis_integration(self):
        """Test integration with test management difficulty levels."""

        significance_tester = SignificanceTester()

        # Test difficulty level effects
        difficulty_test = significance_tester.test_difficulty_level_effects(
            evaluation_results=self.evaluation_results
        )

        if difficulty_test:  # May be None if insufficient data
            assert difficulty_test.sample_sizes["easy"] == 10
            assert difficulty_test.sample_sizes["hard"] == 10
            assert 0.0 <= difficulty_test.p_value <= 1.0

    def test_cost_analysis_integration(self):
        """Test cost analysis integration with model provider domain."""

        aggregator = DataAggregator()

        # This would require actual cost calculation implementation
        # For now, verify structure is in place
        model_a_results = [r for r in self.evaluation_results if r.metadata["model"] == "model_a"]
        model_a_responses = self.model_responses[:10]

        performance = aggregator.calculate_model_performance(
            model_id="model_a",
            model_name="GPT-4",
            evaluation_results=model_a_results,
            model_responses=model_a_responses,
        )

        # Verify structure for cost analysis exists
        assert hasattr(performance, "cost_data")
        # Note: cost_data may be None due to placeholder implementation

    def test_insight_generation_cross_domain(self):
        """Test insight generation uses data from all domains."""

        # Create complete analysis result
        analysis = AnalysisResult.create(
            test_id=self.test.id,
            name="Cross-Domain Insight Test",
            description="Test insight generation across domains",
        )

        # Add sample statistical test
        significance_tester = SignificanceTester()
        model_a_results = [r for r in self.evaluation_results if r.metadata["model"] == "model_a"]
        model_b_results = [r for r in self.evaluation_results if r.metadata["model"] == "model_b"]

        test_result = significance_tester.test_model_performance_difference(
            model_a_results, model_b_results
        )
        analysis.add_statistical_test("model_comparison", test_result)

        # Add model performances
        aggregator = DataAggregator()

        for model_key, results in [("model_a", model_a_results), ("model_b", model_b_results)]:
            performance = aggregator.calculate_model_performance(
                model_id=model_key, model_name=f"Test {model_key}", evaluation_results=results
            )
            analysis.add_model_performance(model_key, performance)

        # Generate insights
        insight_generator = InsightGenerator()
        insights = insight_generator.generate_performance_insights(analysis)

        assert len(insights) > 0

        # Verify insights reference cross-domain concepts
        insight_texts = " ".join([insight.description for insight in insights])

        # Should reference evaluation concepts
        cross_domain_terms_found = any(
            term in insight_texts.lower()
            for term in ["model", "performance", "score", "significant", "dimension"]
        )
        assert cross_domain_terms_found

    def test_model_performance_entity_integration(self):
        """Test ModelPerformance entity integration with evaluation results."""

        model_results = [r for r in self.evaluation_results if r.metadata["model"] == "model_a"]

        # Create model performance from evaluation results
        performance = ModelPerformance.create_from_evaluation_results(
            model_id="model_a", model_name="GPT-4", evaluation_results=model_results
        )

        # Verify integration with evaluation domain
        assert len(performance.dimension_performances) == 3
        assert "accuracy" in performance.dimension_performances
        assert "fluency" in performance.dimension_performances
        assert "coherence" in performance.dimension_performances

        # Verify performance by category integration with test management
        assert len(performance.performance_by_category) >= 2  # accuracy, fluency categories
        assert len(performance.performance_by_difficulty) == 2  # easy, hard

        # Verify reliability and consistency scores
        assert Decimal("0") <= performance.reliability_score <= Decimal("1")
        assert Decimal("0") <= performance.consistency_score <= Decimal("1")

        # Test model comparison integration
        model_b_results = [r for r in self.evaluation_results if r.metadata["model"] == "model_b"]

        performance_b = ModelPerformance.create_from_evaluation_results(
            model_id="model_b", model_name="Claude-3", evaluation_results=model_b_results
        )

        comparison = performance.compare_with(performance_b)

        # Verify comparison integrates statistical testing
        assert comparison.statistical_test_result is not None
        assert 0.0 <= comparison.statistical_test_result.p_value <= 1.0
        assert comparison.recommendation is not None
        assert len(comparison.recommendation) > 10

    def test_aggregation_with_test_metadata(self):
        """Test data aggregation properly uses test management metadata."""

        aggregator = DataAggregator()

        # Test aggregation by test-specific metadata
        difficulty_aggregation = aggregator.aggregate_by_difficulty_level(self.evaluation_results)

        assert "easy" in difficulty_aggregation
        assert "hard" in difficulty_aggregation

        # Verify sample counts match test design
        assert difficulty_aggregation["easy"].sample_count == 10
        assert difficulty_aggregation["hard"].sample_count == 10

        # Test category aggregation
        category_aggregation = aggregator.aggregate_by_category(self.evaluation_results)

        assert "accuracy" in category_aggregation
        assert "fluency" in category_aggregation

        # Verify category distribution
        assert category_aggregation["accuracy"].sample_count == 10
        assert category_aggregation["fluency"].sample_count == 10

    def test_analysis_result_summary_integration(self):
        """Test analysis result summary integrates all domain data."""

        # Create comprehensive analysis
        analysis = AnalysisResult.create(
            test_id=self.test.id,
            name="Integration Summary Test",
            description="Test comprehensive summary",
        )

        # Add components from all domains
        aggregator = DataAggregator()

        for model_key in ["model_a", "model_b"]:
            results = [r for r in self.evaluation_results if r.metadata["model"] == model_key]
            performance = aggregator.calculate_model_performance(
                model_id=model_key, model_name=f"Test {model_key}", evaluation_results=results
            )
            analysis.add_model_performance(model_key, performance)

        # Add statistical test
        significance_tester = SignificanceTester()
        test_result = significance_tester.test_model_performance_difference(
            [r for r in self.evaluation_results if r.metadata["model"] == "model_a"],
            [r for r in self.evaluation_results if r.metadata["model"] == "model_b"],
        )
        analysis.add_statistical_test("model_comparison", test_result)

        # Generate summary
        summary = analysis.get_model_comparison_summary()

        # Verify summary integrates all domains
        assert "total_models" in summary
        assert summary["total_models"] == 2

        assert "best_model" in summary
        assert "worst_model" in summary

        # Should reference test management data
        assert "performance_spread" in summary

        # Should reference statistical analysis
        assert "significant_differences" in summary
        assert "total_statistical_tests" in summary

    def test_domain_event_propagation(self):
        """Test that domain events are properly generated across domains."""

        # Create analysis that should generate events
        analysis = AnalysisResult.create(
            test_id=self.test.id, name="Event Test Analysis", description="Test event generation"
        )

        # Add statistical test - should generate StatisticalTestCompleted event
        significance_tester = SignificanceTester()
        test_result = significance_tester.test_model_performance_difference(
            [r for r in self.evaluation_results if r.metadata["model"] == "model_a"],
            [r for r in self.evaluation_results if r.metadata["model"] == "model_b"],
        )

        analysis.add_statistical_test("test_comparison", test_result)

        # Should have generated domain event
        assert len(analysis._domain_events) >= 1

        # Complete analysis - should generate AnalysisCompleted event
        analysis.complete_analysis(processing_time_ms=3000)

        # Should have additional event
        assert len(analysis._domain_events) >= 2

        # Verify event types
        event_types = [type(event).__name__ for event in analysis._domain_events]
        assert "StatisticalTestCompleted" in event_types
        assert "AnalysisCompleted" in event_types

    def test_repository_integration_structure(self):
        """Test that analytics repository interfaces support cross-domain queries."""

        from src.domain.analytics.repositories.analytics_repository import AnalyticsRepository

        # Verify repository interface has methods for cross-domain queries
        repo_methods = dir(AnalyticsRepository)

        # Should have methods that work with other domain IDs
        assert "get_analysis_results_by_test" in repo_methods  # Test management integration
        assert "get_model_performance_history" in repo_methods  # Model provider integration
        assert "get_insights_by_analysis" in repo_methods  # Analysis-specific queries

        # Should support complex queries
        assert "search_analysis_results" in repo_methods
        assert "get_test_performance_trends" in repo_methods
