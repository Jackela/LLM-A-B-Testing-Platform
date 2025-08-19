"""Tests for Metrics Calculator Service."""

import asyncio
import statistics
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.application.services.analytics.metrics_calculator import (
    AggregationMethod,
    CalculatedMetric,
    MetricCalculationConfig,
    MetricsCalculator,
    MetricType,
    ModelMetricsSummary,
)
from src.domain.analytics.exceptions import CalculationError, ValidationError
from src.domain.analytics.value_objects.performance_score import PerformanceScore
from src.domain.evaluation.entities.evaluation_result import EvaluationResult


class TestMetricsCalculator:
    """Test suite for Metrics Calculator."""

    @pytest.fixture
    def mock_analytics_repository(self):
        """Create mock analytics repository."""
        return AsyncMock()

    @pytest.fixture
    def calculator(self, mock_analytics_repository):
        """Create calculator instance with mocked dependencies."""
        return MetricsCalculator(mock_analytics_repository)

    @pytest.fixture
    def sample_evaluation_results(self):
        """Create sample evaluation results."""
        results = []
        for i in range(20):
            result = Mock(spec=EvaluationResult)
            result.model_id = f"model_{i % 3}"  # Three models
            result.overall_score = Decimal(str(0.7 + (i % 5) * 0.05))
            result.dimension_scores = {
                "accuracy": Decimal(str(0.8 + (i % 4) * 0.05)),
                "fluency": Decimal(str(0.75 + (i % 3) * 0.05)),
                "relevance": Decimal(str(0.85 - (i % 2) * 0.1)),
            }
            result.is_completed.return_value = True
            result.has_error.return_value = False
            result.completed_at = datetime.utcnow() - timedelta(hours=i)
            result.metadata = {
                "model_name": f"Model {i % 3}",
                "response_time_ms": 500 + (i % 10) * 50,
                "cost": 0.01 + (i % 5) * 0.002,
                "tokens": 100 + (i % 20) * 10,
            }
            results.append(result)
        return results

    @pytest.fixture
    def config(self):
        """Create default metric calculation config."""
        return MetricCalculationConfig()

    @pytest.mark.asyncio
    async def test_calculate_comprehensive_metrics_success(
        self, calculator, mock_analytics_repository, sample_evaluation_results, config
    ):
        """Test successful comprehensive metrics calculation."""
        test_id = uuid4()

        # Setup mock
        mock_analytics_repository.get_evaluation_results.return_value = sample_evaluation_results

        # Execute
        result = await calculator.calculate_comprehensive_metrics(test_id, config)

        # Verify
        assert isinstance(result, dict)
        assert len(result) == 3  # Three models

        for model_id, summary in result.items():
            assert isinstance(summary, ModelMetricsSummary)
            assert summary.model_id == model_id
            assert 0 <= summary.overall_score <= 1
            assert summary.quality_grade in ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D"]
            assert summary.ranking_position is not None
            assert summary.percentile_rank is not None
            assert isinstance(summary.recommendations, list)

        mock_analytics_repository.get_evaluation_results.assert_called_once_with(test_id)

    @pytest.mark.asyncio
    async def test_calculate_comprehensive_metrics_no_data(
        self, calculator, mock_analytics_repository, config
    ):
        """Test metrics calculation with no evaluation results."""
        test_id = uuid4()

        # Setup mock to return empty list
        mock_analytics_repository.get_evaluation_results.return_value = []

        # Execute and verify exception
        with pytest.raises(ValidationError, match="No evaluation results found"):
            await calculator.calculate_comprehensive_metrics(test_id, config)

    @pytest.mark.asyncio
    async def test_calculate_model_performance_score_success(
        self, calculator, sample_evaluation_results, config
    ):
        """Test model performance score calculation."""
        # Filter results for one model
        model_results = [r for r in sample_evaluation_results if r.model_id == "model_0"]

        # Execute
        result = await calculator.calculate_model_performance_score(model_results, config=config)

        # Verify
        assert isinstance(result, PerformanceScore)
        assert 0 <= result.overall_score <= 1
        assert len(result.dimension_scores) >= 3
        assert result.sample_count == len(model_results)
        assert 0 <= result.consistency_score <= 1

    @pytest.mark.asyncio
    async def test_calculate_model_performance_score_with_weights(
        self, calculator, sample_evaluation_results
    ):
        """Test model performance score calculation with custom weights."""
        model_results = [r for r in sample_evaluation_results if r.model_id == "model_0"]
        weights = {"accuracy": 0.5, "fluency": 0.3, "relevance": 0.2}

        # Execute
        result = await calculator.calculate_model_performance_score(model_results, weights=weights)

        # Verify
        assert isinstance(result, PerformanceScore)
        assert result.metadata["calculation_method"] == "weighted_average"
        assert result.metadata["weights_used"] == weights

    @pytest.mark.asyncio
    async def test_calculate_model_performance_score_empty_results(self, calculator):
        """Test performance score calculation with empty results."""
        # Execute and verify exception
        with pytest.raises(ValidationError, match="Evaluation results cannot be empty"):
            await calculator.calculate_model_performance_score([])

    @pytest.mark.asyncio
    async def test_calculate_cost_efficiency_metrics_success(self, calculator):
        """Test cost efficiency metrics calculation."""
        # Create mock model performances with cost data
        model_performances = {}
        for i in range(3):
            cost_metrics = Mock()
            cost_metrics.cost_per_sample.amount = Decimal(str(0.01 + i * 0.005))
            cost_metrics.total_cost.amount = Decimal(str(0.5 + i * 0.2))
            cost_metrics.total_tokens = 1000 + i * 200

            performance = Mock()
            performance.model_id = f"model_{i}"
            performance.model_name = f"Model {i}"
            performance.overall_score = Decimal(str(0.8 - i * 0.1))
            performance.cost_metrics = cost_metrics

            model_performances[f"model_{i}"] = performance

        # Execute
        result = await calculator.calculate_cost_efficiency_metrics(model_performances)

        # Verify
        assert isinstance(result, dict)
        assert len(result) == 3

        for model_id, metrics in result.items():
            assert "efficiency_ratio" in metrics
            assert "cost_adjusted_score" in metrics
            assert "value_score" in metrics
            assert "cost_percentile" in metrics
            assert "efficiency_rank" in metrics

    @pytest.mark.asyncio
    async def test_calculate_cost_efficiency_metrics_no_cost_data(self, calculator):
        """Test cost efficiency calculation with no cost data."""
        # Create mock model performances without cost data
        model_performances = {"model_0": Mock(cost_metrics=None)}

        # Execute
        result = await calculator.calculate_cost_efficiency_metrics(model_performances)

        # Verify empty result
        assert result == {}

    @pytest.mark.asyncio
    async def test_calculate_trend_metrics_success(
        self, calculator, mock_analytics_repository, sample_evaluation_results
    ):
        """Test trend metrics calculation."""
        test_id = uuid4()

        # Setup mock
        mock_analytics_repository.get_evaluation_results.return_value = sample_evaluation_results

        # Execute
        result = await calculator.calculate_trend_metrics(test_id, time_window_hours=24)

        # Verify
        assert isinstance(result, dict)
        assert "overall_trend" in result
        assert "model_trends" in result
        assert "time_window_hours" in result

    @pytest.mark.asyncio
    async def test_calculate_trend_metrics_insufficient_data(
        self, calculator, mock_analytics_repository
    ):
        """Test trend metrics with insufficient data."""
        test_id = uuid4()

        # Setup mock with only one result
        single_result = Mock()
        single_result.completed_at = datetime.utcnow()
        single_result.is_completed.return_value = True
        single_result.overall_score = Decimal("0.8")
        mock_analytics_repository.get_evaluation_results.return_value = [single_result]

        # Execute
        result = await calculator.calculate_trend_metrics(test_id)

        # Verify
        assert result["error"] == "Insufficient data for trend analysis"

    @pytest.mark.asyncio
    async def test_calculate_reliability_metrics_success(
        self, calculator, sample_evaluation_results
    ):
        """Test reliability metrics calculation."""
        # Execute
        result = await calculator.calculate_reliability_metrics(sample_evaluation_results)

        # Verify
        assert isinstance(result, dict)
        assert "completion_rate" in result
        assert "error_rate" in result
        assert "score_coefficient_of_variation" in result
        assert "reliability_score" in result
        assert "consistency_grade" in result

        # Verify values are in expected ranges
        assert 0 <= result["completion_rate"] <= 1
        assert 0 <= result["error_rate"] <= 1
        assert 0 <= result["reliability_score"] <= 1

    @pytest.mark.asyncio
    async def test_calculate_reliability_metrics_empty_results(self, calculator):
        """Test reliability metrics with empty results."""
        result = await calculator.calculate_reliability_metrics([])
        assert result == {}

    def test_group_results_by_model(self, calculator, sample_evaluation_results):
        """Test grouping results by model."""
        result = calculator._group_results_by_model(sample_evaluation_results)

        assert isinstance(result, dict)
        assert len(result) == 3  # Three models

        for model_id, results in result.items():
            assert all(r.model_id == model_id for r in results)

    def test_extract_dimension_scores(self, calculator, sample_evaluation_results):
        """Test dimension score extraction."""
        model_results = [r for r in sample_evaluation_results if r.model_id == "model_0"]

        result = calculator._extract_dimension_scores(model_results)

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "fluency" in result
        assert "relevance" in result

        for dimension, score in result.items():
            assert isinstance(score, Decimal)
            assert 0 <= score <= 1

    def test_calculate_weighted_score(self, calculator):
        """Test weighted score calculation."""
        dimension_scores = {
            "accuracy": Decimal("0.8"),
            "fluency": Decimal("0.7"),
            "relevance": Decimal("0.9"),
        }
        weights = {"accuracy": 0.5, "fluency": 0.3, "relevance": 0.2}

        result = calculator._calculate_weighted_score(dimension_scores, weights)

        # Manual calculation: 0.8*0.5 + 0.7*0.3 + 0.9*0.2 = 0.79
        expected = Decimal("0.79")
        assert abs(result - expected) < Decimal("0.01")

    def test_calculate_simple_average(self, calculator):
        """Test simple average calculation."""
        dimension_scores = {
            "accuracy": Decimal("0.8"),
            "fluency": Decimal("0.6"),
            "relevance": Decimal("0.9"),
        }

        result = calculator._calculate_simple_average(dimension_scores)

        # Manual calculation: (0.8 + 0.6 + 0.9) / 3 = 0.7667
        expected = Decimal("2.3") / Decimal("3")
        assert abs(result - expected) < Decimal("0.01")

    def test_calculate_confidence_interval(self, calculator):
        """Test confidence interval calculation."""
        values = [0.7, 0.8, 0.75, 0.85, 0.9, 0.65, 0.77, 0.82]

        with patch("scipy.stats.t") as mock_t:
            mock_t.ppf.return_value = 2.365  # t-critical for 7 df, 95% CI

            lower, upper = calculator._calculate_confidence_interval(values, 0.95)

            assert isinstance(lower, Decimal)
            assert isinstance(upper, Decimal)
            assert lower < upper

    def test_calculate_consistency_score(self, calculator, sample_evaluation_results):
        """Test consistency score calculation."""
        model_results = [r for r in sample_evaluation_results if r.model_id == "model_0"]

        result = calculator._calculate_consistency_score(model_results)

        assert isinstance(result, Decimal)
        assert 0 <= result <= 1

    def test_remove_outliers(self, calculator):
        """Test outlier removal."""
        values = [1.0, 1.1, 1.2, 1.15, 1.05, 5.0, 1.08]  # 5.0 is outlier

        cleaned_values, outliers_count = calculator._remove_outliers(values, threshold=2.0)

        assert len(cleaned_values) == 6  # Should remove one outlier
        assert outliers_count == 1
        assert 5.0 not in cleaned_values

    def test_calculate_percentile(self, calculator):
        """Test percentile calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Test 50th percentile (should be around 50%)
        percentile = calculator._calculate_percentile(5, values)
        assert 40 <= percentile <= 60

        # Test 100th percentile
        percentile = calculator._calculate_percentile(10, values)
        assert percentile == 90  # (9/10) * 100

    def test_calculate_percentile_value(self, calculator):
        """Test percentile value calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # 50th percentile should be around 5
        value = calculator._calculate_percentile_value(values, 50)
        assert 4 <= value <= 6

        # 90th percentile should be around 9
        value = calculator._calculate_percentile_value(values, 90)
        assert 8 <= value <= 10

    def test_has_dimension_scores(self, calculator, sample_evaluation_results):
        """Test dimension scores detection."""
        result = calculator._has_dimension_scores(sample_evaluation_results)
        assert result is True

        # Test with results without dimension scores
        empty_results = [Mock(dimension_scores={})]
        result = calculator._has_dimension_scores(empty_results)
        assert result is False

    def test_grade_consistency(self, calculator):
        """Test consistency grading."""
        assert calculator._grade_consistency(0.03) == "Excellent"
        assert calculator._grade_consistency(0.08) == "Good"
        assert calculator._grade_consistency(0.15) == "Fair"
        assert calculator._grade_consistency(0.25) == "Poor"

    def test_metric_calculation_config_validation(self):
        """Test metric calculation config validation."""
        # Valid config
        config = MetricCalculationConfig(confidence_level=0.95, minimum_sample_size=30)
        assert config.confidence_level == 0.95
        assert config.minimum_sample_size == 30

        # Test defaults
        default_config = MetricCalculationConfig()
        assert default_config.include_confidence_intervals is True
        assert default_config.outlier_detection is True
        assert default_config.weight_by_sample_size is True
