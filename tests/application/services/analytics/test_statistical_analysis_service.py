"""Tests for Statistical Analysis Service."""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.application.services.analytics.statistical_analysis_service import (
    AnalysisConfig,
    ComprehensiveAnalysisResult,
    StatisticalAnalysisService,
)
from src.domain.analytics.entities.analysis_result import AnalysisResult, ModelPerformanceMetrics
from src.domain.analytics.exceptions import InsufficientDataError, ValidationError
from src.domain.analytics.value_objects.cost_data import CostData
from src.domain.analytics.value_objects.test_result import TestResult
from src.domain.evaluation.entities.evaluation_result import EvaluationResult


class TestStatisticalAnalysisService:
    """Test suite for Statistical Analysis Service."""

    @pytest.fixture
    def mock_repositories(self):
        """Create mock repositories."""
        test_repository = AsyncMock()
        analytics_repository = AsyncMock()
        significance_tester = Mock()
        data_aggregator = AsyncMock()
        insight_generator = AsyncMock()
        significance_analyzer = Mock()

        return {
            "test_repository": test_repository,
            "analytics_repository": analytics_repository,
            "significance_tester": significance_tester,
            "data_aggregator": data_aggregator,
            "insight_generator": insight_generator,
            "significance_analyzer": significance_analyzer,
        }

    @pytest.fixture
    def service(self, mock_repositories):
        """Create service instance with mocked dependencies."""
        return StatisticalAnalysisService(**mock_repositories)

    @pytest.fixture
    def sample_test(self):
        """Create sample test entity."""
        test = Mock()
        test.test_id = uuid4()
        test.name = "Sample Test"
        test.description = "Test description"
        return test

    @pytest.fixture
    def sample_evaluation_results(self):
        """Create sample evaluation results."""
        results = []
        for i in range(10):
            result = Mock(spec=EvaluationResult)
            result.model_id = f"model_{i % 2}"  # Two models
            result.overall_score = Decimal(str(0.7 + (i % 3) * 0.1))
            result.dimension_scores = {
                "accuracy": Decimal(str(0.8 + (i % 2) * 0.1)),
                "fluency": Decimal(str(0.75 + (i % 3) * 0.05)),
            }
            result.is_completed.return_value = True
            result.has_error.return_value = False
            result.metadata = {"model_name": f"Model {i % 2}", "cost": 0.01}
            results.append(result)
        return results

    @pytest.mark.asyncio
    async def test_analyze_test_results_success(
        self, service, mock_repositories, sample_test, sample_evaluation_results
    ):
        """Test successful analysis of test results."""
        test_id = uuid4()
        config = AnalysisConfig()

        # Setup mocks
        mock_repositories["test_repository"].find_by_id.return_value = sample_test
        mock_repositories["analytics_repository"].get_evaluation_results.return_value = (
            sample_evaluation_results
        )
        mock_repositories["significance_tester"].test_multiple_models.return_value = {
            "model_0_vs_model_1": Mock(
                test_type="ttest_independent",
                p_value=Decimal("0.05"),
                effect_size=Decimal("0.3"),
                is_significant=Mock(return_value=True),
            )
        }
        mock_repositories["data_aggregator"].aggregate_by_model.return_value = []
        mock_repositories["data_aggregator"].aggregate_by_difficulty.return_value = []
        mock_repositories["data_aggregator"].aggregate_by_dimension.return_value = []
        mock_repositories["insight_generator"].generate_comprehensive_insights.return_value = []

        # Execute
        result = await service.analyze_test_results(test_id, config)

        # Verify
        assert isinstance(result, ComprehensiveAnalysisResult)
        assert result.test_id == test_id
        assert len(result.statistical_tests) >= 0
        assert len(result.model_performances) == 2  # Two models
        assert result.processing_time_ms > 0

        # Verify repository calls
        mock_repositories["test_repository"].find_by_id.assert_called_once_with(test_id)
        mock_repositories["analytics_repository"].get_evaluation_results.assert_called_once_with(
            test_id
        )

    @pytest.mark.asyncio
    async def test_analyze_test_results_test_not_found(self, service, mock_repositories):
        """Test analysis with non-existent test."""
        test_id = uuid4()

        # Setup mock to return None
        mock_repositories["test_repository"].find_by_id.return_value = None

        # Execute and verify exception
        with pytest.raises(ValidationError, match="Test .* not found"):
            await service.analyze_test_results(test_id)

    @pytest.mark.asyncio
    async def test_analyze_test_results_insufficient_data(
        self, service, mock_repositories, sample_test
    ):
        """Test analysis with insufficient data."""
        test_id = uuid4()

        # Setup mocks with insufficient data
        mock_repositories["test_repository"].find_by_id.return_value = sample_test
        mock_repositories["analytics_repository"].get_evaluation_results.return_value = []

        # Execute and verify exception
        with pytest.raises(InsufficientDataError, match="No evaluation results found"):
            await service.analyze_test_results(test_id)

    @pytest.mark.asyncio
    async def test_analyze_model_comparison_success(
        self, service, mock_repositories, sample_evaluation_results
    ):
        """Test successful model comparison analysis."""
        test_id = uuid4()
        model_ids = ["model_0", "model_1"]

        # Setup mocks
        mock_repositories["analytics_repository"].get_evaluation_results.return_value = (
            sample_evaluation_results
        )
        mock_repositories["significance_tester"].test_multiple_models.return_value = {
            "model_0_vs_model_1": Mock(
                test_type="ttest_independent", p_value=Decimal("0.03"), effect_size=Decimal("0.5")
            )
        }

        # Execute
        result = await service.analyze_model_comparison(test_id, model_ids)

        # Verify
        assert isinstance(result, dict)
        assert "model_0_vs_model_1" in result
        mock_repositories["analytics_repository"].get_evaluation_results.assert_called_once_with(
            test_id, model_ids
        )

    @pytest.mark.asyncio
    async def test_analyze_model_comparison_insufficient_models(self, service):
        """Test model comparison with insufficient models."""
        test_id = uuid4()
        model_ids = ["model_0"]  # Only one model

        # Execute and verify exception
        with pytest.raises(ValidationError, match="At least 2 models required"):
            await service.analyze_model_comparison(test_id, model_ids)

    @pytest.mark.asyncio
    async def test_analyze_dimension_performance_success(
        self, service, mock_repositories, sample_evaluation_results
    ):
        """Test successful dimension performance analysis."""
        test_id = uuid4()
        dimensions = ["accuracy", "fluency"]

        # Setup mocks
        mock_repositories["analytics_repository"].get_evaluation_results.return_value = (
            sample_evaluation_results
        )
        mock_repositories["significance_tester"].test_dimension_differences.return_value = {
            "accuracy_vs_fluency": Mock(
                test_type="ttest_paired", p_value=Decimal("0.02"), effect_size=Decimal("0.4")
            )
        }

        # Execute
        result = await service.analyze_dimension_performance(test_id, dimensions)

        # Verify
        assert isinstance(result, dict)
        assert "accuracy_vs_fluency" in result
        mock_repositories["significance_tester"].test_dimension_differences.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_dimension_performance_no_dimensions(self, service):
        """Test dimension analysis with no dimensions."""
        test_id = uuid4()
        dimensions = []

        # Execute and verify exception
        with pytest.raises(ValidationError, match="At least one dimension required"):
            await service.analyze_dimension_performance(test_id, dimensions)

    @pytest.mark.asyncio
    async def test_calculate_required_sample_sizes_success(self, service, mock_repositories):
        """Test sample size calculation."""
        effect_sizes = {"test_1": 0.5, "test_2": 0.3}

        # Setup mock
        mock_repositories["significance_tester"].calculate_required_sample_size.side_effect = [
            50,
            100,
        ]

        # Execute
        result = await service.calculate_required_sample_sizes(effect_sizes)

        # Verify
        assert isinstance(result, dict)
        assert len(result) == 2
        assert result["test_1"] == 50
        assert result["test_2"] == 100

    def test_analysis_config_validation(self):
        """Test analysis configuration validation."""
        # Valid configuration
        config = AnalysisConfig(confidence_level=0.95, minimum_sample_size=30)
        assert config.confidence_level == 0.95
        assert config.minimum_sample_size == 30

        # Test default values
        default_config = AnalysisConfig()
        assert default_config.confidence_level == 0.95
        assert default_config.include_effect_sizes is True
        assert default_config.enable_dimension_analysis is True

    @pytest.mark.asyncio
    async def test_load_and_validate_test_valid(self, service, mock_repositories, sample_test):
        """Test loading and validating a valid test."""
        test_id = uuid4()
        mock_repositories["test_repository"].find_by_id.return_value = sample_test

        result = await service._load_and_validate_test(test_id)

        assert result == sample_test
        mock_repositories["test_repository"].find_by_id.assert_called_once_with(test_id)

    @pytest.mark.asyncio
    async def test_load_and_validate_test_not_found(self, service, mock_repositories):
        """Test loading a non-existent test."""
        test_id = uuid4()
        mock_repositories["test_repository"].find_by_id.return_value = None

        with pytest.raises(ValidationError, match="Test .* not found"):
            await service._load_and_validate_test(test_id)

    def test_validate_sufficient_data_success(self, service, sample_evaluation_results):
        """Test data validation with sufficient data."""
        config = AnalysisConfig(minimum_sample_size=5)

        # Should not raise an exception
        service._validate_sufficient_data(sample_evaluation_results, config)

    def test_validate_sufficient_data_insufficient(self, service, sample_evaluation_results):
        """Test data validation with insufficient data."""
        config = AnalysisConfig(minimum_sample_size=100)  # Require more than available

        with pytest.raises(InsufficientDataError, match="Insufficient data for models"):
            service._validate_sufficient_data(sample_evaluation_results, config)

    def test_validate_sufficient_data_empty(self, service):
        """Test data validation with empty data."""
        config = AnalysisConfig()

        with pytest.raises(InsufficientDataError, match="No evaluation results found"):
            service._validate_sufficient_data([], config)

    def test_interpret_effect_size(self, service):
        """Test effect size interpretation."""
        assert service._interpret_effect_size(Decimal("0.1")) == "negligible"
        assert service._interpret_effect_size(Decimal("0.3")) == "small"
        assert service._interpret_effect_size(Decimal("0.6")) == "medium"
        assert service._interpret_effect_size(Decimal("0.9")) == "large"

    def test_assess_practical_significance(self, service):
        """Test practical significance assessment."""
        config = AnalysisConfig()

        # Mock test result
        test_result = Mock()
        test_result.is_significant.return_value = True
        test_result.effect_size = Decimal("0.3")  # Small effect

        result = service._assess_practical_significance(test_result, config)

        assert isinstance(result, dict)
        assert "is_statistically_significant" in result
        assert "is_practically_significant" in result
        assert "recommendation" in result

    def test_get_practical_recommendation(self, service):
        """Test practical recommendation generation."""
        # Strong evidence case
        rec1 = service._get_practical_recommendation(True, True, 0.8)
        assert "Strong evidence" in rec1

        # Statistical but not practical
        rec2 = service._get_practical_recommendation(True, False, 0.1)
        assert "consider context" in rec2

        # Practical but not statistical
        rec3 = service._get_practical_recommendation(False, True, 0.8)
        assert "collect more data" in rec3

        # Neither
        rec4 = service._get_practical_recommendation(False, False, 0.1)
        assert "no action needed" in rec4
