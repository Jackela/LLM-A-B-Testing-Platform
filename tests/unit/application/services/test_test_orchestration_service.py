"""Unit tests for TestOrchestrationService."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.application.services.test_orchestration_service import TestOrchestrationService
from src.domain.model_provider.value_objects.money import Money
from src.domain.test_management.value_objects.test_status import TestStatus
from tests.factories import ModelConfigFactory, ModelResponseFactory, TestFactory, TestSampleFactory


class TestTestOrchestrationService:
    """Unit tests for TestOrchestrationService."""

    @pytest.fixture
    def mock_uow(self):
        """Mock unit of work."""
        uow = AsyncMock()
        uow.tests = AsyncMock()
        uow.providers = AsyncMock()
        uow.evaluations = AsyncMock()
        return uow

    @pytest.fixture
    def mock_model_service(self):
        """Mock model service."""
        service = AsyncMock()
        service.generate_response.return_value = ModelResponseFactory()
        return service

    @pytest.fixture
    def mock_evaluation_service(self):
        """Mock evaluation service."""
        service = AsyncMock()
        service.evaluate_response.return_value = {"score": 8.5, "reasoning": "Good response"}
        return service

    @pytest.fixture
    def orchestration_service(self, mock_uow, mock_model_service, mock_evaluation_service):
        """Create orchestration service instance."""
        return TestOrchestrationService(
            uow=mock_uow,
            model_service=mock_model_service,
            evaluation_service=mock_evaluation_service,
        )

    @pytest.fixture
    def running_test(self):
        """Create a running test with samples."""
        test = TestFactory()
        test.status = TestStatus.RUNNING
        test.samples = [TestSampleFactory(test_id=test.id) for _ in range(10)]
        test.configuration.model_configs = [ModelConfigFactory() for _ in range(2)]
        return test

    @pytest.mark.asyncio
    async def test_start_test_execution(self, orchestration_service, mock_uow, running_test):
        """Test successful test execution start."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = running_test

        # Act
        result = await orchestration_service.start_test_execution(running_test.id)

        # Assert
        assert result is True
        mock_uow.tests.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_sample_batch(
        self, orchestration_service, mock_model_service, running_test
    ):
        """Test processing of sample batch."""
        # Arrange
        samples = running_test.samples[:5]
        model_config = running_test.configuration.model_configs[0]

        # Act
        responses = await orchestration_service.process_sample_batch(samples, model_config)

        # Assert
        assert len(responses) == len(samples)
        assert mock_model_service.generate_response.call_count == len(samples)

    @pytest.mark.asyncio
    async def test_parallel_model_execution(
        self, orchestration_service, mock_model_service, running_test
    ):
        """Test parallel execution across multiple models."""
        # Arrange
        sample = running_test.samples[0]
        model_configs = running_test.configuration.model_configs

        # Act
        responses = await orchestration_service.execute_models_for_sample(sample, model_configs)

        # Assert
        assert len(responses) == len(model_configs)
        assert mock_model_service.generate_response.call_count == len(model_configs)

    @pytest.mark.asyncio
    async def test_error_handling_in_batch_processing(
        self, orchestration_service, mock_model_service, running_test
    ):
        """Test error handling during batch processing."""
        # Arrange
        samples = running_test.samples[:3]
        model_config = running_test.configuration.model_configs[0]

        # Make second call fail
        mock_model_service.generate_response.side_effect = [
            ModelResponseFactory(),
            Exception("Model service error"),
            ModelResponseFactory(),
        ]

        # Act
        responses = await orchestration_service.process_sample_batch(samples, model_config)

        # Assert
        assert len(responses) == 2  # Only successful responses
        assert mock_model_service.generate_response.call_count == 3

    @pytest.mark.asyncio
    async def test_rate_limit_handling(
        self, orchestration_service, mock_model_service, running_test
    ):
        """Test rate limit handling."""
        # Arrange
        samples = running_test.samples
        model_config = running_test.configuration.model_configs[0]

        # Simulate rate limit exceeded
        from src.domain.model_provider.exceptions import RateLimitExceededError

        mock_model_service.generate_response.side_effect = RateLimitExceededError(
            "Rate limit exceeded"
        )

        # Act
        with patch("asyncio.sleep", return_value=None):  # Mock sleep to speed up test
            responses = await orchestration_service.process_sample_batch(samples[:1], model_config)

        # Assert
        assert len(responses) == 0  # No successful responses due to rate limit

    @pytest.mark.asyncio
    async def test_cost_tracking(self, orchestration_service, mock_model_service, running_test):
        """Test cost tracking during execution."""
        # Arrange
        sample = running_test.samples[0]
        model_config = running_test.configuration.model_configs[0]

        response = ModelResponseFactory()
        response.cost = Money(0.50, "USD")
        mock_model_service.generate_response.return_value = response

        # Act
        responses = await orchestration_service.execute_models_for_sample(sample, [model_config])

        # Assert
        assert len(responses) == 1
        assert responses[0].cost.amount == 0.50

    @pytest.mark.asyncio
    async def test_timeout_handling(self, orchestration_service, mock_model_service, running_test):
        """Test timeout handling for slow responses."""
        # Arrange
        sample = running_test.samples[0]
        model_config = running_test.configuration.model_configs[0]

        # Simulate timeout
        import asyncio

        mock_model_service.generate_response.side_effect = asyncio.TimeoutError("Request timeout")

        # Act
        responses = await orchestration_service.execute_models_for_sample(sample, [model_config])

        # Assert
        assert len(responses) == 0  # No responses due to timeout

    @pytest.mark.asyncio
    async def test_max_cost_enforcement(
        self, orchestration_service, mock_model_service, running_test
    ):
        """Test enforcement of maximum cost limits."""
        # Arrange
        running_test.configuration.max_cost = Money(1.0, "USD")
        sample = running_test.samples[0]
        model_config = running_test.configuration.model_configs[0]

        # High cost response
        response = ModelResponseFactory()
        response.cost = Money(2.0, "USD")
        mock_model_service.generate_response.return_value = response

        # Act
        with pytest.raises(Exception, match="Cost limit exceeded"):
            await orchestration_service.execute_models_for_sample(sample, [model_config])

    @pytest.mark.asyncio
    async def test_concurrent_sample_processing(
        self, orchestration_service, mock_model_service, running_test
    ):
        """Test concurrent processing of multiple samples."""
        # Arrange
        samples = running_test.samples[:5]
        model_config = running_test.configuration.model_configs[0]

        # Act
        with patch.object(orchestration_service, "max_concurrent_requests", 3):
            responses = await orchestration_service.process_sample_batch(samples, model_config)

        # Assert
        assert len(responses) == len(samples)
        assert mock_model_service.generate_response.call_count == len(samples)

    @pytest.mark.asyncio
    async def test_model_unavailable_handling(
        self, orchestration_service, mock_model_service, running_test
    ):
        """Test handling of unavailable models."""
        # Arrange
        sample = running_test.samples[0]
        model_config = running_test.configuration.model_configs[0]

        from src.domain.model_provider.exceptions import ModelUnavailableError

        mock_model_service.generate_response.side_effect = ModelUnavailableError("Model offline")

        # Act
        responses = await orchestration_service.execute_models_for_sample(sample, [model_config])

        # Assert
        assert len(responses) == 0

    @pytest.mark.asyncio
    async def test_progress_tracking(self, orchestration_service, mock_uow, running_test):
        """Test progress tracking during execution."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = running_test

        # Act
        await orchestration_service.start_test_execution(running_test.id)

        # Assert
        # Verify that test progress is updated
        assert mock_uow.tests.save.called
        saved_test = mock_uow.tests.save.call_args[0][0]
        assert hasattr(saved_test, "updated_at")

    @pytest.mark.asyncio
    async def test_evaluation_integration(
        self, orchestration_service, mock_evaluation_service, running_test
    ):
        """Test integration with evaluation service."""
        # Arrange
        response = ModelResponseFactory()
        sample = running_test.samples[0]

        # Act
        evaluation = await orchestration_service.evaluate_response(response, sample)

        # Assert
        assert evaluation is not None
        mock_evaluation_service.evaluate_response.assert_called_once_with(response, sample)

    @pytest.mark.asyncio
    async def test_batch_size_optimization(self, orchestration_service, running_test):
        """Test automatic batch size optimization."""
        # Arrange
        large_sample_set = [TestSampleFactory() for _ in range(100)]
        model_config = running_test.configuration.model_configs[0]

        # Act
        with patch.object(orchestration_service, "optimal_batch_size", return_value=10):
            batches = orchestration_service.create_sample_batches(large_sample_set, model_config)

        # Assert
        assert len(batches) == 10  # 100 samples / 10 per batch
        assert all(len(batch) == 10 for batch in batches)

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, orchestration_service, mock_model_service, running_test):
        """Test retry mechanism for failed requests."""
        # Arrange
        sample = running_test.samples[0]
        model_config = running_test.configuration.model_configs[0]

        # First call fails, second succeeds
        mock_model_service.generate_response.side_effect = [
            Exception("Temporary failure"),
            ModelResponseFactory(),
        ]

        # Act
        with patch.object(orchestration_service, "max_retries", 2):
            responses = await orchestration_service.execute_models_for_sample(
                sample, [model_config]
            )

        # Assert
        assert len(responses) == 1  # Should succeed on retry
        assert mock_model_service.generate_response.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_on_failure(self, orchestration_service, mock_uow, running_test):
        """Test cleanup operations when execution fails."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = running_test
        mock_uow.tests.save.side_effect = Exception("Database error")

        # Act
        with pytest.raises(Exception, match="Database error"):
            await orchestration_service.start_test_execution(running_test.id)

        # Assert
        # Verify cleanup was attempted
        assert running_test.status == TestStatus.RUNNING  # Status should be preserved for retry

    @pytest.mark.asyncio
    async def test_memory_optimization(self, orchestration_service, running_test):
        """Test memory optimization for large datasets."""
        # Arrange
        large_sample_set = [TestSampleFactory() for _ in range(1000)]
        running_test.samples = large_sample_set

        # Act
        with patch.object(orchestration_service, "process_in_chunks", return_value=True):
            result = await orchestration_service.start_test_execution(running_test.id)

        # Assert
        assert result is True
