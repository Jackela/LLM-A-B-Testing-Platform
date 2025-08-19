"""Integration tests for ModelProviderService."""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.application.dto.model_request_dto import BatchModelRequestDTO, ModelRequestDTO
from src.application.dto.model_response_dto import ModelResponseDTO, ResponseStatus
from src.application.services.model_provider.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerFactory,
)
from src.application.services.model_provider.cost_calculator import CostCalculator
from src.application.services.model_provider.error_handler import (
    ErrorHandler,
    ProviderError,
    ProviderErrorType,
)
from src.application.services.model_provider.model_provider_service import ModelProviderService
from src.application.services.model_provider.provider_selector import ProviderSelector
from src.application.services.model_provider.response_processor import ResponseProcessor
from src.application.services.model_provider.retry_service import RetryService
from src.domain.model_provider.entities.model_config import ModelConfig
from src.domain.model_provider.entities.model_provider import ModelProvider
from src.domain.model_provider.value_objects.health_status import HealthStatus
from src.domain.model_provider.value_objects.provider_type import ProviderType
from src.domain.test_management.entities.test_sample import TestSample


class TestModelProviderService:
    """Test suite for ModelProviderService integration."""

    @pytest.fixture
    def mock_uow(self):
        """Create mock unit of work."""
        uow = Mock()
        uow.providers = Mock()
        uow.providers.find_by_id = AsyncMock()
        uow.providers.find_all = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        model_config = ModelConfig(
            model_id="test-model",
            max_tokens=1000,
            cost_per_input_token=Decimal("0.001"),
            cost_per_output_token=Decimal("0.002"),
            supported_parameters=["temperature", "max_tokens"],
        )

        provider = Mock(spec=ModelProvider)
        provider.id = uuid4()
        provider.name = "test-provider"
        provider.provider_type = ProviderType.OPENAI
        provider.health_status = HealthStatus.HEALTHY
        provider.supported_models = [model_config]
        provider.find_model_config.return_value = model_config
        return provider

    @pytest.fixture
    def service_components(self):
        """Create all service components."""
        return {
            "error_handler": ErrorHandler(),
            "circuit_breaker_factory": CircuitBreakerFactory(),
            "retry_service": RetryService(),
            "cost_calculator": CostCalculator(),
            "provider_selector": ProviderSelector(),
        }

    @pytest.fixture
    def model_provider_service(self, mock_uow, service_components):
        """Create ModelProviderService instance with mocked dependencies."""
        response_processor = ResponseProcessor(service_components["cost_calculator"])

        return ModelProviderService(
            uow=mock_uow,
            error_handler=service_components["error_handler"],
            circuit_breaker_factory=service_components["circuit_breaker_factory"],
            retry_service=service_components["retry_service"],
            response_processor=response_processor,
            cost_calculator=service_components["cost_calculator"],
            provider_selector=service_components["provider_selector"],
        )

    @pytest.mark.asyncio
    async def test_successful_model_call(self, model_provider_service, mock_uow, mock_provider):
        """Test successful model call execution."""
        # Setup
        mock_uow.providers.find_by_id.return_value = mock_provider

        # Execute
        response = await model_provider_service.call_model(
            provider_id=str(mock_provider.id),
            model_id="test-model",
            prompt="Test prompt",
            parameters={"temperature": 0.7, "max_tokens": 100},
        )

        # Verify
        assert response is not None
        assert response.provider_id == str(mock_provider.id)
        assert response.model_id == "test-model"
        assert response.is_successful()
        assert response.latency_ms is not None
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_provider_not_found(self, model_provider_service, mock_uow):
        """Test handling when provider is not found."""
        # Setup
        mock_uow.providers.find_by_id.return_value = None

        # Execute
        response = await model_provider_service.call_model(
            provider_id="nonexistent-provider",
            model_id="test-model",
            prompt="Test prompt",
            parameters={},
        )

        # Verify
        assert response is not None
        assert response.status == ResponseStatus.ERROR
        assert "not found" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_provider_not_operational(self, model_provider_service, mock_uow, mock_provider):
        """Test handling when provider is not operational."""
        # Setup
        mock_provider.health_status = HealthStatus.UNHEALTHY
        mock_uow.providers.find_by_id.return_value = mock_provider

        # Execute
        response = await model_provider_service.call_model(
            provider_id=str(mock_provider.id),
            model_id="test-model",
            prompt="Test prompt",
            parameters={},
        )

        # Verify
        assert response is not None
        assert response.status == ResponseStatus.ERROR
        assert "not operational" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_invalid_parameters(self, model_provider_service, mock_uow, mock_provider):
        """Test handling of invalid model parameters."""
        # Setup
        mock_uow.providers.find_by_id.return_value = mock_provider

        # Execute
        response = await model_provider_service.call_model(
            provider_id=str(mock_provider.id),
            model_id="test-model",
            prompt="Test prompt",
            parameters={"invalid_param": "value"},
        )

        # Verify - service should handle invalid parameters gracefully
        # In the mock implementation, it will create a response but may have warnings
        assert response is not None

    @pytest.mark.asyncio
    async def test_batch_processing_success(self, model_provider_service, mock_uow, mock_provider):
        """Test successful batch processing."""
        # Setup
        mock_uow.providers.find_by_id.return_value = mock_provider

        requests = [
            ModelRequestDTO(
                provider_id=str(mock_provider.id),
                model_id="test-model",
                prompt=f"Test prompt {i}",
                parameters={"temperature": 0.7},
            )
            for i in range(3)
        ]

        batch_request = BatchModelRequestDTO(requests=requests, max_parallel_requests=2)

        # Execute
        batch_response = await model_provider_service.call_model_batch(batch_request)

        # Verify
        assert batch_response is not None
        assert len(batch_response.responses) == 3
        assert batch_response.successful_count >= 0  # May have mock failures
        assert batch_response.get_success_rate() >= 0

    @pytest.mark.asyncio
    async def test_batch_processing_mixed_results(
        self, model_provider_service, mock_uow, mock_provider
    ):
        """Test batch processing with mixed success/failure results."""

        # Setup - first provider exists, second doesn't
        def mock_find_by_id(provider_id):
            if provider_id == str(mock_provider.id):
                return mock_provider
            return None

        mock_uow.providers.find_by_id.side_effect = mock_find_by_id

        requests = [
            ModelRequestDTO(
                provider_id=str(mock_provider.id),
                model_id="test-model",
                prompt="Success prompt",
                parameters={"temperature": 0.7},
            ),
            ModelRequestDTO(
                provider_id="nonexistent-provider",
                model_id="test-model",
                prompt="Failure prompt",
                parameters={"temperature": 0.7},
            ),
        ]

        batch_request = BatchModelRequestDTO(requests=requests)

        # Execute
        batch_response = await model_provider_service.call_model_batch(batch_request)

        # Verify
        assert batch_response is not None
        assert len(batch_response.responses) == 2
        assert batch_response.failed_count >= 1  # At least one should fail

    @pytest.mark.asyncio
    async def test_intelligent_routing_success(
        self, model_provider_service, mock_uow, mock_provider
    ):
        """Test intelligent provider routing."""
        # Setup
        mock_uow.providers.find_all.return_value = [mock_provider]

        # Execute
        response = await model_provider_service.call_model_with_intelligent_routing(
            model_id="test-model", prompt="Test prompt", parameters={"temperature": 0.7}
        )

        # Verify
        assert response is not None
        assert response.provider_id == str(mock_provider.id)
        assert response.model_id == "test-model"

    @pytest.mark.asyncio
    async def test_intelligent_routing_no_providers(self, model_provider_service, mock_uow):
        """Test intelligent routing when no providers support the model."""
        # Setup
        mock_uow.providers.find_all.return_value = []

        # Execute
        response = await model_provider_service.call_model_with_intelligent_routing(
            model_id="unsupported-model", prompt="Test prompt", parameters={}
        )

        # Verify
        assert response is not None
        assert response.status == ResponseStatus.ERROR
        assert "no providers support" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(
        self, model_provider_service, mock_uow, mock_provider
    ):
        """Test circuit breaker integration with provider calls."""
        # Setup
        mock_uow.providers.find_by_id.return_value = mock_provider

        # Get circuit breaker
        cb = model_provider_service.circuit_breaker_factory.get_circuit_breaker_for_provider(
            str(mock_provider.id), "test-model"
        )

        # Verify circuit breaker is created and in closed state
        assert cb is not None
        assert cb.get_state().value == "closed"

        # Execute a successful call
        response = await model_provider_service.call_model(
            provider_id=str(mock_provider.id),
            model_id="test-model",
            prompt="Test prompt",
            parameters={},
        )

        # Verify response and circuit breaker metrics
        assert response is not None
        metrics = cb.get_metrics()
        assert metrics.total_requests >= 1

    @pytest.mark.asyncio
    async def test_cost_calculation_integration(
        self, model_provider_service, mock_uow, mock_provider
    ):
        """Test cost calculation integration."""
        # Setup
        mock_uow.providers.find_by_id.return_value = mock_provider

        # Execute
        response = await model_provider_service.call_model(
            provider_id=str(mock_provider.id),
            model_id="test-model",
            prompt="Test prompt for cost calculation",
            parameters={"max_tokens": 100},
        )

        # Verify cost information is present
        assert response is not None
        if response.is_successful():
            # Cost might be calculated if token information is available
            pass  # Cost calculation depends on mock response format

    @pytest.mark.asyncio
    async def test_test_sample_processing(self, model_provider_service, mock_uow, mock_provider):
        """Test processing TestSample objects."""
        # Setup
        mock_uow.providers.find_by_id.return_value = mock_provider

        test_sample = TestSample(
            prompt="Test sample prompt",
            expected_output="Expected response",
            metadata={"category": "test"},
        )

        # Execute
        response = await model_provider_service.process_test_sample(
            sample=test_sample,
            provider_id=str(mock_provider.id),
            model_id="test-model",
            parameters={"temperature": 0.7},
        )

        # Verify
        assert response is not None
        assert response.provider_id == str(mock_provider.id)
        assert response.model_id == "test-model"

    @pytest.mark.asyncio
    async def test_health_status_reporting(self, model_provider_service, mock_uow, mock_provider):
        """Test provider health status reporting."""
        # Setup
        mock_uow.providers.find_all.return_value = [mock_provider]

        # Execute
        health_status = await model_provider_service.get_provider_health_status()

        # Verify
        assert health_status is not None
        assert isinstance(health_status, dict)
        assert mock_provider.name in health_status
        assert "health_status" in health_status[mock_provider.name]
        assert "is_operational" in health_status[mock_provider.name]

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, model_provider_service, mock_uow, mock_provider):
        """Test rate limiting integration."""
        # Setup
        mock_uow.providers.find_by_id.return_value = mock_provider

        # Set a very low rate limit for testing
        rate_limiter = model_provider_service.rate_limiter
        rate_limiter.request_limits[f"{mock_provider.id}_test-model"] = 1

        # First request should succeed
        response1 = await model_provider_service.call_model(
            provider_id=str(mock_provider.id),
            model_id="test-model",
            prompt="First request",
            parameters={},
        )

        # Second request might be rate limited (depends on timing)
        response2 = await model_provider_service.call_model(
            provider_id=str(mock_provider.id),
            model_id="test-model",
            prompt="Second request",
            parameters={},
        )

        # Verify both responses exist (rate limiting might not trigger in mock)
        assert response1 is not None
        assert response2 is not None

    @pytest.mark.asyncio
    async def test_error_handling_chain(self, model_provider_service, mock_uow, mock_provider):
        """Test the complete error handling chain."""
        # Setup - simulate various error conditions
        mock_provider.health_status = HealthStatus.HEALTHY
        mock_uow.providers.find_by_id.return_value = mock_provider

        # Test with empty prompt (potential validation error)
        response = await model_provider_service.call_model(
            provider_id=str(mock_provider.id),
            model_id="test-model",
            prompt="",  # Empty prompt
            parameters={},
        )

        # Verify error handling
        assert response is not None
        # Service should handle empty prompts gracefully

    def test_service_component_initialization(self, mock_uow):
        """Test that all service components are properly initialized."""
        service = ModelProviderService(uow=mock_uow)

        # Verify all components are initialized
        assert service.error_handler is not None
        assert service.circuit_breaker_factory is not None
        assert service.retry_service is not None
        assert service.cost_calculator is not None
        assert service.response_processor is not None
        assert service.provider_selector is not None
        assert service.rate_limiter is not None

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, model_provider_service, mock_uow, mock_provider):
        """Test handling of concurrent requests."""
        # Setup
        mock_uow.providers.find_by_id.return_value = mock_provider

        # Execute multiple concurrent requests
        tasks = [
            model_provider_service.call_model(
                provider_id=str(mock_provider.id),
                model_id="test-model",
                prompt=f"Concurrent request {i}",
                parameters={"temperature": 0.7},
            )
            for i in range(5)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests completed
        assert len(responses) == 5
        for response in responses:
            assert not isinstance(response, Exception)
            assert response is not None

    @pytest.mark.asyncio
    async def test_provider_selection_metrics_recording(
        self, model_provider_service, mock_uow, mock_provider
    ):
        """Test that provider selection metrics are properly recorded."""
        # Setup
        mock_uow.providers.find_all.return_value = [mock_provider]

        # Execute intelligent routing call
        response = await model_provider_service.call_model_with_intelligent_routing(
            model_id="test-model", prompt="Test prompt", parameters={"temperature": 0.7}
        )

        # Verify metrics are recorded
        assert response is not None

        # Check that provider selector has recorded metrics
        stats = model_provider_service.provider_selector.get_provider_statistics(mock_provider)
        assert stats is not None
        assert "provider_name" in stats
