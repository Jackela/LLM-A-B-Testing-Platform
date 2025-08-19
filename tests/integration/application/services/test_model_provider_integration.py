"""Integration tests for model provider services."""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.application.services.model_provider.circuit_breaker import CircuitBreaker
from src.application.services.model_provider.cost_calculator import CostCalculator
from src.application.services.model_provider.model_inference_service import ModelInferenceService
from src.application.services.model_provider_service import ModelProviderService
from src.domain.model_provider.exceptions import (
    InsufficientFundsError,
    ModelUnavailableError,
    RateLimitExceededError,
)
from src.domain.model_provider.value_objects.money import Money
from src.domain.model_provider.value_objects.rate_limits import RateLimits
from tests.factories import ModelConfigFactory, ModelProviderFactory


@pytest.mark.integration
class TestModelProviderIntegration:
    """Integration tests for model provider services."""

    @pytest.fixture
    def mock_http_client(self):
        """Mock HTTP client for external API calls."""
        client = AsyncMock(spec=httpx.AsyncClient)
        return client

    @pytest.fixture
    def model_config(self):
        """Create model configuration for testing."""
        config = ModelConfigFactory()
        config.provider_name = "openai"
        config.model_id = "gpt-4"
        config.parameters = {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9}
        return config

    @pytest.fixture
    def provider_service(self, mock_http_client, mock_uow):
        """Create model provider service with dependencies."""
        return ModelProviderService(uow=mock_uow, http_client=mock_http_client)

    @pytest.fixture
    def inference_service(self, mock_http_client):
        """Create model inference service."""
        return ModelInferenceService(http_client=mock_http_client)

    @pytest.fixture
    def cost_calculator(self):
        """Create cost calculator."""
        return CostCalculator()

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker."""
        return CircuitBreaker(failure_threshold=5, recovery_timeout=60, request_timeout=30)

    @pytest.mark.asyncio
    async def test_successful_model_response_generation(self, inference_service, model_config):
        """Test successful model response generation."""
        # Arrange
        prompt = "What is the capital of France?"
        expected_response = {
            "choices": [
                {"message": {"content": "The capital of France is Paris."}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }

        with patch.object(inference_service, "_make_api_request", return_value=expected_response):
            # Act
            response = await inference_service.generate_response(model_config, prompt)

            # Assert
            assert response is not None
            assert response.response_text == "The capital of France is Paris."
            assert response.tokens_used == 25
            assert response.model_config_id == model_config.id

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, inference_service, model_config):
        """Test rate limit handling with retry mechanism."""
        # Arrange
        prompt = "Test prompt"

        # Simulate rate limit response then success
        rate_limit_response = httpx.Response(
            status_code=429,
            headers={"Retry-After": "1"},
            content=b'{"error": {"message": "Rate limit exceeded"}}',
        )

        success_response = {
            "choices": [{"message": {"content": "Success"}}],
            "usage": {"total_tokens": 10},
        }

        with patch.object(inference_service, "_make_api_request") as mock_request:
            mock_request.side_effect = [
                httpx.HTTPStatusError("Rate limit", request=None, response=rate_limit_response),
                success_response,
            ]

            with patch("asyncio.sleep", return_value=None):  # Speed up test
                # Act
                response = await inference_service.generate_response(model_config, prompt)

                # Assert
                assert response is not None
                assert response.response_text == "Success"
                assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_cost_calculation_integration(self, cost_calculator, model_config):
        """Test cost calculation for different models and usage."""
        # Arrange
        model_config.cost_per_token = Money(Decimal("0.001"), "USD")
        tokens_used = 1000

        # Act
        cost = cost_calculator.calculate_cost(model_config, tokens_used)

        # Assert
        assert cost.amount == Decimal("1.0")
        assert cost.currency == "USD"

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(
        self, circuit_breaker, inference_service, model_config
    ):
        """Test circuit breaker integration with model service."""
        # Arrange
        prompt = "Test prompt"

        # Simulate multiple failures to trigger circuit breaker
        with patch.object(
            inference_service, "_make_api_request", side_effect=Exception("Service error")
        ):
            # Act & Assert
            for i in range(6):  # Exceed failure threshold
                try:
                    await circuit_breaker.call(
                        inference_service.generate_response, model_config, prompt
                    )
                except Exception:
                    pass  # Expected failures

            # Circuit should be open now
            assert circuit_breaker.state == "OPEN"

            # Next call should fail fast
            with pytest.raises(Exception, match="Circuit breaker is OPEN"):
                await circuit_breaker.call(
                    inference_service.generate_response, model_config, prompt
                )

    @pytest.mark.asyncio
    async def test_provider_health_check_integration(self, provider_service, mock_uow):
        """Test provider health check integration."""
        # Arrange
        provider = ModelProviderFactory()
        provider.name = "openai"
        mock_uow.providers.get_by_name.return_value = provider

        with patch.object(provider_service, "_check_provider_health", return_value=True):
            # Act
            is_healthy = await provider_service.check_provider_health("openai")

            # Assert
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_model_availability_check_integration(self, provider_service, mock_uow):
        """Test model availability check integration."""
        # Arrange
        provider = ModelProviderFactory()
        provider.name = "openai"
        mock_uow.providers.get_by_name.return_value = provider

        with patch.object(provider_service, "_check_model_availability", return_value=True):
            # Act
            availability = await provider_service.verify_model_availability(
                {"openai/gpt-4": "gpt-4"}
            )

            # Assert
            assert availability["openai/gpt-4"] is True

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, inference_service, model_config):
        """Test handling of concurrent requests."""
        # Arrange
        prompts = [f"Test prompt {i}" for i in range(10)]
        expected_response = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"total_tokens": 10},
        }

        with patch.object(inference_service, "_make_api_request", return_value=expected_response):
            # Act
            import asyncio

            responses = await asyncio.gather(
                *[inference_service.generate_response(model_config, prompt) for prompt in prompts]
            )

            # Assert
            assert len(responses) == 10
            assert all(response.response_text == "Response" for response in responses)

    @pytest.mark.asyncio
    async def test_model_parameter_validation_integration(self, provider_service):
        """Test model parameter validation integration."""
        # Arrange
        invalid_config = ModelConfigFactory()
        invalid_config.parameters = {
            "temperature": 2.0,  # Invalid: > 1.0
            "max_tokens": -100,  # Invalid: negative
            "top_p": 1.5,  # Invalid: > 1.0
        }

        # Act
        validation_errors = await provider_service.validate_model_parameters(
            {"openai/gpt-4": invalid_config}
        )

        # Assert
        assert "openai/gpt-4" in validation_errors
        errors = validation_errors["openai/gpt-4"]
        assert any("temperature" in error.lower() for error in errors)
        assert any("max_tokens" in error.lower() for error in errors)
        assert any("top_p" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_api_key_validation_integration(self, provider_service, mock_uow):
        """Test API key validation integration."""
        # Arrange
        provider = ModelProviderFactory()
        provider.api_key_hash = "invalid_key_hash"
        mock_uow.providers.get_by_name.return_value = provider

        with patch.object(provider_service, "_validate_api_key", return_value=False):
            # Act & Assert
            with pytest.raises(Exception, match="Invalid API key"):
                await provider_service.verify_model_availability({"openai/gpt-4": "gpt-4"})

    @pytest.mark.asyncio
    async def test_timeout_handling_integration(self, inference_service, model_config):
        """Test timeout handling in API calls."""
        # Arrange
        prompt = "Test prompt"

        with patch.object(
            inference_service, "_make_api_request", side_effect=asyncio.TimeoutError()
        ):
            # Act & Assert
            with pytest.raises(asyncio.TimeoutError):
                await inference_service.generate_response(model_config, prompt)

    @pytest.mark.asyncio
    async def test_error_response_handling_integration(self, inference_service, model_config):
        """Test handling of API error responses."""
        # Arrange
        prompt = "Test prompt"
        error_response = httpx.Response(
            status_code=400, content=b'{"error": {"message": "Invalid request"}}'
        )

        with patch.object(inference_service, "_make_api_request") as mock_request:
            mock_request.side_effect = httpx.HTTPStatusError(
                "Bad request", request=None, response=error_response
            )

            # Act & Assert
            with pytest.raises(httpx.HTTPStatusError):
                await inference_service.generate_response(model_config, prompt)

    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, inference_service, model_config):
        """Test batch processing of multiple prompts."""
        # Arrange
        prompts = [f"Prompt {i}" for i in range(5)]
        expected_response = {
            "choices": [{"message": {"content": "Batch response"}}],
            "usage": {"total_tokens": 15},
        }

        with patch.object(inference_service, "_make_api_request", return_value=expected_response):
            # Act
            responses = await inference_service.generate_batch_responses(model_config, prompts)

            # Assert
            assert len(responses) == 5
            assert all(response.response_text == "Batch response" for response in responses)
            assert all(response.tokens_used == 15 for response in responses)

    @pytest.mark.asyncio
    async def test_provider_switching_integration(self, provider_service, mock_uow):
        """Test automatic provider switching on failure."""
        # Arrange
        primary_provider = ModelProviderFactory()
        primary_provider.name = "openai"
        backup_provider = ModelProviderFactory()
        backup_provider.name = "anthropic"

        mock_uow.providers.get_active_providers.return_value = [primary_provider, backup_provider]

        with patch.object(provider_service, "_check_provider_health", side_effect=[False, True]):
            # Act
            selected_provider = await provider_service.select_available_provider(
                ["openai", "anthropic"]
            )

            # Assert
            assert selected_provider.name == "anthropic"

    @pytest.mark.asyncio
    async def test_cost_tracking_integration(self, provider_service, mock_uow):
        """Test cost tracking across multiple requests."""
        # Arrange
        model_config = ModelConfigFactory()
        model_config.cost_per_token = Money(Decimal("0.002"), "USD")

        # Simulate multiple requests with different token usage
        usage_data = [100, 200, 150, 300, 250]
        total_expected_cost = sum(usage * Decimal("0.002") for usage in usage_data)

        # Act
        total_cost = Money(Decimal("0"), "USD")
        for usage in usage_data:
            cost = provider_service.calculate_request_cost(model_config, usage)
            total_cost = Money(total_cost.amount + cost.amount, cost.currency)

        # Assert
        assert total_cost.amount == total_expected_cost
        assert total_cost.currency == "USD"

    @pytest.mark.asyncio
    async def test_rate_limit_tracking_integration(self, provider_service, mock_uow):
        """Test rate limit tracking and enforcement."""
        # Arrange
        provider = ModelProviderFactory()
        provider.rate_limits = RateLimits(
            requests_per_minute=60,
            requests_per_hour=3600,
            requests_per_day=86400,
            tokens_per_minute=100000,
        )
        mock_uow.providers.get_by_name.return_value = provider

        # Act
        # Simulate rapid requests
        request_count = 0
        for i in range(70):  # Exceed per-minute limit
            can_make_request = await provider_service.check_rate_limits("openai")
            if can_make_request:
                request_count += 1
                await provider_service.record_request("openai", tokens_used=1000)

        # Assert
        assert request_count <= 60  # Should be limited by rate limit
