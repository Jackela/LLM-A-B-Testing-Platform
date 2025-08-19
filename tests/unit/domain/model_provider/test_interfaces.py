"""Tests for Model Provider interfaces."""

from abc import ABC
from typing import List
from unittest.mock import AsyncMock, Mock

import pytest

from src.domain.model_provider.entities.model_config import ModelConfig
from src.domain.model_provider.entities.model_response import ModelResponse
from src.domain.model_provider.interfaces.provider_adapter import ProviderAdapter
from src.domain.model_provider.repositories.provider_repository import ProviderRepository
from src.domain.model_provider.value_objects.health_status import HealthStatus
from src.domain.model_provider.value_objects.rate_limits import RateLimits


class TestProviderAdapter:
    """Tests for ProviderAdapter interface."""

    def test_provider_adapter_is_abstract(self):
        """Test that ProviderAdapter is an abstract base class."""
        assert issubclass(ProviderAdapter, ABC)

        with pytest.raises(TypeError):
            ProviderAdapter()

    def test_provider_adapter_has_required_methods(self):
        """Test that ProviderAdapter defines required abstract methods."""
        required_methods = [
            "call_model",
            "validate_credentials",
            "check_health",
            "get_supported_models",
            "get_rate_limits",
        ]

        for method_name in required_methods:
            assert hasattr(ProviderAdapter, method_name)
            method = getattr(ProviderAdapter, method_name)
            assert getattr(method, "__isabstractmethod__", False)


class TestProviderRepository:
    """Tests for ProviderRepository interface."""

    def test_provider_repository_is_abstract(self):
        """Test that ProviderRepository is an abstract base class."""
        assert issubclass(ProviderRepository, ABC)

        with pytest.raises(TypeError):
            ProviderRepository()

    def test_provider_repository_has_required_methods(self):
        """Test that ProviderRepository defines required abstract methods."""
        required_methods = [
            "save",
            "get_by_id",
            "get_by_name",
            "get_all",
            "delete",
            "get_by_provider_type",
        ]

        for method_name in required_methods:
            assert hasattr(ProviderRepository, method_name)
            method = getattr(ProviderRepository, method_name)
            assert getattr(method, "__isabstractmethod__", False)


class TestProviderAdapterContract:
    """Contract tests for ProviderAdapter implementations."""

    class MockProviderAdapter(ProviderAdapter):
        """Mock implementation for testing."""

        async def call_model(self, prompt, model_config, **kwargs):
            return ModelResponse.create_pending(model_config, prompt)

        async def validate_credentials(self):
            return True

        async def check_health(self):
            from src.domain.model_provider.interfaces.provider_adapter import HealthCheckResult

            return HealthCheckResult(is_healthy=True, response_time_ms=100)

        def get_supported_models(self):
            return []

        def get_rate_limits(self):
            return RateLimits(60, 1000, 0, 0)

        async def get_model_info(self, model_id):
            return None

        async def list_available_models(self):
            return []

        async def estimate_cost(self, model_id, input_tokens, output_tokens):
            return 0.01

        async def get_usage_stats(self):
            return {"requests": 0, "cost": 0.0}

    @pytest.fixture
    def adapter(self):
        """Fixture providing mock adapter."""
        return self.MockProviderAdapter()

    @pytest.mark.asyncio
    async def test_call_model_returns_model_response(self, adapter):
        """Test that call_model returns ModelResponse."""
        from decimal import Decimal

        from src.domain.model_provider.entities.model_config import ModelConfig
        from src.domain.model_provider.value_objects.model_category import ModelCategory

        config = ModelConfig(
            model_id="test-model",
            display_name="Test Model",
            max_tokens=1000,
            cost_per_input_token=Decimal("0.001"),
            cost_per_output_token=Decimal("0.002"),
            supports_streaming=False,
            model_category=ModelCategory.TEXT_GENERATION,
            parameters={},
        )

        response = await adapter.call_model("Hello", config)
        assert isinstance(response, ModelResponse)

    @pytest.mark.asyncio
    async def test_validate_credentials_returns_bool(self, adapter):
        """Test that validate_credentials returns boolean."""
        result = await adapter.validate_credentials()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_check_health_returns_health_check_result(self, adapter):
        """Test that check_health returns HealthCheckResult."""
        from src.domain.model_provider.interfaces.provider_adapter import HealthCheckResult

        result = await adapter.check_health()
        assert isinstance(result, HealthCheckResult)

    def test_get_supported_models_returns_list(self, adapter):
        """Test that get_supported_models returns list."""
        result = adapter.get_supported_models()
        assert isinstance(result, list)

    def test_get_rate_limits_returns_rate_limits(self, adapter):
        """Test that get_rate_limits returns RateLimits."""
        result = adapter.get_rate_limits()
        assert isinstance(result, RateLimits)
