"""Tests for Model Provider entities."""

from decimal import Decimal
from typing import Any, Dict
from uuid import UUID, uuid4

import pytest

from src.domain.model_provider.entities.model_config import ModelConfig
from src.domain.model_provider.entities.model_provider import ModelProvider
from src.domain.model_provider.entities.model_response import ModelResponse
from src.domain.model_provider.exceptions import (
    BusinessRuleViolation,
    ModelNotFound,
    RateLimitExceeded,
    ValidationError,
)
from src.domain.model_provider.value_objects.health_status import HealthStatus
from src.domain.model_provider.value_objects.model_category import ModelCategory
from src.domain.model_provider.value_objects.provider_type import ProviderType
from src.domain.model_provider.value_objects.rate_limits import RateLimits


class TestModelConfig:
    """Tests for ModelConfig entity."""

    def test_create_valid_model_config(self):
        """Test creating a valid model configuration."""
        config = ModelConfig(
            model_id="gpt-4",
            display_name="GPT-4",
            max_tokens=8192,
            cost_per_input_token=Decimal("0.00003"),
            cost_per_output_token=Decimal("0.00006"),
            supports_streaming=True,
            model_category=ModelCategory.TEXT_GENERATION,
            parameters={"temperature": 1.0, "top_p": 1.0},
        )

        assert config.model_id == "gpt-4"
        assert config.display_name == "GPT-4"
        assert config.max_tokens == 8192
        assert config.supports_streaming is True
        assert config.model_category == ModelCategory.TEXT_GENERATION

    def test_model_config_validation_empty_model_id(self):
        """Test that empty model_id raises ValidationError."""
        with pytest.raises(ValidationError, match="Model ID is required"):
            ModelConfig(
                model_id="",
                display_name="GPT-4",
                max_tokens=8192,
                cost_per_input_token=Decimal("0.00003"),
                cost_per_output_token=Decimal("0.00006"),
                supports_streaming=True,
                model_category=ModelCategory.TEXT_GENERATION,
                parameters={},
            )

    def test_model_config_validation_negative_max_tokens(self):
        """Test that negative max_tokens raises ValidationError."""
        with pytest.raises(ValidationError, match="Max tokens must be positive"):
            ModelConfig(
                model_id="gpt-4",
                display_name="GPT-4",
                max_tokens=-1,
                cost_per_input_token=Decimal("0.00003"),
                cost_per_output_token=Decimal("0.00006"),
                supports_streaming=True,
                model_category=ModelCategory.TEXT_GENERATION,
                parameters={},
            )

    def test_model_config_validation_negative_costs(self):
        """Test that negative costs raise ValidationError."""
        with pytest.raises(ValidationError, match="Token costs cannot be negative"):
            ModelConfig(
                model_id="gpt-4",
                display_name="GPT-4",
                max_tokens=8192,
                cost_per_input_token=Decimal("-0.00003"),
                cost_per_output_token=Decimal("0.00006"),
                supports_streaming=True,
                model_category=ModelCategory.TEXT_GENERATION,
                parameters={},
            )

    def test_model_config_parameter_validation_temperature(self):
        """Test temperature parameter validation."""
        with pytest.raises(ValidationError, match="Temperature must be between 0.0 and 2.0"):
            ModelConfig(
                model_id="gpt-4",
                display_name="GPT-4",
                max_tokens=8192,
                cost_per_input_token=Decimal("0.00003"),
                cost_per_output_token=Decimal("0.00006"),
                supports_streaming=True,
                model_category=ModelCategory.TEXT_GENERATION,
                parameters={"temperature": 3.0},
            )

    def test_model_config_parameter_validation_top_p(self):
        """Test top_p parameter validation."""
        with pytest.raises(ValidationError, match="top_p must be between 0.0 and 1.0"):
            ModelConfig(
                model_id="gpt-4",
                display_name="GPT-4",
                max_tokens=8192,
                cost_per_input_token=Decimal("0.00003"),
                cost_per_output_token=Decimal("0.00006"),
                supports_streaming=True,
                model_category=ModelCategory.TEXT_GENERATION,
                parameters={"top_p": 1.5},
            )

    def test_calculate_estimated_cost(self):
        """Test cost calculation for token usage."""
        config = ModelConfig(
            model_id="gpt-4",
            display_name="GPT-4",
            max_tokens=8192,
            cost_per_input_token=Decimal("0.00003"),
            cost_per_output_token=Decimal("0.00006"),
            supports_streaming=True,
            model_category=ModelCategory.TEXT_GENERATION,
            parameters={},
        )

        cost = config.calculate_estimated_cost(1000, 500)
        expected_cost = Decimal("0.00003") * 1000 + Decimal("0.00006") * 500
        assert cost == expected_cost

    def test_supports_parameter(self):
        """Test parameter support checking."""
        config = ModelConfig(
            model_id="gpt-4",
            display_name="GPT-4",
            max_tokens=8192,
            cost_per_input_token=Decimal("0.00003"),
            cost_per_output_token=Decimal("0.00006"),
            supports_streaming=True,
            model_category=ModelCategory.TEXT_GENERATION,
            parameters={"temperature": 1.0, "top_p": 1.0},
        )

        assert config.supports_parameter("temperature") is True
        assert config.supports_parameter("top_p") is True
        assert config.supports_parameter("frequency_penalty") is False


class TestModelProvider:
    """Tests for ModelProvider aggregate root."""

    @pytest.fixture
    def sample_model_configs(self):
        """Fixture providing sample model configurations."""
        return [
            ModelConfig(
                model_id="gpt-4",
                display_name="GPT-4",
                max_tokens=8192,
                cost_per_input_token=Decimal("0.00003"),
                cost_per_output_token=Decimal("0.00006"),
                supports_streaming=True,
                model_category=ModelCategory.TEXT_GENERATION,
                parameters={"temperature": 1.0},
            ),
            ModelConfig(
                model_id="gpt-3.5-turbo",
                display_name="GPT-3.5 Turbo",
                max_tokens=4096,
                cost_per_input_token=Decimal("0.000001"),
                cost_per_output_token=Decimal("0.000002"),
                supports_streaming=True,
                model_category=ModelCategory.TEXT_GENERATION,
                parameters={"temperature": 1.0},
            ),
        ]

    def test_create_valid_provider(self, sample_model_configs):
        """Test creating a valid model provider."""
        provider = ModelProvider.create(
            name="OpenAI",
            provider_type=ProviderType.OPENAI,
            supported_models=sample_model_configs,
            api_credentials={"api_key": "test-key"},
        )

        assert provider.name == "OpenAI"
        assert provider.provider_type == ProviderType.OPENAI
        assert len(provider.supported_models) == 2
        assert provider.health_status == HealthStatus.UNKNOWN
        assert isinstance(provider.id, UUID)

    def test_create_provider_no_models_raises_error(self):
        """Test that creating provider with no models raises BusinessRuleViolation."""
        with pytest.raises(
            BusinessRuleViolation, match="Provider must have at least one supported model"
        ):
            ModelProvider.create(
                name="OpenAI",
                provider_type=ProviderType.OPENAI,
                supported_models=[],
                api_credentials={"api_key": "test-key"},
            )

    def test_find_model_config(self, sample_model_configs):
        """Test finding model configuration by ID."""
        provider = ModelProvider.create(
            name="OpenAI",
            provider_type=ProviderType.OPENAI,
            supported_models=sample_model_configs,
            api_credentials={"api_key": "test-key"},
        )

        config = provider.find_model_config("gpt-4")
        assert config is not None
        assert config.model_id == "gpt-4"

        config = provider.find_model_config("non-existent")
        assert config is None

    def test_call_model_success(self, sample_model_configs):
        """Test successful model call."""
        provider = ModelProvider.create(
            name="OpenAI",
            provider_type=ProviderType.OPENAI,
            supported_models=sample_model_configs,
            api_credentials={"api_key": "test-key"},
        )

        # Set rate limits to allow requests
        provider.rate_limits = RateLimits(
            requests_per_minute=60,
            requests_per_day=1000,
            current_minute_count=0,
            current_day_count=0,
        )

        # Set provider to healthy status to allow operations
        provider.update_health_status(HealthStatus.HEALTHY)

        response = provider.call_model("gpt-4", "Hello, world!")
        assert response.model_config.model_id == "gpt-4"
        assert response.prompt == "Hello, world!"

    def test_call_model_not_found(self, sample_model_configs):
        """Test calling non-existent model raises ModelNotFound."""
        provider = ModelProvider.create(
            name="OpenAI",
            provider_type=ProviderType.OPENAI,
            supported_models=sample_model_configs,
            api_credentials={"api_key": "test-key"},
        )

        with pytest.raises(ModelNotFound, match="Model non-existent not found in provider OpenAI"):
            provider.call_model("non-existent", "Hello, world!")

    def test_call_model_rate_limit_exceeded(self, sample_model_configs):
        """Test calling model when rate limit exceeded raises RateLimitExceeded."""
        provider = ModelProvider.create(
            name="OpenAI",
            provider_type=ProviderType.OPENAI,
            supported_models=sample_model_configs,
            api_credentials={"api_key": "test-key"},
        )

        # Set rate limits to be exceeded
        provider.rate_limits = RateLimits(
            requests_per_minute=1,
            requests_per_day=1000,
            current_minute_count=1,
            current_day_count=0,
        )

        with pytest.raises(RateLimitExceeded, match="Rate limit exceeded for provider OpenAI"):
            provider.call_model("gpt-4", "Hello, world!")

    def test_update_health_status(self, sample_model_configs):
        """Test updating provider health status."""
        provider = ModelProvider.create(
            name="OpenAI",
            provider_type=ProviderType.OPENAI,
            supported_models=sample_model_configs,
            api_credentials={"api_key": "test-key"},
        )

        initial_status = provider.health_status
        provider.update_health_status(HealthStatus.HEALTHY)

        assert provider.health_status == HealthStatus.HEALTHY
        # Should have domain event for health status change
        assert len(provider._domain_events) > 1  # Creation event + health change event


class TestModelResponse:
    """Tests for ModelResponse entity."""

    @pytest.fixture
    def sample_model_config(self):
        """Fixture providing sample model configuration."""
        return ModelConfig(
            model_id="gpt-4",
            display_name="GPT-4",
            max_tokens=8192,
            cost_per_input_token=Decimal("0.00003"),
            cost_per_output_token=Decimal("0.00006"),
            supports_streaming=True,
            model_category=ModelCategory.TEXT_GENERATION,
            parameters={"temperature": 1.0},
        )

    def test_create_pending_response(self, sample_model_config):
        """Test creating a pending model response."""
        response = ModelResponse.create_pending(sample_model_config, "Hello, world!")

        assert response.model_config == sample_model_config
        assert response.prompt == "Hello, world!"
        assert response.response_text is None
        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_complete_response(self, sample_model_config):
        """Test completing a model response."""
        response = ModelResponse.create_pending(sample_model_config, "Hello, world!")

        response.complete_response(
            response_text="Hello! How can I help you today?",
            input_tokens=10,
            output_tokens=20,
            latency_ms=150,
        )

        assert response.response_text == "Hello! How can I help you today?"
        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.latency_ms == 150

    def test_calculate_cost(self, sample_model_config):
        """Test calculating response cost."""
        response = ModelResponse.create_pending(sample_model_config, "Hello, world!")
        response.complete_response(
            response_text="Hello! How can I help you today?",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=150,
        )

        cost = response.calculate_cost()
        expected_cost = Decimal("0.00003") * 1000 + Decimal("0.00006") * 500
        assert cost.amount == expected_cost
        assert cost.currency == "USD"
