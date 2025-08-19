"""Domain-model mapper for Model Provider domain."""

from datetime import datetime
from typing import Any, Dict, List

from .....domain.model_provider.entities.model_config import ModelConfig
from .....domain.model_provider.entities.model_provider import ModelProvider
from .....domain.model_provider.entities.model_response import ModelResponse
from .....domain.model_provider.value_objects.health_status import HealthStatus
from .....domain.model_provider.value_objects.model_category import ModelCategory
from .....domain.model_provider.value_objects.money import Money
from .....domain.model_provider.value_objects.provider_type import ProviderType
from .....domain.model_provider.value_objects.rate_limits import RateLimits
from ...models.provider_models import ModelConfigModel, ModelProviderModel, ModelResponseModel


class ProviderMapper:
    """Mapper between Provider domain entities and database models."""

    def to_model(self, provider: ModelProvider) -> ModelProviderModel:
        """Convert ModelProvider domain entity to database model."""
        provider_model = ModelProviderModel(
            id=provider.id,
            name=provider.name,
            provider_type=provider.provider_type,
            health_status=provider.health_status,
            api_credentials=provider.api_credentials.copy(),
            metadata=provider.metadata.copy() if provider.metadata else {},
            created_at=provider.created_at,
            updated_at=provider.updated_at,
            # Rate limiting fields
            requests_per_minute=provider.rate_limits.requests_per_minute,
            requests_per_day=provider.rate_limits.requests_per_day,
            current_minute_requests=provider.rate_limits.current_minute_requests,
            current_day_requests=provider.rate_limits.current_day_requests,
            last_reset_minute=provider.rate_limits.last_reset_minute,
            last_reset_day=provider.rate_limits.last_reset_day,
        )

        # Convert model configurations
        provider_model.supported_models = [
            self._model_config_to_model(config, provider.id) for config in provider.supported_models
        ]

        return provider_model

    def to_domain(self, provider_model: ModelProviderModel) -> ModelProvider:
        """Convert database model to ModelProvider domain entity."""
        # Reconstruct rate limits
        rate_limits = RateLimits(
            requests_per_minute=provider_model.requests_per_minute,
            requests_per_day=provider_model.requests_per_day,
            current_minute_requests=provider_model.current_minute_requests,
            current_day_requests=provider_model.current_day_requests,
            last_reset_minute=provider_model.last_reset_minute,
            last_reset_day=provider_model.last_reset_day,
        )

        # Convert model configurations
        supported_models = [
            self._model_to_config(config_model) for config_model in provider_model.supported_models
        ]

        # Create provider entity
        provider = ModelProvider(
            id=provider_model.id,
            name=provider_model.name,
            provider_type=provider_model.provider_type,
            supported_models=supported_models,
            rate_limits=rate_limits,
            health_status=provider_model.health_status,
            api_credentials=provider_model.api_credentials.copy(),
            created_at=provider_model.created_at,
            updated_at=provider_model.updated_at,
            metadata=provider_model.metadata.copy() if provider_model.metadata else {},
        )

        # Clear domain events after loading from database
        provider.clear_domain_events()

        return provider

    def _model_config_to_model(self, config: ModelConfig, provider_id) -> ModelConfigModel:
        """Convert ModelConfig to ModelConfigModel."""
        return ModelConfigModel(
            id=config.id,
            provider_id=provider_id,
            model_id=config.model_id,
            model_name=config.model_name,
            model_category=config.model_category,
            max_tokens=config.max_tokens,
            supports_streaming=config.supports_streaming,
            cost_per_input_token=config.cost_per_input_token.amount,
            cost_per_output_token=config.cost_per_output_token.amount,
            supported_parameters=config.supported_parameters.copy(),
            metadata=config.metadata.copy() if config.metadata else {},
            created_at=config.created_at,
            updated_at=config.updated_at,
        )

    def _model_to_config(self, config_model: ModelConfigModel) -> ModelConfig:
        """Convert ModelConfigModel to ModelConfig."""
        return ModelConfig(
            id=config_model.id,
            model_id=config_model.model_id,
            model_name=config_model.model_name,
            model_category=config_model.model_category,
            max_tokens=config_model.max_tokens,
            supports_streaming=config_model.supports_streaming,
            cost_per_input_token=Money(config_model.cost_per_input_token, "USD"),
            cost_per_output_token=Money(config_model.cost_per_output_token, "USD"),
            supported_parameters=config_model.supported_parameters.copy(),
            metadata=config_model.metadata.copy() if config_model.metadata else {},
            created_at=config_model.created_at,
            updated_at=config_model.updated_at,
        )

    def response_to_model(self, response: ModelResponse) -> ModelResponseModel:
        """Convert ModelResponse to ModelResponseModel."""
        return ModelResponseModel(
            id=response.id,
            test_id=response.test_id,
            sample_id=response.sample_id,
            provider_id=response.provider_id,
            model_config_id=response.model_config.id,
            prompt=response.prompt,
            response_text=response.response_text,
            status=response.status,
            error_message=response.error_message,
            parameters_used=response.parameters_used.copy() if response.parameters_used else {},
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_cost=response.total_cost.amount if response.total_cost else None,
            request_time=response.request_time,
            response_time=response.response_time,
            metadata=response.metadata.copy() if response.metadata else {},
        )

    def model_to_response(
        self, response_model: ModelResponseModel, model_config: ModelConfig
    ) -> ModelResponse:
        """Convert ModelResponseModel to ModelResponse."""
        return ModelResponse(
            id=response_model.id,
            test_id=response_model.test_id,
            sample_id=response_model.sample_id,
            provider_id=response_model.provider_id,
            model_config=model_config,
            prompt=response_model.prompt,
            response_text=response_model.response_text,
            status=response_model.status,
            error_message=response_model.error_message,
            parameters_used=(
                response_model.parameters_used.copy() if response_model.parameters_used else {}
            ),
            latency_ms=response_model.latency_ms,
            input_tokens=response_model.input_tokens,
            output_tokens=response_model.output_tokens,
            total_cost=(
                Money(response_model.total_cost, "USD") if response_model.total_cost else None
            ),
            request_time=response_model.request_time,
            response_time=response_model.response_time,
            metadata=response_model.metadata.copy() if response_model.metadata else {},
        )
