"""Tests for Model Provider services."""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.domain.model_provider.entities.model_config import ModelConfig
from src.domain.model_provider.entities.model_provider import ModelProvider
from src.domain.model_provider.exceptions import (
    InvalidProviderConfiguration,
    ProviderNotSupported,
    ValidationError,
)
from src.domain.model_provider.services.model_service import ModelService
from src.domain.model_provider.services.provider_factory import ProviderFactory
from src.domain.model_provider.value_objects.model_category import ModelCategory
from src.domain.model_provider.value_objects.money import Money
from src.domain.model_provider.value_objects.provider_type import ProviderType
from src.domain.model_provider.value_objects.validation_result import ValidationResult
from src.domain.test_management.entities.test_configuration import TestConfiguration


class TestProviderFactory:
    """Tests for ProviderFactory."""

    def test_get_supported_providers(self):
        """Test getting all supported provider types."""
        supported = ProviderFactory.get_supported_providers()

        assert isinstance(supported, list)
        assert ProviderType.OPENAI in supported
        assert ProviderType.ANTHROPIC in supported
        assert ProviderType.GOOGLE in supported
        assert ProviderType.BAIDU in supported
        assert ProviderType.ALIBABA in supported

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        config = {"api_key": "sk-test123", "organization": "test-org"}

        provider = ProviderFactory.create_provider(ProviderType.OPENAI, config)

        assert isinstance(provider, ModelProvider)
        assert provider.provider_type == ProviderType.OPENAI
        assert provider.name == "OpenAI"
        assert len(provider.supported_models) > 0

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        config = {"api_key": "sk-ant-test123"}

        provider = ProviderFactory.create_provider(ProviderType.ANTHROPIC, config)

        assert isinstance(provider, ModelProvider)
        assert provider.provider_type == ProviderType.ANTHROPIC
        assert provider.name == "Anthropic"
        assert len(provider.supported_models) > 0

    def test_create_google_provider(self):
        """Test creating Google provider."""
        config = {"api_key": "test-key", "project_id": "test-project"}

        provider = ProviderFactory.create_provider(ProviderType.GOOGLE, config)

        assert isinstance(provider, ModelProvider)
        assert provider.provider_type == ProviderType.GOOGLE
        assert provider.name == "Google"
        assert len(provider.supported_models) > 0

    def test_create_provider_unsupported_type(self):
        """Test creating provider with unsupported type raises error."""
        with pytest.raises(ProviderNotSupported, match="Provider type unknown is not supported"):
            ProviderFactory.create_provider("unknown", {})

    def test_validate_openai_config_valid(self):
        """Test validating valid OpenAI configuration."""
        config = {"api_key": "sk-test123", "organization": "org-test"}

        result = ProviderFactory.validate_provider_config(ProviderType.OPENAI, config)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_openai_config_missing_api_key(self):
        """Test validating OpenAI configuration with missing API key."""
        config = {"organization": "org-test"}

        result = ProviderFactory.validate_provider_config(ProviderType.OPENAI, config)

        assert result.is_valid is False
        assert "api_key is required for OpenAI provider" in result.errors

    def test_validate_anthropic_config_valid(self):
        """Test validating valid Anthropic configuration."""
        config = {"api_key": "sk-ant-test123"}

        result = ProviderFactory.validate_provider_config(ProviderType.ANTHROPIC, config)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_google_config_missing_project_id(self):
        """Test validating Google configuration with missing project ID."""
        config = {"api_key": "test-key"}

        result = ProviderFactory.validate_provider_config(ProviderType.GOOGLE, config)

        assert result.is_valid is False
        assert "project_id is required for Google provider" in result.errors


class TestModelService:
    """Tests for ModelService domain service."""

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

    @pytest.fixture
    def sample_provider(self, sample_model_config):
        """Fixture providing sample model provider."""
        # Create another model config for completeness
        gpt35_config = ModelConfig(
            model_id="gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo",
            max_tokens=4096,
            cost_per_input_token=Decimal("0.000001"),
            cost_per_output_token=Decimal("0.000002"),
            supports_streaming=True,
            model_category=ModelCategory.TEXT_GENERATION,
            parameters={"temperature": 1.0},
        )

        return ModelProvider.create(
            name="OpenAI",
            provider_type=ProviderType.OPENAI,
            supported_models=[sample_model_config, gpt35_config],
            api_credentials={"api_key": "test-key"},
        )

    @pytest.fixture
    def provider_repository_mock(self, sample_provider):
        """Fixture providing mock provider repository."""
        repo = AsyncMock()
        repo.get_all.return_value = [sample_provider]
        repo.get_by_id.return_value = sample_provider
        repo.get_by_provider_type.return_value = [sample_provider]
        repo.get_operational_providers.return_value = [sample_provider]
        return repo

    @pytest.fixture
    def model_service(self, provider_repository_mock):
        """Fixture providing ModelService instance."""
        return ModelService(provider_repository_mock)

    @pytest.mark.asyncio
    async def test_get_all_models(self, model_service):
        """Test getting all available models across providers."""
        models = await model_service.get_all_models()

        assert isinstance(models, list)
        assert len(models) == 2
        assert models[0].model_id == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_model_by_id(self, model_service):
        """Test getting specific model by ID."""
        model = await model_service.get_model_by_id("gpt-4")

        assert model is not None
        assert model.model_id == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_model_by_id_not_found(self, model_service):
        """Test getting non-existent model returns None."""
        model = await model_service.get_model_by_id("non-existent")

        assert model is None

    @pytest.mark.asyncio
    async def test_get_models_by_provider_type(self, model_service):
        """Test getting models by provider type."""
        models = await model_service.get_models_by_provider_type(ProviderType.OPENAI)

        assert isinstance(models, list)
        assert len(models) == 2
        assert models[0].model_id == "gpt-4"

    @pytest.mark.asyncio
    async def test_validate_test_configuration_valid(self, model_service):
        """Test validating valid test configuration."""
        test_config = TestConfiguration(
            models=["gpt-4", "gpt-3.5-turbo"], max_tokens=1000, temperature=1.0
        )

        result = await model_service.validate_test_configuration(test_config)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_test_configuration_invalid_model(self, model_service):
        """Test validating test configuration with invalid model."""
        test_config = TestConfiguration(
            models=["gpt-4", "non-existent-model"], max_tokens=1000, temperature=1.0
        )

        result = await model_service.validate_test_configuration(test_config)

        assert result.is_valid is False
        assert "Model non-existent-model not found in any provider" in result.errors

    @pytest.mark.asyncio
    async def test_estimate_test_cost(self, model_service):
        """Test estimating test cost."""
        test_config = TestConfiguration(
            models=["gpt-4", "gpt-3.5-turbo"], max_tokens=1000, temperature=1.0
        )

        # Mock some sample data
        sample_count = 10
        estimated_input_tokens = 500
        estimated_output_tokens = 500

        cost = await model_service.estimate_test_cost(
            test_config, sample_count, estimated_input_tokens, estimated_output_tokens
        )

        assert isinstance(cost, Money)
        assert cost.currency == "USD"
        assert cost.amount > 0

    @pytest.mark.asyncio
    async def test_check_provider_health(self, model_service, sample_provider):
        """Test checking provider health."""
        health_result = await model_service.check_provider_health(sample_provider.id)

        assert health_result is not None
        assert "provider_id" in health_result
        assert "health_status" in health_result
        assert "is_operational" in health_result

    @pytest.mark.asyncio
    async def test_get_supported_model_parameters(self, model_service):
        """Test getting supported parameters for a model."""
        parameters = await model_service.get_supported_model_parameters("gpt-4")

        assert isinstance(parameters, dict)
        assert "temperature" in parameters

    @pytest.mark.asyncio
    async def test_compare_model_capabilities(self, model_service):
        """Test comparing capabilities between models."""
        comparison = await model_service.compare_model_capabilities(["gpt-4"])

        assert isinstance(comparison, dict)
        assert "models" in comparison
        assert len(comparison["models"]) == 1
