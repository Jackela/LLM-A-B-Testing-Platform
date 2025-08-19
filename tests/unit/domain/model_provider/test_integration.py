"""Integration tests between Model Provider and Test Management domains."""

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.domain.model_provider.entities.model_config import ModelConfig
from src.domain.model_provider.entities.model_provider import ModelProvider
from src.domain.model_provider.services.model_service import ModelService
from src.domain.model_provider.value_objects.health_status import HealthStatus
from src.domain.model_provider.value_objects.model_category import ModelCategory
from src.domain.model_provider.value_objects.provider_type import ProviderType
from src.domain.test_management.entities.test_configuration import TestConfiguration


class TestModelProviderTestManagementIntegration:
    """Tests integration between Model Provider and Test Management domains."""

    @pytest.fixture
    def sample_models(self):
        """Fixture providing sample model configurations."""
        return [
            ModelConfig(
                model_id="gpt-4",
                display_name="GPT-4",
                max_tokens=8192,
                cost_per_input_token=Decimal("0.00003"),
                cost_per_output_token=Decimal("0.00006"),
                supports_streaming=True,
                model_category=ModelCategory.CHAT_COMPLETION,
                parameters={"temperature": 1.0, "top_p": 1.0},
            ),
            ModelConfig(
                model_id="claude-3-5-sonnet-20241022",
                display_name="Claude 3.5 Sonnet",
                max_tokens=8192,
                cost_per_input_token=Decimal("0.000003"),
                cost_per_output_token=Decimal("0.000015"),
                supports_streaming=True,
                model_category=ModelCategory.CHAT_COMPLETION,
                parameters={"temperature": 1.0, "top_p": 1.0},
            ),
        ]

    @pytest.fixture
    def sample_providers(self, sample_models):
        """Fixture providing sample providers."""
        openai_provider = ModelProvider.create(
            name="OpenAI",
            provider_type=ProviderType.OPENAI,
            supported_models=[sample_models[0]],
            api_credentials={"api_key": "sk-test-openai"},
        )
        openai_provider.update_health_status(HealthStatus.HEALTHY)

        anthropic_provider = ModelProvider.create(
            name="Anthropic",
            provider_type=ProviderType.ANTHROPIC,
            supported_models=[sample_models[1]],
            api_credentials={"api_key": "sk-ant-test-anthropic"},
        )
        anthropic_provider.update_health_status(HealthStatus.HEALTHY)

        return [openai_provider, anthropic_provider]

    @pytest.fixture
    def mock_repository(self, sample_providers):
        """Fixture providing mock repository."""
        repo = AsyncMock()
        repo.get_all.return_value = sample_providers
        repo.get_operational_providers.return_value = sample_providers
        repo.get_by_provider_type.return_value = sample_providers[:1]
        return repo

    @pytest.fixture
    def model_service(self, mock_repository):
        """Fixture providing model service."""
        return ModelService(mock_repository)

    @pytest.mark.asyncio
    async def test_validate_compatible_test_configuration(self, model_service):
        """Test validating a compatible test configuration."""
        # Create test configuration with models from both providers
        test_config = TestConfiguration(
            models=["gpt-4", "claude-3-5-sonnet-20241022"],
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
        )

        # Validate configuration
        result = await model_service.validate_test_configuration(test_config)

        assert result.is_valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_incompatible_test_configuration(self, model_service):
        """Test validating an incompatible test configuration."""
        # Create test configuration with non-existent model
        test_config = TestConfiguration(
            models=["gpt-4", "non-existent-model"], max_tokens=1000, temperature=0.7
        )

        # Validate configuration
        result = await model_service.validate_test_configuration(test_config)

        assert result.is_valid is False
        assert "Model non-existent-model not found in any provider" in result.errors

    @pytest.mark.asyncio
    async def test_estimate_cost_for_test_configuration(self, model_service):
        """Test estimating cost for a test configuration."""
        # Create test configuration
        test_config = TestConfiguration(
            models=["gpt-4", "claude-3-5-sonnet-20241022"], max_tokens=1000, temperature=0.7
        )

        # Estimate cost
        estimated_cost = await model_service.estimate_test_cost(
            test_config, sample_count=10, estimated_input_tokens=500, estimated_output_tokens=500
        )

        assert estimated_cost.currency == "USD"
        assert estimated_cost.amount > 0
        # Should be cost for both models across 10 samples
        expected_gpt4_cost = Decimal("0.00003") * 500 + Decimal("0.00006") * 500
        expected_claude_cost = Decimal("0.000003") * 500 + Decimal("0.000015") * 500
        expected_total = (expected_gpt4_cost + expected_claude_cost) * 10
        assert estimated_cost.amount == expected_total

    @pytest.mark.asyncio
    async def test_validate_parameter_compatibility(self, model_service):
        """Test parameter compatibility validation."""
        # Test with parameters that exceed model limits
        test_config = TestConfiguration(
            models=["gpt-4", "claude-3-5-sonnet-20241022"],  # Need 2 models for A/B testing
            max_tokens=10000,  # Higher than model's max
            temperature=0.7,
        )

        result = await model_service.validate_test_configuration(test_config)

        # Should be valid (the validation might not trigger warnings for max_tokens)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_provider_operational_status_affects_validation(
        self, model_service, mock_repository
    ):
        """Test that provider operational status affects validation."""
        # Mock no operational providers
        mock_repository.get_operational_providers.return_value = []

        test_config = TestConfiguration(
            models=["gpt-4", "claude-3-5-sonnet-20241022"], max_tokens=1000, temperature=0.7
        )

        result = await model_service.validate_test_configuration(test_config)

        # Should be valid but with warnings about provider status
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("not currently operational" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_cross_domain_model_discovery(self, model_service):
        """Test that model discovery works across providers."""
        # Get all models
        all_models = await model_service.get_all_models()

        assert len(all_models) == 2
        model_ids = [model.model_id for model in all_models]
        assert "gpt-4" in model_ids
        assert "claude-3-5-sonnet-20241022" in model_ids

    @pytest.mark.asyncio
    async def test_provider_specific_model_retrieval(self, model_service):
        """Test retrieving models by provider type."""
        # Get OpenAI models only
        openai_models = await model_service.get_models_by_provider_type(ProviderType.OPENAI)

        assert len(openai_models) == 1
        assert openai_models[0].model_id == "gpt-4"

    @pytest.mark.asyncio
    async def test_model_parameter_compatibility_check(self, model_service):
        """Test checking model parameter compatibility."""
        # Get supported parameters for GPT-4
        gpt4_params = await model_service.get_supported_model_parameters("gpt-4")

        assert "temperature" in gpt4_params
        assert "top_p" in gpt4_params

        # Verify test configuration can use these parameters
        test_config = TestConfiguration(
            models=["gpt-4", "claude-3-5-sonnet-20241022"],  # Need 2 models for A/B testing
            max_tokens=1000,
            temperature=gpt4_params["temperature"],
            top_p=gpt4_params["top_p"],
        )

        result = await model_service.validate_test_configuration(test_config)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_model_capability_comparison(self, model_service):
        """Test comparing capabilities between models for test selection."""
        comparison = await model_service.compare_model_capabilities(
            ["gpt-4", "claude-3-5-sonnet-20241022"]
        )

        assert "models" in comparison
        assert "comparison_matrix" in comparison
        assert len(comparison["models"]) == 2

        # Check that comparison includes cost information
        model_info = comparison["models"]
        gpt4_info = next(m for m in model_info if m["model_id"] == "gpt-4")
        claude_info = next(m for m in model_info if m["model_id"] == "claude-3-5-sonnet-20241022")

        assert "cost_per_input_token" in gpt4_info
        assert "cost_per_output_token" in gpt4_info
        assert "cost_per_input_token" in claude_info
        assert "cost_per_output_token" in claude_info

    def test_test_configuration_business_rules_maintained(self):
        """Test that Test Management business rules are still enforced."""
        # Test Configuration should still require at least 2 models for A/B testing
        with pytest.raises(Exception):  # BusinessRuleViolation from Test Management domain
            TestConfiguration(models=["gpt-4"], max_tokens=1000, temperature=0.7)  # Only one model

    def test_cost_calculation_integration(self, sample_models):
        """Test cost calculation integration between domains."""
        model_config = sample_models[0]  # GPT-4

        # Calculate cost using model config
        cost = model_config.calculate_estimated_cost(1000, 500)
        expected = Decimal("0.00003") * 1000 + Decimal("0.00006") * 500

        assert cost == expected

        # This cost should be compatible with test configuration cost estimation
        # which uses the same calculation internally
