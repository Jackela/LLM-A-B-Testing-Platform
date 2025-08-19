"""Domain service for model operations."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from ..entities.model_config import ModelConfig
from ..entities.model_provider import ModelProvider
from ..exceptions import ModelNotFound, ProviderNotFound
from ..repositories.provider_repository import ProviderRepository
from ..value_objects.model_category import ModelCategory
from ..value_objects.money import Money
from ..value_objects.provider_type import ProviderType
from ..value_objects.validation_result import ValidationResult


class ModelService:
    """Domain service for model operations across providers."""

    def __init__(self, provider_repository: ProviderRepository):
        self._provider_repository = provider_repository

    async def get_all_models(self) -> List[ModelConfig]:
        """
        Get all available models across all providers.

        Returns:
            List[ModelConfig]: List of all available model configurations
        """
        all_providers = await self._provider_repository.get_all()
        models = []

        for provider in all_providers:
            models.extend(provider.supported_models)

        return models

    async def get_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get specific model by ID across all providers.

        Args:
            model_id: The ID of the model to find

        Returns:
            Optional[ModelConfig]: Model configuration if found, None otherwise
        """
        all_providers = await self._provider_repository.get_all()

        for provider in all_providers:
            model = provider.find_model_config(model_id)
            if model:
                return model

        return None

    async def get_models_by_provider_type(self, provider_type: ProviderType) -> List[ModelConfig]:
        """
        Get models by provider type.

        Args:
            provider_type: The type of provider to get models from

        Returns:
            List[ModelConfig]: List of models from the specified provider type
        """
        providers = await self._provider_repository.get_by_provider_type(provider_type)
        models = []

        for provider in providers:
            models.extend(provider.supported_models)

        return models

    def get_models_by_category(self, category: ModelCategory) -> List[ModelConfig]:
        """
        Get models by category across all providers.

        Args:
            category: The category of models to retrieve

        Returns:
            List[ModelConfig]: List of models in the specified category
        """
        all_models = self.get_all_models()
        return [model for model in all_models if model.model_category == category]

    def get_streaming_models(self) -> List[ModelConfig]:
        """
        Get all models that support streaming.

        Returns:
            List[ModelConfig]: List of models that support streaming
        """
        all_models = self.get_all_models()
        return [model for model in all_models if model.supports_streaming]

    def get_cheapest_models(self, limit: int = 10) -> List[ModelConfig]:
        """
        Get cheapest models by input token cost.

        Args:
            limit: Maximum number of models to return

        Returns:
            List[ModelConfig]: List of cheapest models sorted by cost
        """
        all_models = self.get_all_models()
        sorted_models = sorted(all_models, key=lambda m: m.cost_per_input_token)
        return sorted_models[:limit]

    def get_most_capable_models(self, limit: int = 10) -> List[ModelConfig]:
        """
        Get most capable models by token capacity.

        Args:
            limit: Maximum number of models to return

        Returns:
            List[ModelConfig]: List of most capable models sorted by capacity
        """
        all_models = self.get_all_models()
        sorted_models = sorted(all_models, key=lambda m: m.max_tokens, reverse=True)
        return sorted_models[:limit]

    async def validate_test_configuration(self, test_config) -> ValidationResult:
        """
        Validate that all models in test configuration are available and properly configured.

        Args:
            test_config: Test configuration from test management domain

        Returns:
            ValidationResult: Result of validation with errors and warnings
        """
        errors = []
        warnings = []

        # Get all providers for validation
        all_providers = await self._provider_repository.get_all()
        all_models = []
        for provider in all_providers:
            all_models.extend(provider.supported_models)

        available_model_ids = [model.model_id for model in all_models]

        # Validate each model in configuration
        for model_id in test_config.models:
            if model_id not in available_model_ids:
                errors.append(f"Model {model_id} not found in any provider")
                continue

            # Find the model and validate compatibility
            model_config = await self.get_model_by_id(model_id)
            if model_config:
                # Check if test config parameters are compatible
                test_params = test_config.get_model_parameters()
                for param_name, param_value in test_params.items():
                    if param_name in model_config.parameters:
                        # Validate parameter value against model's parameter constraints
                        if param_name == "max_tokens" and param_value > model_config.max_tokens:
                            warnings.append(
                                f"Model {model_id}: requested max_tokens ({param_value}) "
                                f"exceeds model limit ({model_config.max_tokens})"
                            )

        # Check if we have operational providers for the requested models
        operational_providers = await self._provider_repository.get_operational_providers()
        operational_model_ids = []
        for provider in operational_providers:
            operational_model_ids.extend([m.model_id for m in provider.supported_models])

        for model_id in test_config.models:
            if model_id not in operational_model_ids:
                warnings.append(f"Model {model_id} provider is not currently operational")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=tuple(errors), warnings=tuple(warnings))

    async def estimate_test_cost(
        self,
        test_config,
        sample_count: int,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> Money:
        """
        Calculate estimated cost for complete test execution.

        Args:
            test_config: Test configuration from test management domain
            sample_count: Number of test samples
            estimated_input_tokens: Estimated input tokens per sample
            estimated_output_tokens: Estimated output tokens per sample

        Returns:
            Money: Estimated total cost for the test
        """
        total_cost = Money.zero("USD")

        for model_id in test_config.models:
            model_config = await self.get_model_by_id(model_id)
            if model_config:
                # Calculate cost per sample for this model
                cost_per_sample = model_config.calculate_estimated_cost(
                    estimated_input_tokens, estimated_output_tokens
                )

                # Multiply by sample count
                model_total = Money(cost_per_sample, "USD") * sample_count
                total_cost = total_cost + model_total

        return total_cost

    async def check_provider_health(self, provider_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Check provider health status.

        Args:
            provider_id: ID of the provider to check

        Returns:
            Optional[Dict[str, Any]]: Health status information if provider found
        """
        provider = await self._provider_repository.get_by_id(provider_id)
        if not provider:
            return None

        # This would typically call the provider adapter to check health
        # For now, return current status
        return {
            "provider_id": str(provider.id),
            "provider_name": provider.name,
            "health_status": provider.health_status.name,
            "is_operational": provider.health_status.is_operational,
            "model_count": len(provider.supported_models),
            "rate_limits": provider.rate_limits.to_dict(),
        }

    async def get_supported_model_parameters(self, model_id: str) -> Dict[str, Any]:
        """
        Get supported parameters for a specific model.

        Args:
            model_id: ID of the model

        Returns:
            Dict[str, Any]: Dictionary of supported parameters
        """
        model_config = await self.get_model_by_id(model_id)
        if not model_config:
            raise ModelNotFound(f"Model {model_id} not found", model_id=model_id)

        return model_config.parameters.copy()

    async def compare_model_capabilities(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Compare capabilities between models.

        Args:
            model_ids: List of model IDs to compare

        Returns:
            Dict[str, Any]: Comparison results
        """
        comparison = {"models": [], "comparison_matrix": {}}

        models = []
        for model_id in model_ids:
            model = await self.get_model_by_id(model_id)
            if model:
                models.append(model)

        for model in models:
            model_info = {
                "model_id": model.model_id,
                "display_name": model.display_name,
                "max_tokens": model.max_tokens,
                "cost_per_input_token": str(model.cost_per_input_token),
                "cost_per_output_token": str(model.cost_per_output_token),
                "supports_streaming": model.supports_streaming,
                "category": model.model_category.value,
                "parameters": model.parameters,
            }
            comparison["models"].append(model_info)

        # Add comparison metrics
        if models:
            comparison["comparison_matrix"] = {
                "cheapest_input": min(models, key=lambda m: m.cost_per_input_token).model_id,
                "cheapest_output": min(models, key=lambda m: m.cost_per_output_token).model_id,
                "highest_capacity": max(models, key=lambda m: m.max_tokens).model_id,
                "streaming_support": [m.model_id for m in models if m.supports_streaming],
            }

        return comparison

    def find_compatible_models(self, requirements: Dict[str, Any]) -> List[ModelConfig]:
        """
        Find models that meet specific requirements.

        Args:
            requirements: Dictionary of requirements to match

        Returns:
            List[ModelConfig]: List of compatible models
        """
        all_models = self.get_all_models()
        compatible_models = []

        for model in all_models:
            is_compatible = True

            # Check minimum token capacity
            if "min_tokens" in requirements:
                if model.max_tokens < requirements["min_tokens"]:
                    is_compatible = False

            # Check maximum cost
            if "max_cost_per_input_token" in requirements:
                if model.cost_per_input_token > requirements["max_cost_per_input_token"]:
                    is_compatible = False

            # Check category
            if "category" in requirements:
                if model.model_category != requirements["category"]:
                    is_compatible = False

            # Check streaming support
            if "supports_streaming" in requirements:
                if model.supports_streaming != requirements["supports_streaming"]:
                    is_compatible = False

            # Check required parameters
            if "required_parameters" in requirements:
                for param in requirements["required_parameters"]:
                    if not model.supports_parameter(param):
                        is_compatible = False
                        break

            if is_compatible:
                compatible_models.append(model)

        return compatible_models
