"""Factory for creating model providers."""

from decimal import Decimal
from typing import Any, Dict, List

from ..entities.model_config import ModelConfig
from ..entities.model_provider import ModelProvider
from ..exceptions import InvalidProviderConfiguration, ProviderNotSupported, ValidationError
from ..value_objects.model_category import ModelCategory
from ..value_objects.provider_type import ProviderType
from ..value_objects.validation_result import ValidationResult


class ProviderFactory:
    """Factory for creating model provider instances."""

    @staticmethod
    def create_provider(provider_type: ProviderType, config: Dict[str, Any]) -> ModelProvider:
        """
        Create provider instance based on type and configuration.

        Args:
            provider_type: The type of provider to create
            config: Configuration dictionary containing API credentials and settings

        Returns:
            ModelProvider: Configured provider instance

        Raises:
            ProviderNotSupported: If provider type is not supported
            InvalidProviderConfiguration: If configuration is invalid
        """
        if not isinstance(provider_type, ProviderType):
            if isinstance(provider_type, str):
                # Try to convert string to ProviderType
                try:
                    provider_type = ProviderType(provider_type.lower())
                except ValueError:
                    raise ProviderNotSupported(f"Provider type {provider_type} is not supported")
            else:
                raise ProviderNotSupported(f"Provider type {provider_type} is not supported")

        # Validate configuration first
        validation_result = ProviderFactory.validate_provider_config(provider_type, config)
        if not validation_result.is_valid:
            error_message = "; ".join(validation_result.errors)
            raise InvalidProviderConfiguration(f"Invalid configuration: {error_message}")

        # Get provider-specific settings
        provider_name = ProviderType.get_display_name(provider_type)
        supported_models = ProviderFactory._create_default_models(provider_type)

        # Create and return provider
        return ModelProvider.create(
            name=provider_name,
            provider_type=provider_type,
            supported_models=supported_models,
            api_credentials=config,
        )

    @staticmethod
    def get_supported_providers() -> List[ProviderType]:
        """
        Get all supported provider types.

        Returns:
            List[ProviderType]: List of supported provider types
        """
        return list(ProviderType)

    @staticmethod
    def validate_provider_config(
        provider_type: ProviderType, config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate provider configuration.

        Args:
            provider_type: The type of provider
            config: Configuration dictionary to validate

        Returns:
            ValidationResult: Result of validation with errors and warnings
        """
        errors = []
        warnings = []

        # Common validation for all providers
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return ValidationResult(is_valid=False, errors=tuple(errors))

        # Provider-specific validation
        if provider_type == ProviderType.OPENAI:
            errors.extend(ProviderFactory._validate_openai_config(config))
            warnings.extend(ProviderFactory._get_openai_warnings(config))

        elif provider_type == ProviderType.ANTHROPIC:
            errors.extend(ProviderFactory._validate_anthropic_config(config))
            warnings.extend(ProviderFactory._get_anthropic_warnings(config))

        elif provider_type == ProviderType.GOOGLE:
            errors.extend(ProviderFactory._validate_google_config(config))
            warnings.extend(ProviderFactory._get_google_warnings(config))

        elif provider_type == ProviderType.BAIDU:
            errors.extend(ProviderFactory._validate_baidu_config(config))
            warnings.extend(ProviderFactory._get_baidu_warnings(config))

        elif provider_type == ProviderType.ALIBABA:
            errors.extend(ProviderFactory._validate_alibaba_config(config))
            warnings.extend(ProviderFactory._get_alibaba_warnings(config))

        else:
            errors.append(f"Validation not implemented for provider type: {provider_type}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=tuple(errors), warnings=tuple(warnings))

    @staticmethod
    def _create_default_models(provider_type: ProviderType) -> List[ModelConfig]:
        """Create default model configurations for a provider type."""
        if provider_type == ProviderType.OPENAI:
            return [
                ModelConfig(
                    model_id="gpt-4",
                    display_name="GPT-4",
                    max_tokens=8192,
                    cost_per_input_token=Decimal("0.00003"),
                    cost_per_output_token=Decimal("0.00006"),
                    supports_streaming=True,
                    model_category=ModelCategory.CHAT_COMPLETION,
                    parameters={
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                    },
                ),
                ModelConfig(
                    model_id="gpt-3.5-turbo",
                    display_name="GPT-3.5 Turbo",
                    max_tokens=4096,
                    cost_per_input_token=Decimal("0.000001"),
                    cost_per_output_token=Decimal("0.000002"),
                    supports_streaming=True,
                    model_category=ModelCategory.CHAT_COMPLETION,
                    parameters={
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                    },
                ),
            ]

        elif provider_type == ProviderType.ANTHROPIC:
            return [
                ModelConfig(
                    model_id="claude-3-5-sonnet-20241022",
                    display_name="Claude 3.5 Sonnet",
                    max_tokens=8192,
                    cost_per_input_token=Decimal("0.000003"),
                    cost_per_output_token=Decimal("0.000015"),
                    supports_streaming=True,
                    model_category=ModelCategory.CHAT_COMPLETION,
                    parameters={"temperature": 1.0, "top_p": 1.0, "top_k": 250},
                ),
                ModelConfig(
                    model_id="claude-3-haiku-20240307",
                    display_name="Claude 3 Haiku",
                    max_tokens=4096,
                    cost_per_input_token=Decimal("0.00000025"),
                    cost_per_output_token=Decimal("0.00000125"),
                    supports_streaming=True,
                    model_category=ModelCategory.CHAT_COMPLETION,
                    parameters={"temperature": 1.0, "top_p": 1.0, "top_k": 250},
                ),
            ]

        elif provider_type == ProviderType.GOOGLE:
            return [
                ModelConfig(
                    model_id="gemini-1.5-pro",
                    display_name="Gemini 1.5 Pro",
                    max_tokens=8192,
                    cost_per_input_token=Decimal("0.0000035"),
                    cost_per_output_token=Decimal("0.0000105"),
                    supports_streaming=True,
                    model_category=ModelCategory.CHAT_COMPLETION,
                    parameters={"temperature": 1.0, "top_p": 0.95, "top_k": 64},
                ),
                ModelConfig(
                    model_id="gemini-1.5-flash",
                    display_name="Gemini 1.5 Flash",
                    max_tokens=4096,
                    cost_per_input_token=Decimal("0.000000075"),
                    cost_per_output_token=Decimal("0.0000003"),
                    supports_streaming=True,
                    model_category=ModelCategory.CHAT_COMPLETION,
                    parameters={"temperature": 1.0, "top_p": 0.95, "top_k": 64},
                ),
            ]

        elif provider_type == ProviderType.BAIDU:
            return [
                ModelConfig(
                    model_id="ernie-4.0-turbo",
                    display_name="ERNIE-4.0-Turbo",
                    max_tokens=8192,
                    cost_per_input_token=Decimal("0.000002"),
                    cost_per_output_token=Decimal("0.000006"),
                    supports_streaming=True,
                    model_category=ModelCategory.CHAT_COMPLETION,
                    parameters={"temperature": 1.0, "top_p": 0.8},
                ),
                ModelConfig(
                    model_id="ernie-3.5",
                    display_name="ERNIE-3.5",
                    max_tokens=4096,
                    cost_per_input_token=Decimal("0.0000008"),
                    cost_per_output_token=Decimal("0.000002"),
                    supports_streaming=True,
                    model_category=ModelCategory.CHAT_COMPLETION,
                    parameters={"temperature": 1.0, "top_p": 0.8},
                ),
            ]

        elif provider_type == ProviderType.ALIBABA:
            return [
                ModelConfig(
                    model_id="qwen-turbo",
                    display_name="Qwen-Turbo",
                    max_tokens=8192,
                    cost_per_input_token=Decimal("0.000002"),
                    cost_per_output_token=Decimal("0.000006"),
                    supports_streaming=True,
                    model_category=ModelCategory.CHAT_COMPLETION,
                    parameters={"temperature": 1.0, "top_p": 0.9},
                ),
                ModelConfig(
                    model_id="qwen-plus",
                    display_name="Qwen-Plus",
                    max_tokens=32768,
                    cost_per_input_token=Decimal("0.000004"),
                    cost_per_output_token=Decimal("0.000012"),
                    supports_streaming=True,
                    model_category=ModelCategory.CHAT_COMPLETION,
                    parameters={"temperature": 1.0, "top_p": 0.9},
                ),
            ]

        return []

    @staticmethod
    def _validate_openai_config(config: Dict[str, Any]) -> List[str]:
        """Validate OpenAI-specific configuration."""
        errors = []

        if "api_key" not in config:
            errors.append("api_key is required for OpenAI provider")
        elif not isinstance(config["api_key"], str) or not config["api_key"].strip():
            errors.append("api_key must be a non-empty string")
        elif not config["api_key"].startswith("sk-"):
            errors.append("OpenAI api_key should start with 'sk-'")

        if "organization" in config and not isinstance(config["organization"], str):
            errors.append("organization must be a string")

        return errors

    @staticmethod
    def _get_openai_warnings(config: Dict[str, Any]) -> List[str]:
        """Get warnings for OpenAI configuration."""
        warnings = []

        if "organization" not in config:
            warnings.append("organization parameter is recommended for OpenAI provider")

        return warnings

    @staticmethod
    def _validate_anthropic_config(config: Dict[str, Any]) -> List[str]:
        """Validate Anthropic-specific configuration."""
        errors = []

        if "api_key" not in config:
            errors.append("api_key is required for Anthropic provider")
        elif not isinstance(config["api_key"], str) or not config["api_key"].strip():
            errors.append("api_key must be a non-empty string")
        elif not config["api_key"].startswith("sk-ant-"):
            errors.append("Anthropic api_key should start with 'sk-ant-'")

        return errors

    @staticmethod
    def _get_anthropic_warnings(config: Dict[str, Any]) -> List[str]:
        """Get warnings for Anthropic configuration."""
        return []

    @staticmethod
    def _validate_google_config(config: Dict[str, Any]) -> List[str]:
        """Validate Google-specific configuration."""
        errors = []

        if "api_key" not in config:
            errors.append("api_key is required for Google provider")
        elif not isinstance(config["api_key"], str) or not config["api_key"].strip():
            errors.append("api_key must be a non-empty string")

        if "project_id" not in config:
            errors.append("project_id is required for Google provider")
        elif not isinstance(config["project_id"], str) or not config["project_id"].strip():
            errors.append("project_id must be a non-empty string")

        return errors

    @staticmethod
    def _get_google_warnings(config: Dict[str, Any]) -> List[str]:
        """Get warnings for Google configuration."""
        return []

    @staticmethod
    def _validate_baidu_config(config: Dict[str, Any]) -> List[str]:
        """Validate Baidu-specific configuration."""
        errors = []

        if "api_key" not in config:
            errors.append("api_key is required for Baidu provider")
        elif not isinstance(config["api_key"], str) or not config["api_key"].strip():
            errors.append("api_key must be a non-empty string")

        if "secret_key" not in config:
            errors.append("secret_key is required for Baidu provider")
        elif not isinstance(config["secret_key"], str) or not config["secret_key"].strip():
            errors.append("secret_key must be a non-empty string")

        return errors

    @staticmethod
    def _get_baidu_warnings(config: Dict[str, Any]) -> List[str]:
        """Get warnings for Baidu configuration."""
        return []

    @staticmethod
    def _validate_alibaba_config(config: Dict[str, Any]) -> List[str]:
        """Validate Alibaba-specific configuration."""
        errors = []

        if "api_key" not in config:
            errors.append("api_key is required for Alibaba provider")
        elif not isinstance(config["api_key"], str) or not config["api_key"].strip():
            errors.append("api_key must be a non-empty string")

        return errors

    @staticmethod
    def _get_alibaba_warnings(config: Dict[str, Any]) -> List[str]:
        """Get warnings for Alibaba configuration."""
        return []
