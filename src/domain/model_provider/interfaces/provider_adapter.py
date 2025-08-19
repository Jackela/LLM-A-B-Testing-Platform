"""Provider adapter interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..entities.model_config import ModelConfig
from ..entities.model_response import ModelResponse
from ..value_objects.health_status import HealthStatus
from ..value_objects.rate_limits import RateLimits


class HealthCheckResult:
    """Result of a provider health check."""

    def __init__(
        self,
        is_healthy: bool,
        response_time_ms: int,
        error_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.is_healthy = is_healthy
        self.response_time_ms = response_time_ms
        self.error_message = error_message
        self.details = details or {}

    @property
    def health_status(self) -> HealthStatus:
        """Get HealthStatus based on check result."""
        if not self.is_healthy:
            return HealthStatus.UNHEALTHY
        return HealthStatus.from_response_time(self.response_time_ms)


class ProviderAdapter(ABC):
    """Abstract interface for LLM provider adapters."""

    @abstractmethod
    async def call_model(self, prompt: str, model_config: ModelConfig, **kwargs) -> ModelResponse:
        """
        Make API call to the provider.

        Args:
            prompt: The input prompt for the model
            model_config: Configuration for the specific model
            **kwargs: Additional parameters for the call

        Returns:
            ModelResponse: Completed response from the model

        Raises:
            ModelNotFound: If the model is not available
            RateLimitExceeded: If rate limits are exceeded
            ProviderHealthCheckFailed: If provider is not healthy
            ValidationError: If parameters are invalid
        """
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """
        Validate API credentials with the provider.

        Returns:
            bool: True if credentials are valid, False otherwise
        """
        pass

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """
        Check provider service health.

        Returns:
            HealthCheckResult: Detailed health check result
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[ModelConfig]:
        """
        Get list of supported models for this provider.

        Returns:
            List[ModelConfig]: List of available model configurations
        """
        pass

    @abstractmethod
    def get_rate_limits(self) -> RateLimits:
        """
        Get current rate limiting information.

        Returns:
            RateLimits: Current rate limit configuration and state
        """
        pass

    @abstractmethod
    async def get_model_info(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: The ID of the model to query

        Returns:
            Optional[ModelConfig]: Model configuration if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_available_models(self) -> List[str]:
        """
        List all model IDs currently available from the provider.

        Returns:
            List[str]: List of available model IDs
        """
        pass

    @abstractmethod
    async def estimate_cost(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        """
        Estimate cost for a model call.

        Args:
            model_id: The ID of the model
            input_tokens: Number of input tokens
            output_tokens: Expected number of output tokens

        Returns:
            Optional[float]: Estimated cost in USD, None if unable to estimate
        """
        pass

    @abstractmethod
    async def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics from the provider.

        Returns:
            Dict[str, Any]: Usage statistics including rate limit usage,
                           costs, request counts, etc.
        """
        pass

    # Optional methods with default implementations

    async def supports_streaming(self, model_id: str) -> bool:
        """
        Check if a model supports streaming responses.

        Args:
            model_id: The ID of the model to check

        Returns:
            bool: True if streaming is supported, False otherwise
        """
        model_config = await self.get_model_info(model_id)
        return model_config.supports_streaming if model_config else False

    async def get_context_window(self, model_id: str) -> Optional[int]:
        """
        Get the context window size for a model.

        Args:
            model_id: The ID of the model

        Returns:
            Optional[int]: Context window size in tokens, None if unknown
        """
        model_config = await self.get_model_info(model_id)
        return model_config.max_tokens if model_config else None

    def get_provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            str: Provider name
        """
        return self.__class__.__name__.replace("Adapter", "")

    async def test_connection(self) -> bool:
        """
        Test basic connectivity to the provider.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            health_result = await self.check_health()
            return health_result.is_healthy
        except Exception:
            return False
