"""Model request DTOs for external provider integration."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...domain.model_provider.value_objects.provider_type import ProviderType


@dataclass
class ModelRequestDTO:
    """Data Transfer Object for model API requests."""

    provider_id: str
    model_id: str
    prompt: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    test_context: Optional[Dict[str, Any]] = field(default_factory=dict)
    request_id: Optional[str] = None
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self) -> None:
        """Validate and normalize request data."""
        if not self.provider_id:
            raise ValueError("provider_id is required")
        if not self.model_id:
            raise ValueError("model_id is required")
        if not self.prompt:
            raise ValueError("prompt is required")

        # Parameters should always be initialized by default_factory=dict
        # No additional normalization needed

        # Set default parameters if not provided
        self.parameters.setdefault("max_tokens", 1000)
        self.parameters.setdefault("temperature", 0.7)

        # Validate numeric parameters
        if "max_tokens" in self.parameters:
            if (
                not isinstance(self.parameters["max_tokens"], int)
                or self.parameters["max_tokens"] <= 0
            ):
                raise ValueError("max_tokens must be a positive integer")

        if "temperature" in self.parameters:
            temp = self.parameters["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                raise ValueError("temperature must be between 0 and 2")

    def to_provider_format(self, provider_type: ProviderType) -> Dict[str, Any]:
        """Convert to provider-specific request format."""
        base_request = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": self.prompt}],
            **self.parameters,
        }

        # Provider-specific transformations
        if provider_type == ProviderType.OPENAI:
            return base_request
        elif provider_type == ProviderType.ANTHROPIC:
            return {
                "model": self.model_id,
                "messages": [{"role": "user", "content": self.prompt}],
                "max_tokens": self.parameters.get("max_tokens", 1000),
                "temperature": self.parameters.get("temperature", 0.7),
            }
        elif provider_type == ProviderType.GOOGLE:
            return {
                "model": self.model_id,
                "contents": [{"parts": [{"text": self.prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": self.parameters.get("max_tokens", 1000),
                    "temperature": self.parameters.get("temperature", 0.7),
                },
            }
        else:
            return base_request

    def get_estimated_tokens(self) -> int:
        """Estimate input token count for cost calculation."""
        # Simple estimation: ~4 characters per token
        return len(self.prompt) // 4 + 10  # Add buffer for system messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "prompt": self.prompt,
            "parameters": self.parameters,
            "test_context": self.test_context,
            "request_id": self.request_id,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }


@dataclass
class BatchModelRequestDTO:
    """Data Transfer Object for batch model requests."""

    requests: List[ModelRequestDTO]
    batch_id: Optional[str] = None
    max_parallel_requests: int = 5
    batch_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        """Validate batch request."""
        if not self.requests:
            raise ValueError("At least one request is required")

        if len(self.requests) > 100:  # Reasonable batch limit
            raise ValueError("Batch size cannot exceed 100 requests")

        if self.max_parallel_requests < 1 or self.max_parallel_requests > 20:
            raise ValueError("max_parallel_requests must be between 1 and 20")

    def group_by_provider(self) -> Dict[str, List[ModelRequestDTO]]:
        """Group requests by provider for efficient processing."""
        grouped: Dict[str, List[ModelRequestDTO]] = {}
        for request in self.requests:
            provider_id = request.provider_id
            if provider_id not in grouped:
                grouped[provider_id] = []
            grouped[provider_id].append(request)
        return grouped

    def get_total_estimated_tokens(self) -> int:
        """Get total estimated tokens for all requests."""
        return sum(request.get_estimated_tokens() for request in self.requests)
