"""Model response DTOs for external provider integration."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ...domain.model_provider.value_objects.provider_type import ProviderType


class ResponseStatus(Enum):
    """Response status enumeration."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"


@dataclass
class ModelResponseDTO:
    """Data Transfer Object for model API responses."""

    provider_id: str
    model_id: str
    status: ResponseStatus
    text: Optional[str] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    response_id: Optional[str] = None

    # Token information
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Cost information
    input_cost: Optional[Decimal] = None
    output_cost: Optional[Decimal] = None
    total_cost: Optional[Decimal] = None

    # Performance metrics
    latency_ms: Optional[int] = None
    first_token_latency_ms: Optional[int] = None
    tokens_per_second: Optional[float] = None

    # Metadata
    finish_reason: Optional[str] = None
    model_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    provider_metadata: Dict[str, Any] = field(default_factory=dict)

    # Retry information
    retry_count: int = 0
    is_retry: bool = False

    def __post_init__(self) -> None:
        """Validate and calculate derived fields."""
        if not self.provider_id:
            raise ValueError("provider_id is required")
        if not self.model_id:
            raise ValueError("model_id is required")
        if not isinstance(self.status, ResponseStatus):
            raise ValueError("status must be a ResponseStatus enum")

        # Calculate total tokens if individual counts are available
        if self.input_tokens is not None and self.output_tokens is not None:
            if self.total_tokens is None:
                self.total_tokens = self.input_tokens + self.output_tokens

        # Calculate total cost if individual costs are available
        if self.input_cost is not None and self.output_cost is not None:
            if self.total_cost is None:
                self.total_cost = self.input_cost + self.output_cost

        # Calculate tokens per second if we have the data
        if (
            self.output_tokens is not None
            and self.latency_ms is not None
            and self.latency_ms > 0
            and self.tokens_per_second is None
        ):
            self.tokens_per_second = (self.output_tokens * 1000) / self.latency_ms

    @classmethod
    def create_success_response(
        cls,
        provider_id: str,
        model_id: str,
        text: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        request_id: Optional[str] = None,
    ) -> "ModelResponseDTO":
        """Factory method for creating successful responses."""
        return cls(
            provider_id=provider_id,
            model_id=model_id,
            status=ResponseStatus.SUCCESS,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            request_id=request_id,
            finish_reason="stop",
        )

    @classmethod
    def create_error_response(
        cls,
        provider_id: str,
        model_id: str,
        error_message: str,
        status: ResponseStatus = ResponseStatus.ERROR,
        request_id: Optional[str] = None,
    ) -> "ModelResponseDTO":
        """Factory method for creating error responses."""
        return cls(
            provider_id=provider_id,
            model_id=model_id,
            status=status,
            error_message=error_message,
            request_id=request_id,
        )

    @classmethod
    def from_provider_response(
        cls,
        provider_id: str,
        model_id: str,
        provider_response: Dict[str, Any],
        provider_type: ProviderType,
        request_start_time: datetime,
        request_id: Optional[str] = None,
    ) -> "ModelResponseDTO":
        """Create response DTO from provider-specific response format."""
        current_time = datetime.utcnow()
        latency_ms = int((current_time - request_start_time).total_seconds() * 1000)

        try:
            if provider_type == ProviderType.OPENAI:
                return cls._from_openai_response(
                    provider_id, model_id, provider_response, latency_ms, request_id
                )
            elif provider_type == ProviderType.ANTHROPIC:
                return cls._from_anthropic_response(
                    provider_id, model_id, provider_response, latency_ms, request_id
                )
            elif provider_type == ProviderType.GOOGLE:
                return cls._from_google_response(
                    provider_id, model_id, provider_response, latency_ms, request_id
                )
            else:
                return cls._from_generic_response(
                    provider_id, model_id, provider_response, latency_ms, request_id
                )
        except Exception as e:
            return cls.create_error_response(
                provider_id,
                model_id,
                f"Failed to parse provider response: {str(e)}",
                request_id=request_id,
            )

    @classmethod
    def _from_openai_response(
        cls,
        provider_id: str,
        model_id: str,
        response: Dict[str, Any],
        latency_ms: int,
        request_id: Optional[str],
    ) -> "ModelResponseDTO":
        """Parse OpenAI response format."""
        if "error" in response:
            return cls.create_error_response(
                provider_id,
                model_id,
                response["error"].get("message", "Unknown error"),
                request_id=request_id,
            )

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})

        return cls(
            provider_id=provider_id,
            model_id=model_id,
            status=ResponseStatus.SUCCESS,
            text=message.get("content", ""),
            response_id=response.get("id"),
            request_id=request_id,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
            model_version=response.get("model"),
            provider_metadata={"raw_response": response},
        )

    @classmethod
    def _from_anthropic_response(
        cls,
        provider_id: str,
        model_id: str,
        response: Dict[str, Any],
        latency_ms: int,
        request_id: Optional[str],
    ) -> "ModelResponseDTO":
        """Parse Anthropic response format."""
        if "error" in response:
            return cls.create_error_response(
                provider_id,
                model_id,
                response["error"].get("message", "Unknown error"),
                request_id=request_id,
            )

        content_block = response.get("content", [{}])[0]
        usage = response.get("usage", {})

        return cls(
            provider_id=provider_id,
            model_id=model_id,
            status=ResponseStatus.SUCCESS,
            text=content_block.get("text", ""),
            response_id=response.get("id"),
            request_id=request_id,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            latency_ms=latency_ms,
            finish_reason=response.get("stop_reason"),
            model_version=response.get("model"),
            provider_metadata={"raw_response": response},
        )

    @classmethod
    def _from_google_response(
        cls,
        provider_id: str,
        model_id: str,
        response: Dict[str, Any],
        latency_ms: int,
        request_id: Optional[str],
    ) -> "ModelResponseDTO":
        """Parse Google response format."""
        if "error" in response:
            return cls.create_error_response(
                provider_id,
                model_id,
                response["error"].get("message", "Unknown error"),
                request_id=request_id,
            )

        candidate = response.get("candidates", [{}])[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [{}])
        text = parts[0].get("text", "") if parts else ""

        # Google doesn't provide token usage in the same way
        estimated_tokens = len(text.split()) if text else 0

        return cls(
            provider_id=provider_id,
            model_id=model_id,
            status=ResponseStatus.SUCCESS,
            text=text,
            request_id=request_id,
            output_tokens=estimated_tokens,
            latency_ms=latency_ms,
            finish_reason=candidate.get("finishReason"),
            provider_metadata={"raw_response": response},
        )

    @classmethod
    def _from_generic_response(
        cls,
        provider_id: str,
        model_id: str,
        response: Dict[str, Any],
        latency_ms: int,
        request_id: Optional[str],
    ) -> "ModelResponseDTO":
        """Parse generic response format."""
        return cls(
            provider_id=provider_id,
            model_id=model_id,
            status=ResponseStatus.SUCCESS,
            text=str(response.get("text", response.get("content", ""))),
            request_id=request_id,
            latency_ms=latency_ms,
            provider_metadata={"raw_response": response},
        )

    def is_successful(self) -> bool:
        """Check if the response was successful."""
        return self.status in [ResponseStatus.SUCCESS, ResponseStatus.PARTIAL_SUCCESS]

    def is_retryable_error(self) -> bool:
        """Check if this error is retryable."""
        return self.status in [
            ResponseStatus.RATE_LIMITED,
            ResponseStatus.TIMEOUT,
            ResponseStatus.ERROR,  # Some errors might be transient
        ]

    def get_text_length(self) -> int:
        """Get the length of the response text."""
        return len(self.text) if self.text else 0

    def calculate_cost(self, input_cost_per_token: Decimal, output_cost_per_token: Decimal) -> None:
        """Calculate costs based on token usage and pricing."""
        if self.input_tokens is not None:
            self.input_cost = (Decimal(self.input_tokens) * input_cost_per_token) / 1000

        if self.output_tokens is not None:
            self.output_cost = (Decimal(self.output_tokens) * output_cost_per_token) / 1000

        if self.input_cost is not None and self.output_cost is not None:
            self.total_cost = self.input_cost + self.output_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "status": self.status.value,
            "text": self.text,
            "error_message": self.error_message,
            "request_id": self.request_id,
            "response_id": self.response_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "first_token_latency_ms": self.first_token_latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "finish_reason": self.finish_reason,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat(),
            "provider_metadata": self.provider_metadata,
            "retry_count": self.retry_count,
            "is_retry": self.is_retry,
        }

        # Convert Decimal fields to strings for JSON serialization
        if self.input_cost is not None:
            result["input_cost"] = str(self.input_cost)
        if self.output_cost is not None:
            result["output_cost"] = str(self.output_cost)
        if self.total_cost is not None:
            result["total_cost"] = str(self.total_cost)

        return result


@dataclass
class BatchModelResponseDTO:
    """Data Transfer Object for batch model responses."""

    batch_id: Optional[str]
    responses: List[ModelResponseDTO]
    successful_count: int = 0
    failed_count: int = 0
    total_latency_ms: Optional[int] = None
    total_cost: Optional[Decimal] = None

    def __post_init__(self) -> None:
        """Calculate summary statistics."""
        self.successful_count = sum(1 for r in self.responses if r.is_successful())
        self.failed_count = len(self.responses) - self.successful_count

        # Calculate total latency (max of all requests in parallel batch)
        latencies = [r.latency_ms for r in self.responses if r.latency_ms is not None]
        if latencies:
            self.total_latency_ms = max(latencies)

        # Calculate total cost
        costs = [r.total_cost for r in self.responses if r.total_cost is not None]
        if costs:
            self.total_cost = sum(costs) or None

    def get_success_rate(self) -> float:
        """Get the success rate for the batch."""
        if not self.responses:
            return 0.0
        return self.successful_count / len(self.responses)

    def get_responses_by_status(self, status: ResponseStatus) -> List[ModelResponseDTO]:
        """Get responses filtered by status."""
        return [r for r in self.responses if r.status == status]
