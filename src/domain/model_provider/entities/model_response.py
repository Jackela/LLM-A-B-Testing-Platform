"""Model response entity."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..exceptions import CostCalculationError, ValidationError
from ..value_objects.money import Money
from .model_config import ModelConfig


@dataclass
class ModelResponse:
    """Entity representing a response from an LLM model."""

    id: UUID
    model_config: ModelConfig
    prompt: str
    response_text: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _domain_events: List[object] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Initialize response after creation."""
        if not self.id:
            self.id = uuid4()

        # Validate initial state
        if self.input_tokens < 0:
            raise ValidationError("Input tokens cannot be negative")

        if self.output_tokens < 0:
            raise ValidationError("Output tokens cannot be negative")

        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValidationError("Latency cannot be negative")

    @classmethod
    def create_pending(cls, model_config: ModelConfig, prompt: str) -> "ModelResponse":
        """Factory method to create a pending response."""
        if not prompt.strip():
            raise ValidationError("Prompt cannot be empty")

        response = cls(
            id=uuid4(), model_config=model_config, prompt=prompt, created_at=datetime.utcnow()
        )

        # Add domain event
        from ..events.provider_events import ModelCallRequested

        event = ModelCallRequested(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=response.id,
            model_id=model_config.model_id,
            prompt_preview=prompt[:100],  # First 100 chars for privacy
            parameters=model_config.parameters.copy(),
        )
        response._domain_events.append(event)

        return response

    def complete_response(
        self,
        response_text: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Complete the response with results."""
        if self.is_completed():
            raise ValidationError("Response is already completed")

        if input_tokens < 0 or output_tokens < 0:
            raise ValidationError("Token counts cannot be negative")

        if latency_ms < 0:
            raise ValidationError("Latency cannot be negative")

        self.response_text = response_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.completed_at = datetime.utcnow()

        if metadata:
            self.metadata.update(metadata)

        # Add domain event
        from ..events.provider_events import ModelCallCompleted

        try:
            cost = self.calculate_cost()
            cost_str = str(cost.amount)
        except CostCalculationError:
            cost_str = "0.00"

        event = ModelCallCompleted(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=self.id,
            model_id=self.model_config.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_amount=cost_str,
        )
        self._domain_events.append(event)

    def fail_response(self, error_message: str, retry_count: int = 0) -> None:
        """Mark the response as failed."""
        if self.is_completed():
            raise ValidationError("Response is already completed")

        self.error_message = error_message
        self.completed_at = datetime.utcnow()

        # Add domain event
        from ..events.provider_events import ModelCallFailed

        event = ModelCallFailed(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=self.id,
            model_id=self.model_config.model_id,
            error_type=type(Exception).__name__,  # This could be improved with actual error types
            error_message=error_message,
            retry_count=retry_count,
        )
        self._domain_events.append(event)

    def calculate_cost(self) -> Money:
        """Calculate cost based on token usage and model pricing."""
        if not self.is_completed() or self.has_error():
            raise CostCalculationError("Cannot calculate cost for incomplete or failed response")

        try:
            cost_amount = self.model_config.calculate_estimated_cost(
                self.input_tokens, self.output_tokens
            )
            return Money(cost_amount, "USD")
        except Exception as e:
            raise CostCalculationError(f"Failed to calculate cost: {str(e)}")

    def is_completed(self) -> bool:
        """Check if response is completed (successfully or with error)."""
        return self.completed_at is not None

    def is_successful(self) -> bool:
        """Check if response completed successfully."""
        return self.is_completed() and not self.has_error()

    def has_error(self) -> bool:
        """Check if response has an error."""
        return self.error_message is not None

    def is_pending(self) -> bool:
        """Check if response is still pending."""
        return not self.is_completed()

    def get_duration_ms(self) -> Optional[int]:
        """Get total duration from creation to completion."""
        if not self.is_completed():
            return None

        duration = (self.completed_at - self.created_at).total_seconds() * 1000
        return int(duration)

    def get_tokens_per_second(self) -> Optional[float]:
        """Get tokens processed per second."""
        if not self.is_successful() or not self.latency_ms:
            return None

        total_tokens = self.input_tokens + self.output_tokens
        if total_tokens == 0:
            return 0.0

        seconds = self.latency_ms / 1000.0
        return total_tokens / seconds if seconds > 0 else 0.0

    def get_cost_per_token(self) -> Optional[Money]:
        """Get average cost per token."""
        if not self.is_successful():
            return None

        total_tokens = self.input_tokens + self.output_tokens
        if total_tokens == 0:
            return Money.zero("USD")

        try:
            total_cost = self.calculate_cost()
            return total_cost / total_tokens
        except CostCalculationError:
            return None

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the response."""
        if not isinstance(key, str):
            raise ValidationError("Metadata key must be a string")

        self.metadata[key] = value

    def get_metadata(self, key: str, default=None):
        """Get metadata value."""
        return self.metadata.get(key, default)

    def get_response_preview(self, max_length: int = 100) -> str:
        """Get a preview of the response text."""
        if not self.response_text:
            return ""

        if len(self.response_text) <= max_length:
            return self.response_text

        return self.response_text[:max_length] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": str(self.id),
            "model_id": self.model_config.model_id,
            "prompt": self.prompt,
            "response_text": self.response_text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "metadata": self.metadata.copy(),
            "is_completed": self.is_completed(),
            "is_successful": self.is_successful(),
            "has_error": self.has_error(),
        }

        # Add cost if calculable
        if self.is_successful():
            try:
                cost = self.calculate_cost()
                result["cost"] = {"amount": str(cost.amount), "currency": cost.currency}
            except CostCalculationError:
                result["cost"] = None

        # Add performance metrics
        duration = self.get_duration_ms()
        if duration is not None:
            result["duration_ms"] = duration

        tokens_per_second = self.get_tokens_per_second()
        if tokens_per_second is not None:
            result["tokens_per_second"] = tokens_per_second

        return result

    def __str__(self) -> str:
        """String representation."""
        status = "completed" if self.is_completed() else "pending"
        if self.has_error():
            status = "failed"

        return f"ModelResponse(id={str(self.id)[:8]}, model={self.model_config.model_id}, status={status})"

    def __eq__(self, other) -> bool:
        """Equality comparison based on ID."""
        if not isinstance(other, ModelResponse):
            return False
        return self.id == other.id
