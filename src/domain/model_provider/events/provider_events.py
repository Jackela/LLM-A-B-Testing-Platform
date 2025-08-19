"""Domain events for provider lifecycle."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

from ..value_objects.health_status import HealthStatus


@dataclass(frozen=True)
class DomainEvent:
    """Base domain event."""

    occurred_at: datetime
    event_id: UUID
    aggregate_id: UUID

    def __post_init__(self):
        """Initialize event timestamp."""
        if self.occurred_at is None:
            object.__setattr__(self, "occurred_at", datetime.utcnow())


@dataclass(frozen=True)
class ProviderCreated(DomainEvent):
    """Event raised when a provider is created."""

    provider_name: str
    provider_type: str


@dataclass(frozen=True)
class ProviderHealthChanged(DomainEvent):
    """Event raised when provider health status changes."""

    old_status: HealthStatus
    new_status: HealthStatus
    check_details: Optional[dict] = None


@dataclass(frozen=True)
class ModelCallRequested(DomainEvent):
    """Event raised when a model call is requested."""

    model_id: str
    prompt_preview: str  # First 100 characters for privacy
    parameters: dict


@dataclass(frozen=True)
class ModelCallCompleted(DomainEvent):
    """Event raised when a model call is completed."""

    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_amount: str  # String representation to avoid decimal serialization issues


@dataclass(frozen=True)
class ModelCallFailed(DomainEvent):
    """Event raised when a model call fails."""

    model_id: str
    error_type: str
    error_message: str
    retry_count: int


@dataclass(frozen=True)
class RateLimitExceeded(DomainEvent):
    """Event raised when rate limit is exceeded."""

    limit_type: str  # "minute" or "day"
    current_count: int
    limit_value: int
    retry_after_seconds: int


@dataclass(frozen=True)
class ProviderCredentialsUpdated(DomainEvent):
    """Event raised when provider credentials are updated."""

    credential_keys: list  # List of credential keys that were updated (no values)
    updated_by: Optional[str] = None


@dataclass(frozen=True)
class ProviderConfigurationChanged(DomainEvent):
    """Event raised when provider configuration changes."""

    changed_fields: list
    previous_values: dict  # Non-sensitive values only
    new_values: dict  # Non-sensitive values only
