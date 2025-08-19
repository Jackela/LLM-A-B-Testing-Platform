"""Model provider aggregate root."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..exceptions import (
    BusinessRuleViolation,
    ModelNotFound,
    RateLimitExceeded,
    ValidationError,
)
from ..value_objects.health_status import HealthStatus
from ..value_objects.provider_type import ProviderType
from ..value_objects.rate_limits import RateLimits
from .model_config import ModelConfig
from .model_response import ModelResponse


@dataclass
class ModelProvider:
    """Model Provider aggregate root."""

    id: UUID
    name: str
    provider_type: ProviderType
    supported_models: List[ModelConfig]
    rate_limits: RateLimits
    health_status: HealthStatus
    api_credentials: Dict[str, str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _domain_events: List[object] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        """Initialize provider after creation."""
        if not self.id:
            self.id = uuid4()

        # Validate basic invariants
        self._validate_basic_rules()

    @classmethod
    def create(
        cls,
        name: str,
        provider_type: ProviderType,
        supported_models: List[ModelConfig],
        api_credentials: Dict[str, str],
    ) -> "ModelProvider":
        """Factory method for creating providers."""

        # Validate business rules
        if not supported_models:
            raise BusinessRuleViolation(
                "Provider must have at least one supported model", rule_name="provider_has_models"
            )

        if not name.strip():
            raise ValidationError("Provider name cannot be empty")

        if not isinstance(provider_type, ProviderType):
            raise ValidationError("Provider type must be a valid ProviderType")

        if not api_credentials:
            raise ValidationError("API credentials are required")

        # Validate all model configurations
        for model in supported_models:
            if not isinstance(model, ModelConfig):
                raise ValidationError("All supported models must be ModelConfig instances")
            model.validate()  # This will raise if invalid

        # Create provider instance
        provider = cls(
            id=uuid4(),
            name=name.strip(),
            provider_type=provider_type,
            supported_models=supported_models.copy(),
            rate_limits=RateLimits.default_for_provider(provider_type),
            health_status=HealthStatus.UNKNOWN,
            api_credentials=api_credentials.copy(),
        )

        # Add domain event
        from ..events.provider_events import ProviderCreated

        event = ProviderCreated(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=provider.id,
            provider_name=provider.name,
            provider_type=provider.provider_type.value,
        )
        provider._domain_events.append(event)

        return provider

    def call_model(self, model_id: str, prompt: str, **parameters) -> ModelResponse:
        """Call specific model with parameters."""

        # Find model configuration
        model_config = self.find_model_config(model_id)
        if not model_config:
            raise ModelNotFound(
                f"Model {model_id} not found in provider {self.name}", model_id=model_id
            )

        # Check rate limits
        if not self.rate_limits.can_make_request():
            raise RateLimitExceeded(
                f"Rate limit exceeded for provider {self.name}",
                provider_name=self.name,
                retry_after_seconds=self.rate_limits.time_until_reset(),
            )

        # Check health status
        if not self.health_status.is_operational:
            raise ValidationError(
                f"Provider {self.name} is not operational (status: {self.health_status.name})"
            )

        # Validate parameters against model config
        self._validate_call_parameters(model_config, parameters)

        # Update rate limits
        self.rate_limits.record_request()
        self.updated_at = datetime.utcnow()

        # Create pending response (actual call happens in infrastructure layer)
        response = ModelResponse.create_pending(model_config, prompt)

        return response

    def find_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Find model configuration by ID."""
        return next((m for m in self.supported_models if m.model_id == model_id), None)

    def update_health_status(
        self, status: HealthStatus, check_details: Optional[dict] = None
    ) -> None:
        """Update provider health status."""
        if not isinstance(status, HealthStatus):
            raise ValidationError("Status must be a valid HealthStatus")

        if self.health_status != status:
            old_status = self.health_status
            self.health_status = status
            self.updated_at = datetime.utcnow()

            # Add domain event
            from ..events.provider_events import ProviderHealthChanged

            event = ProviderHealthChanged(
                occurred_at=datetime.utcnow(),
                event_id=uuid4(),
                aggregate_id=self.id,
                old_status=old_status,
                new_status=status,
                check_details=check_details,
            )
            self._domain_events.append(event)

    def update_credentials(
        self, new_credentials: Dict[str, str], updated_by: Optional[str] = None
    ) -> None:
        """Update API credentials."""
        if not new_credentials:
            raise ValidationError("New credentials cannot be empty")

        # Validate required keys for provider type
        required_keys = self._get_required_credential_keys()
        missing_keys = [key for key in required_keys if key not in new_credentials]
        if missing_keys:
            raise ValidationError(f"Missing required credential keys: {missing_keys}")

        self.api_credentials = new_credentials.copy()
        self.updated_at = datetime.utcnow()

        # Add domain event (without credential values for security)
        from ..events.provider_events import ProviderCredentialsUpdated

        event = ProviderCredentialsUpdated(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=self.id,
            credential_keys=list(new_credentials.keys()),
            updated_by=updated_by,
        )
        self._domain_events.append(event)

    def add_model(self, model_config: ModelConfig) -> None:
        """Add a new supported model."""
        if not isinstance(model_config, ModelConfig):
            raise ValidationError("Model config must be a ModelConfig instance")

        # Check if model already exists
        if self.find_model_config(model_config.model_id):
            raise ValidationError(f"Model {model_config.model_id} already exists")

        model_config.validate()  # Ensure it's valid
        self.supported_models.append(model_config)
        self.updated_at = datetime.utcnow()

    def remove_model(self, model_id: str) -> bool:
        """Remove a supported model."""
        original_count = len(self.supported_models)
        self.supported_models = [m for m in self.supported_models if m.model_id != model_id]

        removed = len(self.supported_models) < original_count
        if removed:
            # Ensure we still have at least one model
            if not self.supported_models:
                raise BusinessRuleViolation(
                    "Cannot remove last model from provider", rule_name="provider_has_models"
                )
            self.updated_at = datetime.utcnow()

        return removed

    def update_rate_limits(self, new_rate_limits: RateLimits) -> None:
        """Update rate limiting configuration."""
        if not isinstance(new_rate_limits, RateLimits):
            raise ValidationError("Rate limits must be a RateLimits instance")

        self.rate_limits = new_rate_limits
        self.updated_at = datetime.utcnow()

    def reset_rate_limits(self) -> None:
        """Reset rate limit counters."""
        self.rate_limits.reset_minute_counter()
        self.rate_limits.reset_day_counter()
        self.updated_at = datetime.utcnow()

    def get_models_by_category(self, category) -> List[ModelConfig]:
        """Get models filtered by category."""
        return [m for m in self.supported_models if m.model_category == category]

    def get_streaming_models(self) -> List[ModelConfig]:
        """Get models that support streaming."""
        return [m for m in self.supported_models if m.supports_streaming]

    def calculate_total_capacity(self) -> int:
        """Calculate total token capacity across all models."""
        return sum(model.max_tokens for model in self.supported_models)

    def get_cheapest_model(self) -> Optional[ModelConfig]:
        """Get the model with lowest input token cost."""
        if not self.supported_models:
            return None

        return min(self.supported_models, key=lambda m: m.cost_per_input_token)

    def get_most_capable_model(self) -> Optional[ModelConfig]:
        """Get the model with highest token capacity."""
        if not self.supported_models:
            return None

        return max(self.supported_models, key=lambda m: m.max_tokens)

    def is_model_available(self, model_id: str) -> bool:
        """Check if a specific model is available and operational."""
        model_exists = self.find_model_config(model_id) is not None
        provider_operational = self.health_status.is_operational
        rate_limit_ok = self.rate_limits.can_make_request()

        return model_exists and provider_operational and rate_limit_ok

    def _validate_basic_rules(self) -> None:
        """Validate basic business rules."""
        if not self.supported_models:
            raise BusinessRuleViolation(
                "Provider must have at least one supported model", rule_name="provider_has_models"
            )

        if not self.name.strip():
            raise ValidationError("Provider name cannot be empty")

    def _validate_call_parameters(self, model_config: ModelConfig, parameters: dict) -> None:
        """Validate parameters for model call."""
        for param_name, param_value in parameters.items():
            if not model_config.supports_parameter(param_name):
                raise ValidationError(
                    f"Parameter '{param_name}' not supported by model {model_config.model_id}"
                )

        # Validate specific parameter values
        if "max_tokens" in parameters:
            max_tokens = parameters["max_tokens"]
            if max_tokens > model_config.max_tokens:
                raise ValidationError(
                    f"Requested max_tokens ({max_tokens}) exceeds model limit ({model_config.max_tokens})"
                )

    def _get_required_credential_keys(self) -> List[str]:
        """Get required credential keys for provider type."""
        required_keys = {
            ProviderType.OPENAI: ["api_key"],
            ProviderType.ANTHROPIC: ["api_key"],
            ProviderType.GOOGLE: ["api_key"],
            ProviderType.BAIDU: ["api_key"],
            ProviderType.ALIBABA: ["api_key"],
        }
        return required_keys.get(self.provider_type, ["api_key"])

    def _add_domain_event(self, event) -> None:
        """Add domain event to the list."""
        self._domain_events.append(event)

    def clear_domain_events(self) -> List[object]:
        """Clear and return domain events."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events

    def to_dict(self, include_credentials: bool = False) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": str(self.id),
            "name": self.name,
            "provider_type": self.provider_type.value,
            "supported_models": [model.to_dict() for model in self.supported_models],
            "rate_limits": self.rate_limits.to_dict(),
            "health_status": self.health_status.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata.copy(),
            "model_count": len(self.supported_models),
            "is_operational": self.health_status.is_operational,
        }

        if include_credentials:
            # In production, this should be carefully controlled
            result["api_credentials"] = self.api_credentials.copy()
        else:
            # Show only the keys for security
            result["credential_keys"] = list(self.api_credentials.keys())

        return result

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ModelProvider(name='{self.name}', type={self.provider_type.name}, "
            f"models={len(self.supported_models)}, status={self.health_status.name})"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison based on ID."""
        if not isinstance(other, ModelProvider):
            return False
        return self.id == other.id
