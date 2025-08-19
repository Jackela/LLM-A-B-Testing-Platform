"""Model Provider Domain - Supporting Domain for LLM Provider Management.

This domain handles the management of multiple LLM providers, their model configurations,
cost tracking, rate limiting, and health monitoring for the A/B testing platform.
"""

from .entities.model_config import ModelConfig

# Main entities
from .entities.model_provider import ModelProvider
from .entities.model_response import ModelResponse

# Events
from .events.provider_events import (
    ModelCallCompleted,
    ModelCallFailed,
    ModelCallRequested,
    ProviderConfigurationChanged,
    ProviderCreated,
    ProviderCredentialsUpdated,
    ProviderHealthChanged,
    RateLimitExceeded,
)

# Exceptions
from .exceptions import (
    BusinessRuleViolation,
    CostCalculationError,
    InvalidProviderConfiguration,
    ModelNotFound,
    ModelProviderDomainException,
    ProviderHealthCheckFailed,
    ProviderNotFound,
    ProviderNotSupported,
    RateLimitExceeded,
    ValidationError,
)

# Interfaces
from .interfaces.provider_adapter import HealthCheckResult, ProviderAdapter
from .repositories.provider_repository import ProviderRepository
from .services.model_service import ModelService

# Services
from .services.provider_factory import ProviderFactory
from .value_objects.health_status import HealthStatus
from .value_objects.model_category import ModelCategory
from .value_objects.money import Money

# Value objects
from .value_objects.provider_type import ProviderType
from .value_objects.rate_limits import RateLimits
from .value_objects.validation_result import ValidationResult

__all__ = [
    # Entities
    "ModelProvider",
    "ModelConfig",
    "ModelResponse",
    # Value Objects
    "ProviderType",
    "HealthStatus",
    "ModelCategory",
    "RateLimits",
    "Money",
    "ValidationResult",
    # Services
    "ProviderFactory",
    "ModelService",
    # Interfaces
    "ProviderAdapter",
    "HealthCheckResult",
    "ProviderRepository",
    # Events
    "ProviderCreated",
    "ProviderHealthChanged",
    "ModelCallRequested",
    "ModelCallCompleted",
    "ModelCallFailed",
    "RateLimitExceeded",
    "ProviderCredentialsUpdated",
    "ProviderConfigurationChanged",
    # Exceptions
    "ModelProviderDomainException",
    "BusinessRuleViolation",
    "ValidationError",
    "ModelNotFound",
    "ProviderNotFound",
    "ProviderNotSupported",
    "InvalidProviderConfiguration",
    "RateLimitExceeded",
    "ProviderHealthCheckFailed",
    "CostCalculationError",
]
