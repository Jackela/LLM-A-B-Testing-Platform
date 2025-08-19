"""Domain exceptions for Model Provider."""

from typing import Any, Dict, Optional


class ModelProviderDomainException(Exception):
    """Base exception for Model Provider domain."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class BusinessRuleViolation(ModelProviderDomainException):
    """Exception raised when business rules are violated."""

    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.rule_name = rule_name


class ValidationError(ModelProviderDomainException):
    """Exception raised when validation fails."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.field_name = field_name


class ModelNotFound(ModelProviderDomainException):
    """Exception raised when model is not found."""

    def __init__(self, message: str, model_id: Optional[str] = None):
        super().__init__(message)
        self.model_id = model_id


class ProviderNotFound(ModelProviderDomainException):
    """Exception raised when provider is not found."""

    def __init__(self, message: str, provider_id: Optional[str] = None):
        super().__init__(message)
        self.provider_id = provider_id


class ProviderNotSupported(ModelProviderDomainException):
    """Exception raised when provider type is not supported."""

    def __init__(self, message: str, provider_type: Optional[str] = None):
        super().__init__(message)
        self.provider_type = provider_type


class InvalidProviderConfiguration(ModelProviderDomainException):
    """Exception raised when provider configuration is invalid."""

    pass


class RateLimitExceeded(ModelProviderDomainException):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        retry_after_seconds: Optional[int] = None,
    ):
        super().__init__(message)
        self.provider_name = provider_name
        self.retry_after_seconds = retry_after_seconds


class ProviderHealthCheckFailed(ModelProviderDomainException):
    """Exception raised when provider health check fails."""

    pass


class CostCalculationError(ModelProviderDomainException):
    """Exception raised when cost calculation fails."""

    pass
