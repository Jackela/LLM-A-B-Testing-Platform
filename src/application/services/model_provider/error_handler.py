"""Comprehensive error handling for model provider integration."""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class ProviderErrorType(Enum):
    """Categorized provider error types."""

    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MODEL_NOT_FOUND = "model_not_found"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    PROVIDER_OVERLOADED = "provider_overloaded"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    INVALID_REQUEST = "invalid_request"
    QUOTA_EXCEEDED = "quota_exceeded"
    UNKNOWN_ERROR = "unknown_error"


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    def __init__(
        self,
        message: str,
        error_type: ProviderErrorType,
        provider_name: str = None,
        model_id: str = None,
        retry_after: Optional[int] = None,
        original_error: Exception = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.provider_name = provider_name
        self.model_id = model_id
        self.retry_after = retry_after
        self.original_error = original_error


class AuthenticationError(ProviderError):
    """Authentication/authorization error."""

    def __init__(self, message: str, provider_name: str = None, original_error: Exception = None):
        super().__init__(
            message,
            ProviderErrorType.AUTHENTICATION_ERROR,
            provider_name=provider_name,
            original_error=original_error,
        )


class RateLimitExceededError(ProviderError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str,
        provider_name: str = None,
        retry_after: int = None,
        original_error: Exception = None,
    ):
        super().__init__(
            message,
            ProviderErrorType.RATE_LIMIT_EXCEEDED,
            provider_name=provider_name,
            retry_after=retry_after,
            original_error=original_error,
        )


class ModelNotFoundError(ProviderError):
    """Model not found error."""

    def __init__(
        self,
        message: str,
        provider_name: str = None,
        model_id: str = None,
        original_error: Exception = None,
    ):
        super().__init__(
            message,
            ProviderErrorType.MODEL_NOT_FOUND,
            provider_name=provider_name,
            model_id=model_id,
            original_error=original_error,
        )


class TokenLimitExceededError(ProviderError):
    """Token limit exceeded error."""

    def __init__(
        self,
        message: str,
        provider_name: str = None,
        model_id: str = None,
        original_error: Exception = None,
    ):
        super().__init__(
            message,
            ProviderErrorType.TOKEN_LIMIT_EXCEEDED,
            provider_name=provider_name,
            model_id=model_id,
            original_error=original_error,
        )


class ProviderOverloadedError(ProviderError):
    """Provider is overloaded error."""

    def __init__(
        self,
        message: str,
        provider_name: str = None,
        retry_after: int = None,
        original_error: Exception = None,
    ):
        super().__init__(
            message,
            ProviderErrorType.PROVIDER_OVERLOADED,
            provider_name=provider_name,
            retry_after=retry_after,
            original_error=original_error,
        )


class NetworkError(ProviderError):
    """Network-related error."""

    def __init__(self, message: str, provider_name: str = None, original_error: Exception = None):
        super().__init__(
            message,
            ProviderErrorType.NETWORK_ERROR,
            provider_name=provider_name,
            original_error=original_error,
        )


class TimeoutError(ProviderError):
    """Request timeout error."""

    def __init__(self, message: str, provider_name: str = None, original_error: Exception = None):
        super().__init__(
            message,
            ProviderErrorType.TIMEOUT_ERROR,
            provider_name=provider_name,
            original_error=original_error,
        )


class ErrorHandler:
    """Comprehensive error handler for model provider integration."""

    # OpenAI error mappings
    OPENAI_ERROR_MAPPINGS = {
        "invalid_api_key": AuthenticationError,
        "invalid_request_error": lambda msg, **kwargs: ProviderError(
            msg, ProviderErrorType.INVALID_REQUEST, **kwargs
        ),
        "rate_limit_exceeded": RateLimitExceededError,
        "quota_exceeded": lambda msg, **kwargs: ProviderError(
            msg, ProviderErrorType.QUOTA_EXCEEDED, **kwargs
        ),
        "model_not_found": ModelNotFoundError,
        "context_length_exceeded": TokenLimitExceededError,
        "server_error": ProviderOverloadedError,
        "service_unavailable": ProviderOverloadedError,
        "timeout": TimeoutError,
    }

    # Anthropic error mappings
    ANTHROPIC_ERROR_MAPPINGS = {
        "invalid_request_error": lambda msg, **kwargs: ProviderError(
            msg, ProviderErrorType.INVALID_REQUEST, **kwargs
        ),
        "authentication_error": AuthenticationError,
        "rate_limit_error": RateLimitExceededError,
        "overloaded_error": ProviderOverloadedError,
        "api_error": ProviderOverloadedError,
        "timeout_error": TimeoutError,
    }

    # Google error mappings
    GOOGLE_ERROR_MAPPINGS = {
        "UNAUTHENTICATED": AuthenticationError,
        "PERMISSION_DENIED": AuthenticationError,
        "RESOURCE_EXHAUSTED": RateLimitExceededError,
        "INVALID_ARGUMENT": lambda msg, **kwargs: ProviderError(
            msg, ProviderErrorType.INVALID_REQUEST, **kwargs
        ),
        "NOT_FOUND": ModelNotFoundError,
        "UNAVAILABLE": ProviderOverloadedError,
        "DEADLINE_EXCEEDED": TimeoutError,
    }

    # HTTP status code mappings
    HTTP_STATUS_MAPPINGS = {
        400: ProviderErrorType.INVALID_REQUEST,
        401: ProviderErrorType.AUTHENTICATION_ERROR,
        403: ProviderErrorType.AUTHENTICATION_ERROR,
        404: ProviderErrorType.MODEL_NOT_FOUND,
        429: ProviderErrorType.RATE_LIMIT_EXCEEDED,
        500: ProviderErrorType.PROVIDER_OVERLOADED,
        502: ProviderErrorType.PROVIDER_OVERLOADED,
        503: ProviderErrorType.PROVIDER_OVERLOADED,
        504: ProviderErrorType.TIMEOUT_ERROR,
    }

    def __init__(self):
        self.logger = logger

    def handle_provider_error(
        self,
        error: Exception,
        provider_name: str,
        model_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProviderError:
        """Convert provider-specific errors to application errors."""

        # Log the original error
        self.logger.error(
            f"Provider error from {provider_name}: {error}",
            extra={
                "provider": provider_name,
                "model": model_id,
                "error_type": type(error).__name__,
                "context": context or {},
            },
        )

        # Handle OpenAI errors
        if self._is_openai_error(error):
            return self._handle_openai_error(error, provider_name, model_id)

        # Handle Anthropic errors
        elif self._is_anthropic_error(error):
            return self._handle_anthropic_error(error, provider_name, model_id)

        # Handle Google errors
        elif self._is_google_error(error):
            return self._handle_google_error(error, provider_name, model_id)

        # Handle HTTP errors
        elif hasattr(error, "status_code"):
            return self._handle_http_error(error, provider_name, model_id)

        # Handle network/timeout errors
        elif self._is_network_error(error):
            return NetworkError(
                f"Network error when calling {provider_name}: {str(error)}",
                provider_name=provider_name,
                original_error=error,
            )

        elif self._is_timeout_error(error):
            return TimeoutError(
                f"Timeout when calling {provider_name}: {str(error)}",
                provider_name=provider_name,
                original_error=error,
            )

        # Default to unknown error
        return ProviderError(
            f"Unknown error from {provider_name}: {str(error)}",
            ProviderErrorType.UNKNOWN_ERROR,
            provider_name=provider_name,
            model_id=model_id,
            original_error=error,
        )

    def _is_openai_error(self, error: Exception) -> bool:
        """Check if error is from OpenAI."""
        return (
            "openai" in error.__class__.__module__.lower() if error.__class__.__module__ else False
        )

    def _is_anthropic_error(self, error: Exception) -> bool:
        """Check if error is from Anthropic."""
        return (
            "anthropic" in error.__class__.__module__.lower()
            if error.__class__.__module__
            else False
        )

    def _is_google_error(self, error: Exception) -> bool:
        """Check if error is from Google."""
        return (
            "google" in error.__class__.__module__.lower() if error.__class__.__module__ else False
        )

    def _is_network_error(self, error: Exception) -> bool:
        """Check if error is network-related."""
        network_error_types = [
            "ConnectionError",
            "ConnectTimeout",
            "ReadTimeout",
            "NetworkError",
            "HTTPError",
            "URLError",
        ]
        return any(err_type in str(type(error)) for err_type in network_error_types)

    def _is_timeout_error(self, error: Exception) -> bool:
        """Check if error is timeout-related."""
        timeout_indicators = ["timeout", "deadline", "TimeoutError"]
        return any(indicator.lower() in str(error).lower() for indicator in timeout_indicators)

    def _handle_openai_error(
        self, error: Exception, provider_name: str, model_id: Optional[str]
    ) -> ProviderError:
        """Handle OpenAI-specific errors."""
        error_message = str(error).lower()

        for error_key, error_class in self.OPENAI_ERROR_MAPPINGS.items():
            if error_key.lower() in error_message:
                if callable(error_class):
                    return error_class(
                        str(error),
                        provider_name=provider_name,
                        model_id=model_id,
                        original_error=error,
                    )
                else:
                    return error_class(
                        str(error), provider_name=provider_name, original_error=error
                    )

        # Check for rate limit with retry-after
        if "rate limit" in error_message:
            retry_after = self._extract_retry_after(str(error))
            return RateLimitExceededError(
                str(error),
                provider_name=provider_name,
                retry_after=retry_after,
                original_error=error,
            )

        return ProviderError(
            str(error),
            ProviderErrorType.UNKNOWN_ERROR,
            provider_name=provider_name,
            model_id=model_id,
            original_error=error,
        )

    def _handle_anthropic_error(
        self, error: Exception, provider_name: str, model_id: Optional[str]
    ) -> ProviderError:
        """Handle Anthropic-specific errors."""
        error_message = str(error).lower()

        for error_key, error_class in self.ANTHROPIC_ERROR_MAPPINGS.items():
            if error_key.lower() in error_message:
                if callable(error_class):
                    return error_class(
                        str(error),
                        provider_name=provider_name,
                        model_id=model_id,
                        original_error=error,
                    )
                else:
                    return error_class(
                        str(error), provider_name=provider_name, original_error=error
                    )

        return ProviderError(
            str(error),
            ProviderErrorType.UNKNOWN_ERROR,
            provider_name=provider_name,
            model_id=model_id,
            original_error=error,
        )

    def _handle_google_error(
        self, error: Exception, provider_name: str, model_id: Optional[str]
    ) -> ProviderError:
        """Handle Google-specific errors."""
        error_message = str(error).upper()

        for error_key, error_class in self.GOOGLE_ERROR_MAPPINGS.items():
            if error_key in error_message:
                if callable(error_class):
                    return error_class(
                        str(error),
                        provider_name=provider_name,
                        model_id=model_id,
                        original_error=error,
                    )
                else:
                    return error_class(
                        str(error), provider_name=provider_name, original_error=error
                    )

        return ProviderError(
            str(error),
            ProviderErrorType.UNKNOWN_ERROR,
            provider_name=provider_name,
            model_id=model_id,
            original_error=error,
        )

    def _handle_http_error(
        self, error: Exception, provider_name: str, model_id: Optional[str]
    ) -> ProviderError:
        """Handle HTTP status code errors."""
        status_code = getattr(error, "status_code", None)
        error_type = self.HTTP_STATUS_MAPPINGS.get(status_code, ProviderErrorType.UNKNOWN_ERROR)

        # Extract retry-after header for rate limits
        retry_after = None
        if status_code == 429:
            retry_after = self._extract_retry_after_from_headers(error)

        if error_type == ProviderErrorType.RATE_LIMIT_EXCEEDED:
            return RateLimitExceededError(
                f"HTTP {status_code}: {str(error)}",
                provider_name=provider_name,
                retry_after=retry_after,
                original_error=error,
            )
        elif error_type == ProviderErrorType.AUTHENTICATION_ERROR:
            return AuthenticationError(
                f"HTTP {status_code}: {str(error)}",
                provider_name=provider_name,
                original_error=error,
            )
        else:
            return ProviderError(
                f"HTTP {status_code}: {str(error)}",
                error_type,
                provider_name=provider_name,
                model_id=model_id,
                original_error=error,
            )

    def _extract_retry_after(self, error_message: str) -> Optional[int]:
        """Extract retry-after value from error message."""
        import re

        # Common patterns for retry-after in error messages
        patterns = [
            r"retry after (\d+) seconds",
            r"try again in (\d+) seconds",
            r"wait (\d+) seconds",
            r"retry-after: (\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message.lower())
            if match:
                return int(match.group(1))

        return None

    def _extract_retry_after_from_headers(self, error: Exception) -> Optional[int]:
        """Extract retry-after value from HTTP headers."""
        if hasattr(error, "response") and hasattr(error.response, "headers"):
            retry_after = error.response.headers.get("retry-after")
            if retry_after:
                try:
                    return int(retry_after)
                except ValueError:
                    pass
        return None

    def is_retryable_error(self, error: ProviderError) -> bool:
        """Determine if an error is retryable."""
        retryable_types = [
            ProviderErrorType.RATE_LIMIT_EXCEEDED,
            ProviderErrorType.PROVIDER_OVERLOADED,
            ProviderErrorType.NETWORK_ERROR,
            ProviderErrorType.TIMEOUT_ERROR,
        ]
        return error.error_type in retryable_types

    def get_retry_delay(self, error: ProviderError, attempt: int) -> float:
        """Get recommended retry delay based on error type and attempt number."""
        base_delays = {
            ProviderErrorType.RATE_LIMIT_EXCEEDED: 60.0,  # 1 minute base
            ProviderErrorType.PROVIDER_OVERLOADED: 30.0,  # 30 seconds base
            ProviderErrorType.NETWORK_ERROR: 5.0,  # 5 seconds base
            ProviderErrorType.TIMEOUT_ERROR: 10.0,  # 10 seconds base
        }

        base_delay = base_delays.get(error.error_type, 5.0)

        # Use retry-after if provided
        if error.retry_after:
            return float(error.retry_after)

        # Exponential backoff with jitter
        import random

        exponential_delay = base_delay * (2 ** (attempt - 1))
        jitter = random.uniform(0.8, 1.2)  # ±20% jitter

        return min(exponential_delay * jitter, 300.0)  # Cap at 5 minutes
