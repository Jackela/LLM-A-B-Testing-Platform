"""Configurable retry service with exponential backoff for model provider calls."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type

from .error_handler import (
    NetworkError,
    ProviderError,
    ProviderErrorType,
    ProviderOverloadedError,
    RateLimitExceededError,
    TimeoutError,
)

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to delays
    jitter_factor: float = 0.2  # Jitter range (±20%)


@dataclass
class RetryContext:
    """Context information for retry operations."""

    operation_name: str
    provider_name: str
    model_id: Optional[str] = None
    request_id: Optional[str] = None
    start_time: Optional[datetime] = None
    total_timeout: Optional[float] = None  # Maximum total time for all retries


class RetryStats:
    """Statistics for retry operations."""

    def __init__(self):
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.total_delay_time = 0.0
        self.retry_by_error_type: Dict[ProviderErrorType, int] = {}
        self.last_attempt_time: Optional[datetime] = None

    def record_attempt(self, success: bool, error_type: Optional[ProviderErrorType] = None):
        """Record an attempt."""
        self.total_attempts += 1
        self.last_attempt_time = datetime.utcnow()

        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
            if error_type:
                self.retry_by_error_type[error_type] = (
                    self.retry_by_error_type.get(error_type, 0) + 1
                )

    def record_delay(self, delay_seconds: float):
        """Record delay time."""
        self.total_delay_time += delay_seconds

    def get_success_rate(self) -> float:
        """Get success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts


class MaxRetriesExceededError(Exception):
    """Exception raised when maximum retries are exceeded."""

    def __init__(self, message: str, last_error: Exception, retry_stats: RetryStats):
        super().__init__(message)
        self.last_error = last_error
        self.retry_stats = retry_stats


class RetryTimeoutError(Exception):
    """Exception raised when total retry timeout is exceeded."""

    def __init__(self, message: str, elapsed_time: float, retry_stats: RetryStats):
        super().__init__(message)
        self.elapsed_time = elapsed_time
        self.retry_stats = retry_stats


class RetryService:
    """Service for executing operations with configurable retry logic."""

    # Define which error types are retryable
    RETRYABLE_ERROR_TYPES = {
        ProviderErrorType.RATE_LIMIT_EXCEEDED,
        ProviderErrorType.PROVIDER_OVERLOADED,
        ProviderErrorType.NETWORK_ERROR,
        ProviderErrorType.TIMEOUT_ERROR,
        ProviderErrorType.UNKNOWN_ERROR,  # Some unknown errors might be transient
    }

    # Error-specific retry configurations
    ERROR_TYPE_CONFIGS = {
        ProviderErrorType.RATE_LIMIT_EXCEEDED: RetryConfig(
            max_retries=5, base_delay=60.0, max_delay=300.0, backoff_factor=1.5
        ),
        ProviderErrorType.PROVIDER_OVERLOADED: RetryConfig(
            max_retries=4, base_delay=30.0, max_delay=120.0, backoff_factor=2.0
        ),
        ProviderErrorType.NETWORK_ERROR: RetryConfig(
            max_retries=3, base_delay=5.0, max_delay=30.0, backoff_factor=2.0
        ),
        ProviderErrorType.TIMEOUT_ERROR: RetryConfig(
            max_retries=2, base_delay=10.0, max_delay=60.0, backoff_factor=2.0
        ),
    }

    def __init__(self, default_config: Optional[RetryConfig] = None):
        self.default_config = default_config or RetryConfig()
        self.logger = logger

    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        context: RetryContext,
        retry_config: Optional[RetryConfig] = None,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute operation with retry logic.

        Args:
            operation: Async function to execute
            context: Context information for the retry operation
            retry_config: Optional retry configuration (uses default if not provided)
            retryable_exceptions: Additional exception types to retry on
            *args, **kwargs: Arguments to pass to the operation

        Returns:
            Result of the successful operation

        Raises:
            MaxRetriesExceededError: When all retries are exhausted
            RetryTimeoutError: When total timeout is exceeded
        """
        config = retry_config or self.default_config
        stats = RetryStats()
        last_exception = None
        start_time = time.time()

        # Set context start time if not provided
        if context.start_time is None:
            context.start_time = datetime.utcnow()

        for attempt in range(config.max_retries + 1):  # +1 for initial attempt
            try:
                # Check total timeout
                if context.total_timeout:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > context.total_timeout:
                        raise RetryTimeoutError(
                            f"Total retry timeout ({context.total_timeout}s) exceeded for {context.operation_name}",
                            elapsed_time,
                            stats,
                        )

                self.logger.debug(
                    f"Attempting {context.operation_name} (attempt {attempt + 1}/{config.max_retries + 1})",
                    extra={"context": context.__dict__, "attempt": attempt + 1},
                )

                # Execute the operation
                result = await operation(*args, **kwargs)

                # Success!
                stats.record_attempt(True)

                if attempt > 0:
                    self.logger.info(
                        f"Operation {context.operation_name} succeeded after {attempt + 1} attempts",
                        extra={
                            "context": context.__dict__,
                            "attempts": attempt + 1,
                            "stats": stats.__dict__,
                        },
                    )

                return result

            except Exception as e:
                last_exception = e
                stats.record_attempt(False)

                # Determine if this error is retryable
                if not self._is_retryable_error(e, retryable_exceptions):
                    self.logger.warning(
                        f"Non-retryable error in {context.operation_name}: {e}",
                        extra={"context": context.__dict__, "error_type": type(e).__name__},
                    )
                    raise e

                # If this was the last attempt, don't retry
                if attempt >= config.max_retries:
                    break

                # Calculate delay for next attempt
                delay = self._calculate_delay(e, attempt, config)
                stats.record_delay(delay)

                self.logger.warning(
                    f"Retryable error in {context.operation_name} (attempt {attempt + 1}): {e}. "
                    f"Retrying in {delay:.2f} seconds...",
                    extra={
                        "context": context.__dict__,
                        "attempt": attempt + 1,
                        "delay": delay,
                        "error_type": type(e).__name__,
                    },
                )

                # Wait before retrying
                await asyncio.sleep(delay)

        # All retries exhausted
        error_type = (
            getattr(last_exception, "error_type", None)
            if isinstance(last_exception, ProviderError)
            else None
        )
        if error_type:
            stats.retry_by_error_type[error_type] = stats.retry_by_error_type.get(error_type, 0) + 1

        self.logger.error(
            f"All retries exhausted for {context.operation_name}. Last error: {last_exception}",
            extra={"context": context.__dict__, "stats": stats.__dict__},
        )

        raise MaxRetriesExceededError(
            f"Operation {context.operation_name} failed after {config.max_retries + 1} attempts",
            last_exception,
            stats,
        )

    def _is_retryable_error(
        self, error: Exception, additional_retryable: Optional[List[Type[Exception]]] = None
    ) -> bool:
        """Determine if an error is retryable."""

        # Check provider-specific error types
        if isinstance(error, ProviderError):
            return error.error_type in self.RETRYABLE_ERROR_TYPES

        # Check additional retryable exceptions
        if additional_retryable:
            for exception_type in additional_retryable:
                if isinstance(error, exception_type):
                    return True

        # Check for specific exception types that are generally retryable
        retryable_exception_types = (
            RateLimitExceededError,
            ProviderOverloadedError,
            NetworkError,
            TimeoutError,
            ConnectionError,
            asyncio.TimeoutError,
        )

        return isinstance(error, retryable_exception_types)

    def _calculate_delay(self, error: Exception, attempt: int, config: RetryConfig) -> float:
        """Calculate delay before next retry attempt."""

        # Use error-specific configuration if available
        if isinstance(error, ProviderError):
            error_config = self.ERROR_TYPE_CONFIGS.get(error.error_type)
            if error_config:
                config = error_config

        # Check for rate limit specific retry-after
        if isinstance(error, (RateLimitExceededError, ProviderError)):
            if hasattr(error, "retry_after") and error.retry_after:
                base_delay = float(error.retry_after)
            else:
                base_delay = config.base_delay
        else:
            base_delay = config.base_delay

        # Calculate exponential backoff
        exponential_delay = base_delay * (config.backoff_factor**attempt)

        # Apply jitter if enabled
        if config.jitter:
            jitter_range = exponential_delay * config.jitter_factor
            jitter = random.uniform(-jitter_range, jitter_range)
            delay = exponential_delay + jitter
        else:
            delay = exponential_delay

        # Ensure delay is within bounds
        return max(0.1, min(delay, config.max_delay))

    def get_retry_config_for_error(self, error: Exception) -> RetryConfig:
        """Get appropriate retry configuration for a specific error."""
        if isinstance(error, ProviderError):
            return self.ERROR_TYPE_CONFIGS.get(error.error_type, self.default_config)
        return self.default_config

    async def execute_with_adaptive_retry(
        self, operation: Callable[..., Any], context: RetryContext, *args, **kwargs
    ) -> Any:
        """
        Execute operation with adaptive retry logic that adjusts based on error types.
        """
        last_exception = None

        try:
            # First attempt with no delay
            return await operation(*args, **kwargs)
        except Exception as e:
            last_exception = e

            # If not retryable, fail immediately
            if not self._is_retryable_error(e):
                raise e

            # Get adaptive configuration based on error type
            adaptive_config = self.get_retry_config_for_error(e)

            self.logger.info(
                f"Using adaptive retry config for {context.operation_name}: "
                f"max_retries={adaptive_config.max_retries}, "
                f"base_delay={adaptive_config.base_delay}",
                extra={"context": context.__dict__, "error_type": type(e).__name__},
            )

            # Execute with adaptive configuration
            return await self.execute_with_retry(
                operation, context, adaptive_config, None, *args, **kwargs
            )

    def create_context(
        self,
        operation_name: str,
        provider_name: str,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        total_timeout: Optional[float] = None,
    ) -> RetryContext:
        """Helper method to create retry context."""
        return RetryContext(
            operation_name=operation_name,
            provider_name=provider_name,
            model_id=model_id,
            request_id=request_id,
            start_time=datetime.utcnow(),
            total_timeout=total_timeout,
        )
