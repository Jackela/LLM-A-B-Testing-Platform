"""Integration tests for reliability patterns (circuit breaker, retry, error handling)."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.application.services.model_provider.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerFactory,
    CircuitBreakerOpenException,
    CircuitBreakerState,
)
from src.application.services.model_provider.error_handler import (
    AuthenticationError,
    ErrorHandler,
    NetworkError,
    ProviderError,
    ProviderErrorType,
    RateLimitExceededError,
)
from src.application.services.model_provider.retry_service import (
    MaxRetriesExceededError,
    RetryConfig,
    RetryContext,
    RetryService,
)


class TestCircuitBreakerIntegration:
    """Test circuit breaker reliability patterns."""

    @pytest.fixture
    def circuit_breaker_config(self):
        """Create test circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for tests
            success_threshold=2,
            timeout_seconds=5.0,
        )

    @pytest.fixture
    def circuit_breaker(self, circuit_breaker_config):
        """Create circuit breaker instance."""
        return CircuitBreaker("test-breaker", circuit_breaker_config)

    @pytest.mark.asyncio
    async def test_circuit_breaker_success_flow(self, circuit_breaker):
        """Test successful operations through circuit breaker."""

        async def successful_operation():
            return "success"

        # Execute successful operation
        result = await circuit_breaker.execute(successful_operation)

        # Verify
        assert result == "success"
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED

        metrics = circuit_breaker.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold."""
        failure_count = 0

        async def failing_operation():
            nonlocal failure_count
            failure_count += 1
            raise ProviderError("Simulated failure", ProviderErrorType.PROVIDER_OVERLOADED)

        # Execute operations until circuit opens
        for i in range(3):  # failure_threshold = 3
            with pytest.raises(ProviderError):
                await circuit_breaker.execute(failing_operation)

        # Circuit should now be open
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN

        # Next request should fail fast
        with pytest.raises(CircuitBreakerOpenException):
            await circuit_breaker.execute(failing_operation)

        # Verify failure count didn't increase (failed fast)
        assert failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test circuit breaker recovery from open to closed state."""

        async def failing_operation():
            raise ProviderError("Failure", ProviderErrorType.NETWORK_ERROR)

        async def successful_operation():
            return "recovered"

        # Force circuit to open
        for i in range(3):
            with pytest.raises(ProviderError):
                await circuit_breaker.execute(failing_operation)

        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)  # recovery_timeout = 1.0

        # Execute successful operations to close circuit
        result1 = await circuit_breaker.execute(successful_operation)
        assert result1 == "recovered"
        assert circuit_breaker.get_state() == CircuitBreakerState.HALF_OPEN

        result2 = await circuit_breaker.execute(successful_operation)
        assert result2 == "recovered"
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout_handling(self, circuit_breaker):
        """Test circuit breaker timeout handling."""

        async def slow_operation():
            await asyncio.sleep(10)  # Exceeds timeout_seconds = 5.0
            return "too_slow"

        # Execute operation that times out
        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.execute(slow_operation)

        # Verify timeout is recorded as failure
        metrics = circuit_breaker.get_metrics()
        assert metrics.failed_requests == 1

    def test_circuit_breaker_factory(self):
        """Test circuit breaker factory functionality."""
        factory = CircuitBreakerFactory()

        # Create circuit breaker for provider/model
        cb1 = factory.get_circuit_breaker_for_provider("provider1", "model1")
        cb2 = factory.get_circuit_breaker_for_provider("provider1", "model1")
        cb3 = factory.get_circuit_breaker_for_provider("provider2", "model1")

        # Verify same provider/model returns same instance
        assert cb1 is cb2
        assert cb1 is not cb3

        # Verify status summary
        status = factory.get_status_summary()
        assert status["total_circuit_breakers"] >= 2
        assert "states" in status
        assert "circuit_breakers" in status


class TestRetryServiceIntegration:
    """Test retry service reliability patterns."""

    @pytest.fixture
    def retry_config(self):
        """Create test retry configuration."""
        return RetryConfig(
            max_retries=3,
            base_delay=0.1,  # Short delay for tests
            max_delay=1.0,
            backoff_factor=2.0,
        )

    @pytest.fixture
    def retry_service(self, retry_config):
        """Create retry service instance."""
        return RetryService(retry_config)

    @pytest.fixture
    def retry_context(self):
        """Create retry context."""
        return RetryContext(
            operation_name="test_operation", provider_name="test_provider", model_id="test_model"
        )

    @pytest.mark.asyncio
    async def test_retry_service_success_first_attempt(self, retry_service, retry_context):
        """Test successful operation on first attempt."""
        attempt_count = 0

        async def successful_operation():
            nonlocal attempt_count
            attempt_count += 1
            return "success"

        result = await retry_service.execute_with_retry(successful_operation, retry_context)

        assert result == "success"
        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_retry_service_success_after_retries(self, retry_service, retry_context):
        """Test successful operation after retries."""
        attempt_count = 0

        async def eventually_successful_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RateLimitExceededError("Temporary failure", retry_after=0.1)
            return "success"

        result = await retry_service.execute_with_retry(
            eventually_successful_operation, retry_context
        )

        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_retry_service_max_retries_exceeded(self, retry_service, retry_context):
        """Test max retries exceeded scenario."""
        attempt_count = 0

        async def always_failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            raise NetworkError("Persistent failure")

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            await retry_service.execute_with_retry(always_failing_operation, retry_context)

        # Verify all retry attempts were made
        assert attempt_count == retry_service.default_config.max_retries + 1
        assert exc_info.value.retry_stats.total_attempts == attempt_count

    @pytest.mark.asyncio
    async def test_retry_service_non_retryable_error(self, retry_service, retry_context):
        """Test handling of non-retryable errors."""
        attempt_count = 0

        async def non_retryable_operation():
            nonlocal attempt_count
            attempt_count += 1
            raise AuthenticationError("Invalid credentials")

        # Should fail immediately without retries
        with pytest.raises(AuthenticationError):
            await retry_service.execute_with_retry(non_retryable_operation, retry_context)

        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_retry_service_adaptive_retry(self, retry_service, retry_context):
        """Test adaptive retry logic."""

        async def rate_limited_operation():
            raise RateLimitExceededError("Rate limited", retry_after=0.2)

        with pytest.raises(MaxRetriesExceededError):
            await retry_service.execute_with_adaptive_retry(rate_limited_operation, retry_context)

    @pytest.mark.asyncio
    async def test_retry_with_timeout(self, retry_service):
        """Test retry with total timeout."""
        context = RetryContext(
            operation_name="timeout_test",
            provider_name="test_provider",
            total_timeout=0.5,  # Short timeout
        )

        async def slow_failing_operation():
            await asyncio.sleep(0.2)  # Each attempt takes time
            raise NetworkError("Slow failure")

        with pytest.raises(MaxRetriesExceededError):
            await retry_service.execute_with_retry(slow_failing_operation, context)


class TestErrorHandlerIntegration:
    """Test error handler integration."""

    @pytest.fixture
    def error_handler(self):
        """Create error handler instance."""
        return ErrorHandler()

    def test_openai_error_handling(self, error_handler):
        """Test OpenAI-specific error handling."""

        # Simulate OpenAI error
        class MockOpenAIError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.__module__ = "openai.error"

        error = MockOpenAIError("Rate limit exceeded")

        provider_error = error_handler.handle_provider_error(error, "openai-provider", "gpt-4")

        assert isinstance(provider_error, ProviderError)
        assert provider_error.provider_name == "openai-provider"
        assert provider_error.model_id == "gpt-4"

    def test_anthropic_error_handling(self, error_handler):
        """Test Anthropic-specific error handling."""

        class MockAnthropicError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.__module__ = "anthropic.error"

        error = MockAnthropicError("Authentication error")

        provider_error = error_handler.handle_provider_error(
            error, "anthropic-provider", "claude-3"
        )

        assert isinstance(provider_error, ProviderError)
        assert provider_error.provider_name == "anthropic-provider"

    def test_http_error_handling(self, error_handler):
        """Test HTTP status code error handling."""

        class MockHTTPError(Exception):
            def __init__(self, status_code, message):
                super().__init__(message)
                self.status_code = status_code

        # Test rate limit error
        error = MockHTTPError(429, "Too Many Requests")

        provider_error = error_handler.handle_provider_error(error, "test-provider")

        assert isinstance(provider_error, RateLimitExceededError)

    def test_network_error_handling(self, error_handler):
        """Test network error handling."""

        class MockConnectionError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.__class__.__name__ = "ConnectionError"

        error = MockConnectionError("Connection failed")

        provider_error = error_handler.handle_provider_error(error, "test-provider")

        assert isinstance(provider_error, NetworkError)

    def test_error_retry_determination(self, error_handler):
        """Test determining if errors are retryable."""
        # Retryable errors
        rate_limit_error = RateLimitExceededError("Rate limited")
        network_error = NetworkError("Network failed")

        assert error_handler.is_retryable_error(rate_limit_error)
        assert error_handler.is_retryable_error(network_error)

        # Non-retryable errors
        auth_error = AuthenticationError("Invalid key")
        assert not error_handler.is_retryable_error(auth_error)

    def test_retry_delay_calculation(self, error_handler):
        """Test retry delay calculation."""
        rate_limit_error = RateLimitExceededError("Rate limited", retry_after=30)

        # Should use retry_after value
        delay = error_handler.get_retry_delay(rate_limit_error, attempt=1)
        assert delay == 30.0

        # Test exponential backoff for other errors
        network_error = NetworkError("Network error")
        delay1 = error_handler.get_retry_delay(network_error, attempt=1)
        delay2 = error_handler.get_retry_delay(network_error, attempt=2)

        assert delay2 > delay1  # Exponential backoff


class TestReliabilityPatternsIntegration:
    """Test integration of all reliability patterns together."""

    @pytest.fixture
    def integrated_service(self):
        """Create service with all reliability patterns integrated."""
        error_handler = ErrorHandler()
        circuit_breaker_factory = CircuitBreakerFactory()
        retry_service = RetryService()

        return {
            "error_handler": error_handler,
            "circuit_breaker_factory": circuit_breaker_factory,
            "retry_service": retry_service,
        }

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retry(self, integrated_service):
        """Test circuit breaker working with retry service."""
        error_handler = integrated_service["error_handler"]
        circuit_breaker = integrated_service["circuit_breaker_factory"].get_circuit_breaker(
            "integrated-test", CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.5)
        )
        retry_service = integrated_service["retry_service"]

        failure_count = 0

        async def unstable_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 5:  # Fail first 5 attempts
                raise NetworkError(f"Failure {failure_count}")
            return "success"

        retry_context = RetryContext(
            operation_name="integrated_test", provider_name="test_provider"
        )

        # This should eventually succeed after retries
        with pytest.raises(MaxRetriesExceededError):
            # First attempt through circuit breaker and retry
            await circuit_breaker.execute(
                lambda: retry_service.execute_with_retry(unstable_operation, retry_context)
            )

        # Circuit breaker should track the failures
        metrics = circuit_breaker.get_metrics()
        assert metrics.failed_requests > 0

    @pytest.mark.asyncio
    async def test_error_type_specific_handling(self, integrated_service):
        """Test that different error types are handled appropriately."""
        error_handler = integrated_service["error_handler"]
        retry_service = integrated_service["retry_service"]

        # Test rate limit error with specific retry behavior
        async def rate_limited_operation():
            raise RateLimitExceededError("Rate limited", retry_after=0.1)

        retry_context = RetryContext(
            operation_name="rate_limit_test", provider_name="test_provider"
        )

        with pytest.raises(MaxRetriesExceededError):
            await retry_service.execute_with_adaptive_retry(rate_limited_operation, retry_context)

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self, integrated_service):
        """Test prevention of cascading failures."""
        circuit_breaker = integrated_service["circuit_breaker_factory"].get_circuit_breaker(
            "cascade-test", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0)
        )

        async def failing_service():
            raise ProviderError("Service down", ProviderErrorType.PROVIDER_OVERLOADED)

        # Generate failures to open circuit
        for i in range(3):
            with pytest.raises(ProviderError):
                await circuit_breaker.execute(failing_service)

        # Circuit should now be open
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN

        # Further requests should fail fast
        start_time = datetime.now()
        with pytest.raises(CircuitBreakerOpenException):
            await circuit_breaker.execute(failing_service)
        end_time = datetime.now()

        # Should fail very quickly (no actual service call)
        assert (end_time - start_time).total_seconds() < 0.1

    def test_metrics_collection_integration(self, integrated_service):
        """Test that metrics are properly collected across all patterns."""
        circuit_breaker_factory = integrated_service["circuit_breaker_factory"]

        # Create multiple circuit breakers
        cb1 = circuit_breaker_factory.get_circuit_breaker("service1")
        cb2 = circuit_breaker_factory.get_circuit_breaker("service2")

        # Get status summary
        status = circuit_breaker_factory.get_status_summary()

        assert status["total_circuit_breakers"] >= 2
        assert "states" in status
        assert "circuit_breakers" in status
        assert len(status["circuit_breakers"]) >= 2
