"""Circuit breaker implementation for model provider reliability."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional

from .error_handler import ProviderError, ProviderErrorType


class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back up


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Number of failures to open circuit
    recovery_timeout: float = 60.0  # Seconds to wait before half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout_seconds: float = 30.0  # Request timeout
    monitoring_window: float = 300.0  # Window for failure counting (5 minutes)


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by the circuit breaker."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0
    last_state_change: Optional[float] = None

    def get_failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        self._failure_times: list = []  # Track failure times for windowed counting

    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection."""
        async with self._lock:
            # Check if we can execute the request
            await self._check_state()

            # Record request attempt
            self.metrics.total_requests += 1

        start_time = time.time()

        try:
            # Execute the operation with timeout
            result = await asyncio.wait_for(
                operation(*args, **kwargs), timeout=self.config.timeout_seconds
            )

            # Record success
            await self._record_success()
            return result

        except asyncio.TimeoutError as e:
            await self._record_failure(
                ProviderError(
                    f"Operation timed out after {self.config.timeout_seconds}s",
                    ProviderErrorType.TIMEOUT_ERROR,
                )
            )
            raise e

        except Exception as e:
            # Determine if this is a provider error that should count as failure
            if isinstance(e, ProviderError):
                await self._record_failure(e)
            else:
                # Convert to provider error and record failure
                provider_error = ProviderError(
                    str(e), ProviderErrorType.UNKNOWN_ERROR, original_error=e
                )
                await self._record_failure(provider_error)
            raise e

    async def _check_state(self) -> None:
        """Check current state and potentially transition."""
        current_time = time.time()

        if self.state == CircuitBreakerState.OPEN:
            # Check if we should move to half-open
            if (
                self.metrics.last_failure_time
                and current_time - self.metrics.last_failure_time >= self.config.recovery_timeout
            ):
                await self._transition_to_half_open()
            else:
                # Still in open state, reject request
                retry_after = self.config.recovery_timeout - (
                    current_time - (self.metrics.last_failure_time or current_time)
                )
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is OPEN. Service unavailable.",
                    retry_after=max(0, retry_after),
                )

        elif self.state == CircuitBreakerState.HALF_OPEN:
            # In half-open state, allow limited requests to test service
            pass  # Request will be allowed

        # CLOSED state allows all requests

    async def _record_success(self) -> None:
        """Record a successful operation."""
        async with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()

            # If we're in half-open state, check if we should close
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to_closed()

    async def _record_failure(self, error: ProviderError) -> None:
        """Record a failed operation."""
        async with self._lock:
            current_time = time.time()

            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = current_time

            # Add to windowed failure tracking
            self._failure_times.append(current_time)
            self._clean_failure_window(current_time)

            # Check if we should open the circuit
            if self.state == CircuitBreakerState.CLOSED:
                if self._should_open_circuit():
                    await self._transition_to_open()

            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state should open the circuit
                await self._transition_to_open()

    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on failure metrics."""
        # Check consecutive failures
        if self.metrics.consecutive_failures >= self.config.failure_threshold:
            return True

        # Check failure rate in monitoring window
        current_time = time.time()
        recent_failures = [
            t for t in self._failure_times if current_time - t <= self.config.monitoring_window
        ]

        if len(recent_failures) >= self.config.failure_threshold:
            return True

        return False

    def _clean_failure_window(self, current_time: float) -> None:
        """Remove old failure times outside the monitoring window."""
        cutoff_time = current_time - self.config.monitoring_window
        self._failure_times = [t for t in self._failure_times if t > cutoff_time]

    async def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        if self.state != CircuitBreakerState.OPEN:
            self.state = CircuitBreakerState.OPEN
            self.metrics.state_changes += 1
            self.metrics.last_state_change = time.time()

            print(f"Circuit breaker '{self.name}' transitioned to OPEN state")

    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        if self.state != CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.HALF_OPEN
            self.metrics.consecutive_successes = 0  # Reset for testing
            self.metrics.state_changes += 1
            self.metrics.last_state_change = time.time()

            print(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN state")

    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        if self.state != CircuitBreakerState.CLOSED:
            self.state = CircuitBreakerState.CLOSED
            self.metrics.consecutive_failures = 0  # Reset failure count
            self.metrics.state_changes += 1
            self.metrics.last_state_change = time.time()
            self._failure_times.clear()  # Clear failure history

            print(f"Circuit breaker '{self.name}' transitioned to CLOSED state")

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        return self.metrics

    def is_available(self) -> bool:
        """Check if the circuit breaker allows requests."""
        return self.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]

    async def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        async with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.metrics = CircuitBreakerMetrics()
            self._failure_times.clear()

            print(f"Circuit breaker '{self.name}' has been reset")

    def to_dict(self) -> Dict[str, Any]:
        """Convert circuit breaker state to dictionary."""
        return {
            "name": self.name,
            "state": self.state.value,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "monitoring_window": self.config.monitoring_window,
            },
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "failure_rate": self.metrics.get_failure_rate(),
                "success_rate": self.metrics.get_success_rate(),
                "state_changes": self.metrics.state_changes,
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
                "last_state_change": self.metrics.last_state_change,
            },
            "is_available": self.is_available(),
        }


class CircuitBreakerFactory:
    """Factory for creating and managing circuit breakers."""

    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

    def get_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for the given name."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, config)

        return self._circuit_breakers[name]

    def get_circuit_breaker_for_provider(
        self, provider_id: str, model_id: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get circuit breaker for specific provider and model combination."""
        name = f"{provider_id}_{model_id}"
        return self.get_circuit_breaker(name, config)

    def get_all_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all managed circuit breakers."""
        return self._circuit_breakers.copy()

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for cb in self._circuit_breakers.values():
            await cb.reset()

    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary of all circuit breakers."""
        summary = {
            "total_circuit_breakers": len(self._circuit_breakers),
            "states": {"closed": 0, "open": 0, "half_open": 0},
            "circuit_breakers": {},
        }

        for name, cb in self._circuit_breakers.items():
            state = cb.get_state()
            summary["states"][state.value] += 1
            summary["circuit_breakers"][name] = cb.to_dict()

        return summary
