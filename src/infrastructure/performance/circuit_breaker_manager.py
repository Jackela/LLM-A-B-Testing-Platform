"""Enhanced circuit breaker management with advanced failure detection and recovery."""

import asyncio
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Type

from src.application.services.model_provider.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    CircuitBreakerState,
)

from .metrics_collector import MetricsCollector


class FailurePattern(Enum):
    """Types of failure patterns detected."""

    CONSECUTIVE_FAILURES = "consecutive_failures"
    TIMEOUT_SPIKE = "timeout_spike"
    ERROR_RATE_SPIKE = "error_rate_spike"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CASCADING_FAILURE = "cascading_failure"


@dataclass
class CircuitBreakerGroup:
    """Group of related circuit breakers."""

    name: str
    circuit_breakers: Dict[str, CircuitBreaker] = field(default_factory=dict)
    shared_config: Optional[CircuitBreakerConfig] = None
    failure_correlation_threshold: float = 0.7  # Threshold for detecting correlated failures
    auto_recovery_enabled: bool = True


@dataclass
class FailureAnalysis:
    """Analysis of failure patterns across circuit breakers."""

    pattern_type: FailurePattern
    affected_services: List[str]
    correlation_score: float
    severity: str  # low, medium, high, critical
    recommended_actions: List[str]
    detected_at: datetime
    root_cause_hints: List[str] = field(default_factory=list)


class CircuitBreakerManager:
    """Advanced circuit breaker management with intelligent failure detection."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._circuit_breaker_groups: Dict[str, CircuitBreakerGroup] = {}

        # Failure analysis
        self._failure_history: List[FailureAnalysis] = []
        self._correlation_window_minutes = 5
        self._max_failure_history = 1000

        # Health monitoring
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._monitoring_enabled = False

        # Auto-recovery settings
        self._auto_recovery_enabled = True
        self._recovery_backoff_multiplier = 2.0
        self._max_recovery_attempts = 3

        # Circuit breaker policies
        self._default_policies = {
            "model_provider": CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                success_threshold=2,
                timeout_seconds=10.0,
                monitoring_window=120.0,
            ),
            "database": CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                success_threshold=3,
                timeout_seconds=30.0,
                monitoring_window=300.0,
            ),
            "external_api": CircuitBreakerConfig(
                failure_threshold=4,
                recovery_timeout=45.0,
                success_threshold=2,
                timeout_seconds=20.0,
                monitoring_window=180.0,
            ),
            "cache": CircuitBreakerConfig(
                failure_threshold=10,
                recovery_timeout=15.0,
                success_threshold=3,
                timeout_seconds=5.0,
                monitoring_window=60.0,
            ),
        }

    def get_circuit_breaker(
        self,
        name: str,
        policy_name: str = "external_api",
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker with specified policy."""
        if name not in self._circuit_breakers:
            # Use provided config or default policy
            if config is None:
                config = self._default_policies.get(
                    policy_name, self._default_policies["external_api"]
                )

            self._circuit_breakers[name] = CircuitBreaker(name, config)

            # Record metrics if available
            if self.metrics_collector:
                self.metrics_collector.increment_custom_counter(
                    "circuit_breakers_created", labels={"name": name, "policy": policy_name}
                )

        return self._circuit_breakers[name]

    def create_circuit_breaker_group(
        self,
        group_name: str,
        service_names: List[str],
        policy_name: str = "external_api",
        shared_config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreakerGroup:
        """Create a group of related circuit breakers."""
        if shared_config is None:
            shared_config = self._default_policies.get(
                policy_name, self._default_policies["external_api"]
            )

        group = CircuitBreakerGroup(name=group_name, shared_config=shared_config)

        # Create circuit breakers for each service in the group
        for service_name in service_names:
            cb_name = f"{group_name}_{service_name}"
            group.circuit_breakers[service_name] = self.get_circuit_breaker(
                cb_name, policy_name, shared_config
            )

        self._circuit_breaker_groups[group_name] = group
        return group

    @asynccontextmanager
    async def protected_call(
        self,
        service_name: str,
        operation: Callable,
        policy_name: str = "external_api",
        *args,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Execute operation with circuit breaker protection."""
        circuit_breaker = self.get_circuit_breaker(service_name, policy_name)

        start_time = time.time()
        success = False
        error_details = None

        try:
            result = await circuit_breaker.execute(operation, *args, **kwargs)
            success = True
            yield result

        except CircuitBreakerOpenException as e:
            # Circuit breaker is open
            error_details = f"Circuit breaker open: {e}"
            self._record_circuit_breaker_event(service_name, "rejected", error_details)
            raise

        except Exception as e:
            # Operation failed
            error_details = str(e)
            self._record_circuit_breaker_event(service_name, "failed", error_details)
            raise

        finally:
            duration = time.time() - start_time

            # Record metrics
            if self.metrics_collector:
                status = "success" if success else "failure"
                self.metrics_collector.record_custom_timer(
                    f"circuit_breaker_{service_name}_duration", duration, labels={"status": status}
                )

            # Analyze for patterns if this was a failure
            if not success:
                await self._analyze_failure_patterns()

    def _record_circuit_breaker_event(
        self, service_name: str, event_type: str, details: Optional[str] = None
    ) -> None:
        """Record circuit breaker events for analysis."""
        if self.metrics_collector:
            self.metrics_collector.increment_custom_counter(
                f"circuit_breaker_events",
                labels={"service": service_name, "event_type": event_type},
            )

        # Log for debugging
        timestamp = datetime.utcnow().isoformat()
        print(f"[{timestamp}] Circuit Breaker Event: {service_name} - {event_type}")
        if details:
            print(f"  Details: {details}")

    async def _analyze_failure_patterns(self) -> None:
        """Analyze failure patterns across circuit breakers."""
        current_time = time.time()

        # Get recent failures (within correlation window)
        recent_failures = []
        correlation_window_seconds = self._correlation_window_minutes * 60

        for cb_name, cb in self._circuit_breakers.items():
            if (
                cb.metrics.last_failure_time
                and current_time - cb.metrics.last_failure_time < correlation_window_seconds
            ):
                recent_failures.append((cb_name, cb))

        # Analyze patterns
        if len(recent_failures) >= 2:
            await self._detect_cascading_failures(recent_failures)
            await self._detect_error_rate_spikes(recent_failures)
            await self._detect_timeout_spikes(recent_failures)

    async def _detect_cascading_failures(self, recent_failures: List[tuple]) -> None:
        """Detect cascading failure patterns."""
        if len(recent_failures) < 2:
            return

        # Check if failures are correlated in time
        failure_times = [
            cb.metrics.last_failure_time
            for _, cb in recent_failures
            if cb.metrics.last_failure_time
        ]

        if len(failure_times) < 2:
            return

        # Calculate time correlation
        time_spread = max(failure_times) - min(failure_times)
        if time_spread < 60:  # Failures within 1 minute
            service_names = [name for name, _ in recent_failures]

            analysis = FailureAnalysis(
                pattern_type=FailurePattern.CASCADING_FAILURE,
                affected_services=service_names,
                correlation_score=1.0 - (time_spread / 60),  # Closer in time = higher correlation
                severity="high" if len(service_names) > 3 else "medium",
                recommended_actions=[
                    "Investigate upstream service dependencies",
                    "Check for resource exhaustion",
                    "Review service mesh configuration",
                    "Consider implementing bulkhead pattern",
                ],
                detected_at=datetime.utcnow(),
                root_cause_hints=[
                    "Multiple services failing simultaneously",
                    f"Time spread: {time_spread:.1f} seconds",
                    "Possible shared dependency failure",
                ],
            )

            self._failure_history.append(analysis)
            await self._handle_failure_analysis(analysis)

    async def _detect_error_rate_spikes(self, recent_failures: List[tuple]) -> None:
        """Detect error rate spikes."""
        for service_name, cb in recent_failures:
            error_rate = cb.metrics.get_failure_rate()

            # Check for sudden error rate increase
            if error_rate > 0.5:  # More than 50% error rate
                analysis = FailureAnalysis(
                    pattern_type=FailurePattern.ERROR_RATE_SPIKE,
                    affected_services=[service_name],
                    correlation_score=error_rate,
                    severity="high" if error_rate > 0.8 else "medium",
                    recommended_actions=[
                        "Check service health and logs",
                        "Verify downstream dependencies",
                        "Consider service restart",
                        "Review recent deployments",
                    ],
                    detected_at=datetime.utcnow(),
                    root_cause_hints=[
                        f"Error rate: {error_rate:.1%}",
                        f"Total requests: {cb.metrics.total_requests}",
                        f"Failed requests: {cb.metrics.failed_requests}",
                    ],
                )

                self._failure_history.append(analysis)
                await self._handle_failure_analysis(analysis)

    async def _detect_timeout_spikes(self, recent_failures: List[tuple]) -> None:
        """Detect timeout spikes indicating performance degradation."""
        # This would analyze circuit breaker timeout patterns
        # Implementation depends on more detailed timeout tracking
        pass

    async def _handle_failure_analysis(self, analysis: FailureAnalysis) -> None:
        """Handle detected failure patterns."""
        print(f"FAILURE PATTERN DETECTED: {analysis.pattern_type.value}")
        print(f"  Affected services: {', '.join(analysis.affected_services)}")
        print(f"  Severity: {analysis.severity}")
        print(f"  Correlation score: {analysis.correlation_score:.2f}")

        # Auto-recovery actions
        if self._auto_recovery_enabled and analysis.severity in ["high", "critical"]:
            await self._attempt_auto_recovery(analysis)

        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.increment_custom_counter(
                "failure_patterns_detected",
                labels={"pattern_type": analysis.pattern_type.value, "severity": analysis.severity},
            )

    async def _attempt_auto_recovery(self, analysis: FailureAnalysis) -> None:
        """Attempt automatic recovery actions."""
        print(f"Attempting auto-recovery for pattern: {analysis.pattern_type.value}")

        for service_name in analysis.affected_services:
            if service_name in self._circuit_breakers:
                cb = self._circuit_breakers[service_name]

                # Reset circuit breaker to attempt recovery
                await cb.reset()
                print(f"  Reset circuit breaker for {service_name}")

                # Record recovery attempt
                if self.metrics_collector:
                    self.metrics_collector.increment_custom_counter(
                        "auto_recovery_attempts", labels={"service": service_name}
                    )

    async def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._monitoring_enabled:
            return

        self._monitoring_enabled = True
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
        print("Circuit breaker health monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._monitoring_enabled = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        print("Circuit breaker health monitoring stopped")

    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while self._monitoring_enabled:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self._health_check_interval)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all circuit breakers."""
        for name, cb in self._circuit_breakers.items():
            state = cb.get_state()
            metrics = cb.get_metrics()

            # Update metrics
            if self.metrics_collector:
                self.metrics_collector.set_custom_gauge(
                    f"circuit_breaker_state",
                    1 if state == CircuitBreakerState.CLOSED else 0,
                    labels={"service": name, "state": state.value},
                )

                self.metrics_collector.set_custom_gauge(
                    f"circuit_breaker_failure_rate",
                    metrics.get_failure_rate(),
                    labels={"service": name},
                )

    def get_circuit_breaker_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of circuit breakers."""
        if name:
            # Get specific circuit breaker status
            if name in self._circuit_breakers:
                return self._circuit_breakers[name].to_dict()
            else:
                return {"error": f"Circuit breaker '{name}' not found"}

        # Get all circuit breakers status
        status = {
            "total_circuit_breakers": len(self._circuit_breakers),
            "circuit_breakers": {},
            "groups": {},
            "recent_failures": [],
        }

        # Individual circuit breakers
        for name, cb in self._circuit_breakers.items():
            status["circuit_breakers"][name] = cb.to_dict()

        # Groups
        for group_name, group in self._circuit_breaker_groups.items():
            group_status = {
                "name": group.name,
                "circuit_breakers": list(group.circuit_breakers.keys()),
                "overall_health": "healthy",
            }

            # Check group health
            for cb in group.circuit_breakers.values():
                if cb.get_state() != CircuitBreakerState.CLOSED:
                    group_status["overall_health"] = "degraded"
                    break

            status["groups"][group_name] = group_status

        # Recent failure analyses
        status["recent_failures"] = [
            {
                "pattern_type": analysis.pattern_type.value,
                "affected_services": analysis.affected_services,
                "severity": analysis.severity,
                "detected_at": analysis.detected_at.isoformat(),
                "correlation_score": analysis.correlation_score,
            }
            for analysis in self._failure_history[-10:]  # Last 10 analyses
        ]

        return status

    async def reset_all_circuit_breakers(self) -> Dict[str, Any]:
        """Reset all circuit breakers."""
        results = {}

        for name, cb in self._circuit_breakers.items():
            try:
                await cb.reset()
                results[name] = "reset"
            except Exception as e:
                results[name] = f"error: {e}"

        return {"reset_results": results}

    async def configure_policy(self, policy_name: str, config: CircuitBreakerConfig) -> None:
        """Configure a circuit breaker policy."""
        self._default_policies[policy_name] = config
        print(f"Updated circuit breaker policy: {policy_name}")

    def get_failure_analysis_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get failure analysis summary for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_analyses = [
            analysis for analysis in self._failure_history if analysis.detected_at > cutoff_time
        ]

        # Group by pattern type
        pattern_counts = {}
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for analysis in recent_analyses:
            pattern_type = analysis.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            severity_counts[analysis.severity] += 1

        return {
            "time_period_hours": hours,
            "total_failure_patterns": len(recent_analyses),
            "pattern_breakdown": pattern_counts,
            "severity_breakdown": severity_counts,
            "most_affected_services": self._get_most_affected_services(recent_analyses),
            "recommendations": self._generate_overall_recommendations(recent_analyses),
        }

    def _get_most_affected_services(self, analyses: List[FailureAnalysis]) -> List[str]:
        """Get services most affected by failures."""
        service_counts = {}

        for analysis in analyses:
            for service in analysis.affected_services:
                service_counts[service] = service_counts.get(service, 0) + 1

        # Sort by count and return top 5
        sorted_services = sorted(service_counts.items(), key=lambda x: x[1], reverse=True)
        return [service for service, count in sorted_services[:5]]

    def _generate_overall_recommendations(self, analyses: List[FailureAnalysis]) -> List[str]:
        """Generate overall recommendations based on failure patterns."""
        recommendations = set()

        # Count pattern types
        pattern_counts = {}
        for analysis in analyses:
            pattern_type = analysis.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1

        # Generate recommendations based on most common patterns
        if pattern_counts.get("cascading_failure", 0) > 2:
            recommendations.add("Implement service mesh for better isolation")
            recommendations.add("Review service dependencies and implement bulkheads")

        if pattern_counts.get("error_rate_spike", 0) > 3:
            recommendations.add("Implement better error handling and retry logic")
            recommendations.add("Review service health checks and monitoring")

        if pattern_counts.get("timeout_spike", 0) > 2:
            recommendations.add("Optimize service performance and resource allocation")
            recommendations.add("Review timeout configurations across services")

        # Default recommendations
        if len(analyses) > 5:
            recommendations.add("Consider implementing chaos engineering practices")
            recommendations.add("Review and strengthen monitoring and alerting")

        return list(recommendations)
