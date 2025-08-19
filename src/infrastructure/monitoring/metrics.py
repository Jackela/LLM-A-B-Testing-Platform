"""Prometheus metrics collection and monitoring."""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
)

from .structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricConfig:
    """Configuration for a metric."""

    name: str
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


class MetricsCollector:
    """Centralized metrics collection system."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._summaries: Dict[str, Summary] = {}
        self._info: Dict[str, Info] = {}

        # System metrics
        self._setup_system_metrics()

        # Application metrics
        self._setup_application_metrics()

        # Business metrics
        self._setup_business_metrics()

        # Start background metric collection
        self._start_background_collection()

    def _setup_system_metrics(self):
        """Setup system-level metrics."""
        # System info
        self._info["system"] = Info("system_info", "System information", registry=self.registry)

        # CPU metrics
        self._gauges["cpu_usage"] = Gauge(
            "system_cpu_usage_percent", "CPU usage percentage", ["cpu"], registry=self.registry
        )

        # Memory metrics
        self._gauges["memory_usage"] = Gauge(
            "system_memory_usage_bytes", "Memory usage in bytes", ["type"], registry=self.registry
        )

        self._gauges["memory_percent"] = Gauge(
            "system_memory_usage_percent", "Memory usage percentage", registry=self.registry
        )

        # Disk metrics
        self._gauges["disk_usage"] = Gauge(
            "system_disk_usage_bytes",
            "Disk usage in bytes",
            ["device", "type"],
            registry=self.registry,
        )

        self._gauges["disk_percent"] = Gauge(
            "system_disk_usage_percent", "Disk usage percentage", ["device"], registry=self.registry
        )

        # Network metrics
        self._counters["network_bytes"] = Counter(
            "system_network_bytes_total",
            "Network bytes transferred",
            ["interface", "direction"],
            registry=self.registry,
        )

        # Process metrics
        self._gauges["process_threads"] = Gauge(
            "process_threads_total", "Number of process threads", registry=self.registry
        )

        self._gauges["process_fds"] = Gauge(
            "process_open_fds_total", "Number of open file descriptors", registry=self.registry
        )

    def _setup_application_metrics(self):
        """Setup application-level metrics."""
        # HTTP request metrics
        self._counters["http_requests"] = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry,
        )

        self._histograms["http_request_duration"] = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        # Database metrics
        self._counters["db_queries"] = Counter(
            "database_queries_total",
            "Total database queries",
            ["operation", "table"],
            registry=self.registry,
        )

        self._histograms["db_query_duration"] = Histogram(
            "database_query_duration_seconds",
            "Database query duration in seconds",
            ["operation", "table"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry,
        )

        self._gauges["db_connections"] = Gauge(
            "database_connections_active",
            "Active database connections",
            ["pool"],
            registry=self.registry,
        )

        # Cache metrics
        self._counters["cache_operations"] = Counter(
            "cache_operations_total",
            "Total cache operations",
            ["operation", "result"],
            registry=self.registry,
        )

        self._histograms["cache_operation_duration"] = Histogram(
            "cache_operation_duration_seconds",
            "Cache operation duration in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            registry=self.registry,
        )

        # Authentication metrics
        self._counters["auth_attempts"] = Counter(
            "authentication_attempts_total",
            "Total authentication attempts",
            ["result", "method"],
            registry=self.registry,
        )

        self._histograms["auth_duration"] = Histogram(
            "authentication_duration_seconds",
            "Authentication duration in seconds",
            ["method"],
            registry=self.registry,
        )

        # Error metrics
        self._counters["errors"] = Counter(
            "application_errors_total",
            "Total application errors",
            ["component", "error_type"],
            registry=self.registry,
        )

        # Security metrics
        self._counters["security_events"] = Counter(
            "security_events_total",
            "Total security events",
            ["event_type", "severity"],
            registry=self.registry,
        )

        self._counters["rate_limit_hits"] = Counter(
            "rate_limit_hits_total",
            "Total rate limit hits",
            ["client_type", "endpoint"],
            registry=self.registry,
        )

    def _setup_business_metrics(self):
        """Setup business-level metrics."""
        # Test management metrics
        self._counters["tests_created"] = Counter(
            "tests_created_total", "Total tests created", ["user_role"], registry=self.registry
        )

        self._gauges["tests_active"] = Gauge(
            "tests_active_total", "Currently active tests", registry=self.registry
        )

        self._histograms["test_duration"] = Histogram(
            "test_duration_hours",
            "Test duration in hours",
            buckets=[1, 6, 12, 24, 48, 72, 168],  # 1h to 1 week
            registry=self.registry,
        )

        # Model provider metrics
        self._counters["model_requests"] = Counter(
            "model_requests_total",
            "Total model requests",
            ["provider", "model", "status"],
            registry=self.registry,
        )

        self._histograms["model_response_time"] = Histogram(
            "model_response_time_seconds",
            "Model response time in seconds",
            ["provider", "model"],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        self._counters["model_tokens"] = Counter(
            "model_tokens_total",
            "Total tokens processed",
            ["provider", "model", "token_type"],  # token_type: input/output
            registry=self.registry,
        )

        # User activity metrics
        self._counters["user_actions"] = Counter(
            "user_actions_total",
            "Total user actions",
            ["action", "user_role"],
            registry=self.registry,
        )

        self._gauges["active_users"] = Gauge(
            "active_users_total", "Currently active users", registry=self.registry
        )

        # Analytics metrics
        self._counters["analytics_queries"] = Counter(
            "analytics_queries_total",
            "Total analytics queries",
            ["query_type"],
            registry=self.registry,
        )

        self._histograms["analytics_computation_time"] = Histogram(
            "analytics_computation_time_seconds",
            "Analytics computation time in seconds",
            ["computation_type"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

    def _start_background_collection(self):
        """Start background metric collection thread."""

        def collect_system_metrics():
            while True:
                try:
                    self._collect_system_metrics()
                    time.sleep(10)  # Collect every 10 seconds
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(30)  # Wait longer on error

        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()

    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self._gauges["cpu_usage"].labels(cpu="total").set(cpu_percent)

            # Per-core CPU usage
            cpu_percents = psutil.cpu_percent(interval=None, percpu=True)
            for i, cpu_pct in enumerate(cpu_percents):
                self._gauges["cpu_usage"].labels(cpu=f"core_{i}").set(cpu_pct)

            # Memory metrics
            memory = psutil.virtual_memory()
            self._gauges["memory_usage"].labels(type="used").set(memory.used)
            self._gauges["memory_usage"].labels(type="available").set(memory.available)
            self._gauges["memory_usage"].labels(type="total").set(memory.total)
            self._gauges["memory_percent"].set(memory.percent)

            # Disk metrics
            for partition in psutil.disk_partitions():
                if partition.mountpoint:
                    try:
                        disk_usage = psutil.disk_usage(partition.mountpoint)
                        device = partition.device.replace(":", "").replace("\\", "_")

                        self._gauges["disk_usage"].labels(device=device, type="used").set(
                            disk_usage.used
                        )
                        self._gauges["disk_usage"].labels(device=device, type="free").set(
                            disk_usage.free
                        )
                        self._gauges["disk_usage"].labels(device=device, type="total").set(
                            disk_usage.total
                        )
                        self._gauges["disk_percent"].labels(device=device).set(disk_usage.percent)
                    except (PermissionError, OSError):
                        continue

            # Process metrics
            process = psutil.Process()
            self._gauges["process_threads"].set(process.num_threads())

            try:
                self._gauges["process_fds"].set(process.num_fds())
            except (AttributeError, OSError):
                # num_fds not available on Windows
                pass

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    # HTTP metrics methods
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self._counters["http_requests"].labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()

        self._histograms["http_request_duration"].labels(method=method, endpoint=endpoint).observe(
            duration
        )

    # Database metrics methods
    def record_db_query(self, operation: str, table: str, duration: float):
        """Record database query metrics."""
        self._counters["db_queries"].labels(operation=operation, table=table).inc()
        self._histograms["db_query_duration"].labels(operation=operation, table=table).observe(
            duration
        )

    def set_db_connections(self, pool_name: str, count: int):
        """Set database connection count."""
        self._gauges["db_connections"].labels(pool=pool_name).set(count)

    # Cache metrics methods
    def record_cache_operation(self, operation: str, result: str, duration: float):
        """Record cache operation metrics."""
        self._counters["cache_operations"].labels(operation=operation, result=result).inc()
        self._histograms["cache_operation_duration"].labels(operation=operation).observe(duration)

    # Authentication metrics methods
    def record_auth_attempt(self, result: str, method: str, duration: float):
        """Record authentication attempt."""
        self._counters["auth_attempts"].labels(result=result, method=method).inc()
        self._histograms["auth_duration"].labels(method=method).observe(duration)

    # Error metrics methods
    def record_error(self, component: str, error_type: str):
        """Record application error."""
        self._counters["errors"].labels(component=component, error_type=error_type).inc()

    # Security metrics methods
    def record_security_event(self, event_type: str, severity: str):
        """Record security event."""
        self._counters["security_events"].labels(event_type=event_type, severity=severity).inc()

    def record_rate_limit_hit(self, client_type: str, endpoint: str):
        """Record rate limit hit."""
        self._counters["rate_limit_hits"].labels(client_type=client_type, endpoint=endpoint).inc()

    # Business metrics methods
    def record_test_created(self, user_role: str):
        """Record test creation."""
        self._counters["tests_created"].labels(user_role=user_role).inc()

    def set_active_tests(self, count: int):
        """Set active tests count."""
        self._gauges["tests_active"].set(count)

    def record_test_completion(self, duration_hours: float):
        """Record test completion."""
        self._histograms["test_duration"].observe(duration_hours)

    def record_model_request(
        self,
        provider: str,
        model: str,
        status: str,
        response_time: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Record model request metrics."""
        self._counters["model_requests"].labels(provider=provider, model=model, status=status).inc()
        self._histograms["model_response_time"].labels(provider=provider, model=model).observe(
            response_time
        )

        if input_tokens:
            self._counters["model_tokens"].labels(provider=provider, model=model, type="input").inc(
                input_tokens
            )
        if output_tokens:
            self._counters["model_tokens"].labels(
                provider=provider, model=model, type="output"
            ).inc(output_tokens)

    def record_user_action(self, action: str, user_role: str):
        """Record user action."""
        self._counters["user_actions"].labels(action=action, user_role=user_role).inc()

    def set_active_users(self, count: int):
        """Set active users count."""
        self._gauges["active_users"].set(count)

    def record_analytics_query(self, query_type: str, computation_time: float):
        """Record analytics query."""
        self._counters["analytics_queries"].labels(query_type=query_type).inc()
        self._histograms["analytics_computation_time"].labels(computation_type=query_type).observe(
            computation_time
        )

    # Utility methods
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode("utf-8")

    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


class PerformanceTracker:
    """Track performance metrics with context."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._active_operations: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start_operation(self, operation_id: str, operation_type: str, **context):
        """Start tracking an operation."""
        with self._lock:
            self._active_operations[operation_id] = {
                "type": operation_type,
                "start_time": time.time(),
                "context": context,
            }

    def end_operation(self, operation_id: str, success: bool = True, **additional_context):
        """End tracking an operation."""
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Attempted to end unknown operation: {operation_id}")
                return

            operation = self._active_operations.pop(operation_id)
            duration = time.time() - operation["start_time"]

            # Record metrics based on operation type
            op_type = operation["type"]
            context = {**operation["context"], **additional_context}

            if op_type == "http_request":
                self.metrics.record_http_request(
                    context.get("method", "unknown"),
                    context.get("endpoint", "unknown"),
                    context.get("status_code", 500 if not success else 200),
                    duration,
                )
            elif op_type == "db_query":
                self.metrics.record_db_query(
                    context.get("operation", "unknown"), context.get("table", "unknown"), duration
                )
            elif op_type == "model_request":
                self.metrics.record_model_request(
                    context.get("provider", "unknown"),
                    context.get("model", "unknown"),
                    "success" if success else "error",
                    duration,
                    context.get("input_tokens", 0),
                    context.get("output_tokens", 0),
                )

            # Log performance
            logger.log_performance(
                operation=f"{op_type}:{operation_id}", duration=duration, success=success, **context
            )

    def get_active_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active operations."""
        with self._lock:
            return self._active_operations.copy()


# Context manager for tracking operations
class track_operation:
    """Context manager for tracking operation performance."""

    def __init__(
        self, tracker: PerformanceTracker, operation_type: str, operation_id: str = None, **context
    ):
        self.tracker = tracker
        self.operation_type = operation_type
        self.operation_id = operation_id or f"{operation_type}_{int(time.time()*1000)}"
        self.context = context
        self.success = True

    def __enter__(self):
        self.tracker.start_operation(self.operation_id, self.operation_type, **self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
        self.tracker.end_operation(self.operation_id, self.success)

    def set_success(self, success: bool):
        """Set operation success status."""
        self.success = success

    def add_context(self, **context):
        """Add additional context to the operation."""
        self.context.update(context)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None
_performance_tracker: Optional[PerformanceTracker] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker(get_metrics_collector())
    return _performance_tracker
