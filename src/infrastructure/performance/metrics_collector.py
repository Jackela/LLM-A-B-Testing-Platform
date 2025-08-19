"""Comprehensive performance metrics collection and monitoring."""

import asyncio
import statistics
import time
import weakref
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server


@dataclass
class PerformanceMetrics:
    """Container for various performance metrics."""

    # Response time metrics
    response_times: List[float] = field(default_factory=list)
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0

    # Throughput metrics
    requests_per_second: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0

    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_memory_usage: float = 0.0

    # Database metrics
    db_query_count: int = 0
    db_avg_query_time: float = 0.0
    db_slow_queries: int = 0

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_requests + self.failed_requests
        return self.successful_requests / total if total > 0 else 0.0


class MetricsCollector:
    """Advanced metrics collection with Prometheus integration."""

    def __init__(self, enable_prometheus: bool = True, prometheus_port: int = 8000):
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port

        # Time-series data storage
        self._response_times = deque(maxlen=10000)  # Keep last 10k response times
        self._throughput_samples = deque(maxlen=1000)  # Keep last 1k throughput samples
        self._memory_samples = deque(maxlen=1000)
        self._cpu_samples = deque(maxlen=1000)

        # Counters
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._db_query_count = 0
        self._db_slow_queries = 0

        # Timing data
        self._db_query_times = deque(maxlen=10000)
        self._last_throughput_calculation = time.time()
        self._request_count_at_last_calc = 0

        # Custom metrics storage
        self._custom_counters: Dict[str, int] = defaultdict(int)
        self._custom_gauges: Dict[str, float] = {}
        self._custom_timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Prometheus metrics
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
            self._start_prometheus_server()

    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        # Request metrics
        self.prom_request_count = Counter(
            "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
        )

        self.prom_request_duration = Histogram(
            "http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"]
        )

        # Database metrics
        self.prom_db_queries = Counter(
            "database_queries_total", "Total database queries", ["operation", "table"]
        )

        self.prom_db_query_duration = Histogram(
            "database_query_duration_seconds", "Database query duration", ["operation", "table"]
        )

        # Cache metrics
        self.prom_cache_operations = Counter(
            "cache_operations_total", "Total cache operations", ["operation", "layer", "result"]
        )

        # System metrics
        self.prom_memory_usage = Gauge("memory_usage_bytes", "Memory usage in bytes")

        self.prom_cpu_usage = Gauge("cpu_usage_percent", "CPU usage percentage")

        self.prom_active_connections = Gauge(
            "active_database_connections", "Active database connections"
        )

    def _start_prometheus_server(self) -> None:
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.prometheus_port)
            print(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            print(f"Failed to start Prometheus server: {e}")
            self.enable_prometheus = False

    @asynccontextmanager
    async def measure_request(
        self, method: str = "GET", endpoint: str = "/", labels: Optional[Dict[str, str]] = None
    ) -> AsyncGenerator[None, None]:
        """Context manager to measure request duration and count."""
        start_time = time.time()
        labels = labels or {}

        try:
            yield

            # Request succeeded
            duration = time.time() - start_time
            self._record_request_success(duration, method, endpoint, labels)

        except Exception as e:
            # Request failed
            duration = time.time() - start_time
            self._record_request_failure(duration, method, endpoint, labels, str(e))
            raise

    def _record_request_success(
        self, duration: float, method: str, endpoint: str, labels: Dict[str, str]
    ) -> None:
        """Record successful request metrics."""
        self._total_requests += 1
        self._successful_requests += 1
        self._response_times.append(duration)

        if self.enable_prometheus:
            self.prom_request_count.labels(method=method, endpoint=endpoint, status="success").inc()

            self.prom_request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def _record_request_failure(
        self, duration: float, method: str, endpoint: str, labels: Dict[str, str], error: str
    ) -> None:
        """Record failed request metrics."""
        self._total_requests += 1
        self._failed_requests += 1
        self._response_times.append(duration)

        if self.enable_prometheus:
            self.prom_request_count.labels(method=method, endpoint=endpoint, status="error").inc()

    @asynccontextmanager
    async def measure_database_query(
        self, operation: str = "SELECT", table: str = "unknown", query_id: Optional[str] = None
    ) -> AsyncGenerator[None, None]:
        """Context manager to measure database query duration."""
        start_time = time.time()

        try:
            yield

            # Query succeeded
            duration = time.time() - start_time
            self._record_db_query_success(duration, operation, table, query_id)

        except Exception as e:
            # Query failed
            duration = time.time() - start_time
            self._record_db_query_failure(duration, operation, table, query_id, str(e))
            raise

    def _record_db_query_success(
        self, duration: float, operation: str, table: str, query_id: Optional[str]
    ) -> None:
        """Record successful database query metrics."""
        self._db_query_count += 1
        self._db_query_times.append(duration)

        # Check for slow query (>1 second)
        if duration > 1.0:
            self._db_slow_queries += 1
            print(f"SLOW QUERY DETECTED: {operation} on {table} took {duration:.3f}s")

        if self.enable_prometheus:
            self.prom_db_queries.labels(operation=operation, table=table).inc()

            self.prom_db_query_duration.labels(operation=operation, table=table).observe(duration)

    def _record_db_query_failure(
        self, duration: float, operation: str, table: str, query_id: Optional[str], error: str
    ) -> None:
        """Record failed database query metrics."""
        self._db_query_count += 1
        print(f"DATABASE QUERY FAILED: {operation} on {table} - {error}")

    def record_cache_operation(
        self,
        operation: str,  # get, set, delete, clear
        layer: str,  # memory, redis, hybrid
        result: str,  # hit, miss, success, error
        duration: float = 0.0,
    ) -> None:
        """Record cache operation metrics."""
        if self.enable_prometheus:
            self.prom_cache_operations.labels(operation=operation, layer=layer, result=result).inc()

    def record_memory_usage(self, usage_bytes: float) -> None:
        """Record memory usage metrics."""
        usage_mb = usage_bytes / (1024 * 1024)
        self._memory_samples.append(usage_mb)

        if self.enable_prometheus:
            self.prom_memory_usage.set(usage_bytes)

    def record_cpu_usage(self, usage_percent: float) -> None:
        """Record CPU usage metrics."""
        self._cpu_samples.append(usage_percent)

        if self.enable_prometheus:
            self.prom_cpu_usage.set(usage_percent)

    def record_active_connections(self, count: int) -> None:
        """Record active database connections."""
        if self.enable_prometheus:
            self.prom_active_connections.set(count)

    def increment_custom_counter(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a custom counter metric."""
        self._custom_counters[name] += value

    def set_custom_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a custom gauge metric."""
        self._custom_gauges[name] = value

    def record_custom_timer(
        self, name: str, duration: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a custom timer metric."""
        self._custom_timers[name].append(duration)

    @asynccontextmanager
    async def custom_timer(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> AsyncGenerator[None, None]:
        """Context manager for custom timing measurements."""
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_custom_timer(name, duration, labels)

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot."""
        metrics = PerformanceMetrics()

        # Response time metrics
        if self._response_times:
            metrics.response_times = list(self._response_times)
            metrics.avg_response_time = statistics.mean(self._response_times)

            sorted_times = sorted(self._response_times)
            if len(sorted_times) >= 20:  # Need reasonable sample size
                p95_index = int(len(sorted_times) * 0.95)
                p99_index = int(len(sorted_times) * 0.99)
                metrics.p95_response_time = sorted_times[p95_index]
                metrics.p99_response_time = sorted_times[p99_index]

        # Request count metrics
        metrics.total_requests = self._total_requests
        metrics.successful_requests = self._successful_requests
        metrics.failed_requests = self._failed_requests

        # Calculate throughput
        current_time = time.time()
        time_diff = current_time - self._last_throughput_calculation
        if time_diff > 0:
            request_diff = self._total_requests - self._request_count_at_last_calc
            metrics.requests_per_second = request_diff / time_diff

        # Resource metrics
        if self._memory_samples:
            metrics.memory_usage_mb = self._memory_samples[-1]

        if self._cpu_samples:
            metrics.cpu_usage_percent = self._cpu_samples[-1]

        # Database metrics
        metrics.db_query_count = self._db_query_count
        metrics.db_slow_queries = self._db_slow_queries

        if self._db_query_times:
            metrics.db_avg_query_time = statistics.mean(self._db_query_times)

        # Custom metrics
        metrics.custom_metrics = {
            "counters": dict(self._custom_counters),
            "gauges": dict(self._custom_gauges),
            "timers": {
                name: {
                    "count": len(times),
                    "avg": statistics.mean(times) if times else 0,
                    "min": min(times) if times else 0,
                    "max": max(times) if times else 0,
                }
                for name, times in self._custom_timers.items()
            },
        }

        return metrics

    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get metrics summary for the specified time window."""
        metrics = self.get_current_metrics()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "window_minutes": window_minutes,
            "performance": {
                "avg_response_time_ms": metrics.avg_response_time * 1000,
                "p95_response_time_ms": metrics.p95_response_time * 1000,
                "p99_response_time_ms": metrics.p99_response_time * 1000,
                "requests_per_second": metrics.requests_per_second,
                "success_rate": metrics.get_success_rate(),
                "total_requests": metrics.total_requests,
            },
            "resources": {
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "active_connections": metrics.active_connections,
            },
            "database": {
                "query_count": metrics.db_query_count,
                "avg_query_time_ms": metrics.db_avg_query_time * 1000,
                "slow_queries": metrics.db_slow_queries,
                "slow_query_rate": metrics.db_slow_queries / max(metrics.db_query_count, 1),
            },
            "cache": {
                "hit_rate": metrics.cache_hit_rate,
                "memory_usage": metrics.cache_memory_usage,
            },
            "custom_metrics": metrics.custom_metrics,
        }

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self._response_times.clear()
        self._throughput_samples.clear()
        self._memory_samples.clear()
        self._cpu_samples.clear()
        self._db_query_times.clear()

        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._db_query_count = 0
        self._db_slow_queries = 0

        self._custom_counters.clear()
        self._custom_gauges.clear()
        self._custom_timers.clear()

        self._last_throughput_calculation = time.time()
        self._request_count_at_last_calc = 0

    async def start_background_collection(self, interval_seconds: int = 30) -> None:
        """Start background metrics collection."""
        import psutil

        async def collect_system_metrics():
            while True:
                try:
                    # Collect system metrics
                    process = psutil.Process()

                    # Memory usage
                    memory_info = process.memory_info()
                    self.record_memory_usage(memory_info.rss)

                    # CPU usage
                    cpu_percent = process.cpu_percent()
                    self.record_cpu_usage(cpu_percent)

                    # Update throughput calculation
                    self._last_throughput_calculation = time.time()
                    self._request_count_at_last_calc = self._total_requests

                except Exception as e:
                    print(f"Error collecting system metrics: {e}")

                await asyncio.sleep(interval_seconds)

        # Start background task
        asyncio.create_task(collect_system_metrics())


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def init_metrics_collector(
    enable_prometheus: bool = True, prometheus_port: int = 8000
) -> MetricsCollector:
    """Initialize global metrics collector."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(enable_prometheus, prometheus_port)
    return _metrics_collector


# Decorator for measuring function execution time
def measure_execution_time(metric_name: str):
    """Decorator to measure function execution time."""

    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics_collector()
            async with metrics.custom_timer(f"{metric_name}_{func.__name__}"):
                return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            metrics = get_metrics_collector()
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics.record_custom_timer(f"{metric_name}_{func.__name__}", duration)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
