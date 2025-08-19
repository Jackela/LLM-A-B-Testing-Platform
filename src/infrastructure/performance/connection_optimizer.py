"""Advanced connection optimization for databases and external services."""

import asyncio
import time
import weakref
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.pool import Pool

from .metrics_collector import MetricsCollector


@dataclass
class ConnectionMetrics:
    """Connection performance metrics."""

    total_connections_created: int = 0
    total_connections_closed: int = 0
    active_connections: int = 0
    peak_connections: int = 0
    connection_errors: int = 0
    average_connection_time_ms: float = 0.0
    connection_timeouts: int = 0
    connection_pool_hits: int = 0
    connection_pool_misses: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HTTPConnectionMetrics:
    """HTTP connection specific metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    average_response_time_ms: float = 0.0
    connections_reused: int = 0
    keep_alive_effectiveness: float = 0.0
    ssl_handshake_time_ms: float = 0.0


class ConnectionOptimizer:
    """Advanced connection optimization for various connection types."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector

        # Connection metrics tracking
        self._db_metrics = ConnectionMetrics()
        self._http_metrics = HTTPConnectionMetrics()

        # Connection monitoring
        self._connection_tracking: Dict[str, Dict] = defaultdict(dict)
        self._response_times = deque(maxlen=1000)
        self._connection_times = deque(maxlen=1000)

        # HTTP client optimizations
        self._optimized_http_clients: Dict[str, httpx.AsyncClient] = {}
        self._http_client_configs = {
            "default": {
                "timeout": httpx.Timeout(30.0, connect=10.0),
                "limits": httpx.Limits(max_keepalive_connections=20, max_connections=100),
                "http2": True,
                "verify": True,
            },
            "model_provider": {
                "timeout": httpx.Timeout(60.0, connect=15.0),
                "limits": httpx.Limits(max_keepalive_connections=50, max_connections=200),
                "http2": True,
                "verify": True,
            },
            "fast_api": {
                "timeout": httpx.Timeout(5.0, connect=2.0),
                "limits": httpx.Limits(max_keepalive_connections=10, max_connections=50),
                "http2": False,  # Disable HTTP/2 for fast APIs
                "verify": True,
            },
        }

        # Database connection optimization
        self._db_connection_observers: List[Callable] = []

        # Connection health monitoring
        self._health_monitoring_enabled = False
        self._health_check_interval = 60  # seconds
        self._health_check_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize connection optimizer."""
        # Initialize optimized HTTP clients
        for name, config in self._http_client_configs.items():
            self._optimized_http_clients[name] = httpx.AsyncClient(**config)

        print("Connection optimizer initialized")

    async def shutdown(self) -> None:
        """Shutdown connection optimizer."""
        # Stop health monitoring
        await self.stop_health_monitoring()

        # Close all HTTP clients
        for client in self._optimized_http_clients.values():
            await client.aclose()

        self._optimized_http_clients.clear()
        print("Connection optimizer shutdown complete")

    def get_http_client(self, client_type: str = "default") -> httpx.AsyncClient:
        """Get optimized HTTP client for specific use case."""
        return self._optimized_http_clients.get(
            client_type, self._optimized_http_clients["default"]
        )

    @asynccontextmanager
    async def monitored_http_request(
        self, client_type: str = "default", request_info: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Context manager for monitored HTTP requests."""
        client = self.get_http_client(client_type)
        start_time = time.time()

        try:
            yield client

            # Request succeeded
            duration = (time.time() - start_time) * 1000  # Convert to ms
            self._record_http_success(duration, request_info)

        except httpx.TimeoutException as e:
            duration = (time.time() - start_time) * 1000
            self._record_http_timeout(duration, request_info)
            raise

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_http_failure(duration, request_info, str(e))
            raise

    def _record_http_success(
        self, duration_ms: float, request_info: Optional[Dict[str, Any]]
    ) -> None:
        """Record successful HTTP request metrics."""
        self._http_metrics.total_requests += 1
        self._http_metrics.successful_requests += 1
        self._response_times.append(duration_ms)

        # Update average response time
        if self._response_times:
            self._http_metrics.average_response_time_ms = sum(self._response_times) / len(
                self._response_times
            )

        # Check for connection reuse (simplified heuristic)
        if duration_ms < 100:  # Fast responses likely indicate connection reuse
            self._http_metrics.connections_reused += 1

        # Update keep-alive effectiveness
        total_requests = self._http_metrics.total_requests
        if total_requests > 0:
            self._http_metrics.keep_alive_effectiveness = (
                self._http_metrics.connections_reused / total_requests
            )

        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_custom_timer("http_request_duration", duration_ms / 1000)

    def _record_http_timeout(
        self, duration_ms: float, request_info: Optional[Dict[str, Any]]
    ) -> None:
        """Record HTTP timeout metrics."""
        self._http_metrics.total_requests += 1
        self._http_metrics.timeout_requests += 1

        if self.metrics_collector:
            self.metrics_collector.increment_custom_counter(
                "http_request_timeouts", labels={"type": "timeout"}
            )

    def _record_http_failure(
        self, duration_ms: float, request_info: Optional[Dict[str, Any]], error: str
    ) -> None:
        """Record HTTP failure metrics."""
        self._http_metrics.total_requests += 1
        self._http_metrics.failed_requests += 1

        if self.metrics_collector:
            self.metrics_collector.increment_custom_counter(
                "http_request_failures", labels={"error_type": "connection_error"}
            )

    @asynccontextmanager
    async def monitored_db_connection(
        self, session_factory: Callable, connection_info: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[AsyncSession, None]:
        """Context manager for monitored database connections."""
        start_time = time.time()
        connection_id = f"db_conn_{int(time.time() * 1000)}"

        try:
            # Create session
            session = session_factory()
            connection_time = (time.time() - start_time) * 1000

            self._record_db_connection_success(connection_id, connection_time, connection_info)

            yield session

        except Exception as e:
            connection_time = (time.time() - start_time) * 1000
            self._record_db_connection_failure(
                connection_id, connection_time, connection_info, str(e)
            )
            raise

        finally:
            # Connection cleanup handled by session manager
            pass

    def _record_db_connection_success(
        self,
        connection_id: str,
        connection_time_ms: float,
        connection_info: Optional[Dict[str, Any]],
    ) -> None:
        """Record successful database connection metrics."""
        self._db_metrics.total_connections_created += 1
        self._db_metrics.active_connections += 1
        self._connection_times.append(connection_time_ms)

        # Update peak connections
        if self._db_metrics.active_connections > self._db_metrics.peak_connections:
            self._db_metrics.peak_connections = self._db_metrics.active_connections

        # Update average connection time
        if self._connection_times:
            self._db_metrics.average_connection_time_ms = sum(self._connection_times) / len(
                self._connection_times
            )

        # Track connection
        self._connection_tracking[connection_id] = {
            "created_at": time.time(),
            "connection_time_ms": connection_time_ms,
            "info": connection_info,
        }

        # Check if this was a pool hit (fast connection)
        if connection_time_ms < 10:  # Less than 10ms indicates pool hit
            self._db_metrics.connection_pool_hits += 1
        else:
            self._db_metrics.connection_pool_misses += 1

        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_custom_timer(
                "db_connection_time", connection_time_ms / 1000
            )
            self.metrics_collector.set_custom_gauge(
                "active_db_connections", self._db_metrics.active_connections
            )

    def _record_db_connection_failure(
        self,
        connection_id: str,
        connection_time_ms: float,
        connection_info: Optional[Dict[str, Any]],
        error: str,
    ) -> None:
        """Record database connection failure metrics."""
        self._db_metrics.connection_errors += 1

        if "timeout" in error.lower():
            self._db_metrics.connection_timeouts += 1

        if self.metrics_collector:
            self.metrics_collector.increment_custom_counter(
                "db_connection_failures", labels={"error_type": "connection_error"}
            )

    def record_db_connection_closed(self, connection_id: str) -> None:
        """Record database connection closure."""
        if connection_id in self._connection_tracking:
            del self._connection_tracking[connection_id]

        self._db_metrics.total_connections_closed += 1
        self._db_metrics.active_connections = max(0, self._db_metrics.active_connections - 1)

        if self.metrics_collector:
            self.metrics_collector.set_custom_gauge(
                "active_db_connections", self._db_metrics.active_connections
            )

    async def optimize_database_connections(
        self, engine: AsyncEngine, target_utilization: float = 0.8
    ) -> Dict[str, Any]:
        """Optimize database connection pool settings."""
        pool = engine.pool
        optimization_results = {}

        try:
            # Get current pool status
            if hasattr(pool, "size"):
                current_size = pool.size()
                checked_out = pool.checkedout()
                utilization = checked_out / current_size if current_size > 0 else 0

                optimization_results["current_pool_status"] = {
                    "pool_size": current_size,
                    "checked_out": checked_out,
                    "utilization": utilization,
                }

                # Suggest optimizations
                suggestions = []

                if utilization > target_utilization:
                    suggestions.append("Consider increasing pool_size")
                    suggested_size = int(current_size * 1.25)
                    suggestions.append(f"Suggested pool_size: {suggested_size}")

                elif utilization < 0.3:
                    suggestions.append("Consider decreasing pool_size")
                    suggested_size = max(5, int(current_size * 0.8))
                    suggestions.append(f"Suggested pool_size: {suggested_size}")

                # Check connection times
                if self._db_metrics.average_connection_time_ms > 100:
                    suggestions.append("High connection times detected")
                    suggestions.append("Consider increasing pool_pre_ping or pool_recycle settings")

                # Check error rates
                total_attempts = (
                    self._db_metrics.total_connections_created + self._db_metrics.connection_errors
                )
                if total_attempts > 0:
                    error_rate = self._db_metrics.connection_errors / total_attempts
                    if error_rate > 0.05:  # 5% error rate
                        suggestions.append(f"High connection error rate: {error_rate:.1%}")
                        suggestions.append("Check database connectivity and timeout settings")

                optimization_results["suggestions"] = suggestions

            else:
                optimization_results["error"] = "Pool does not support size introspection"

        except Exception as e:
            optimization_results["error"] = str(e)

        return optimization_results

    async def optimize_http_clients(self) -> Dict[str, Any]:
        """Optimize HTTP client configurations based on usage patterns."""
        optimization_results = {}

        for client_name, client in self._optimized_http_clients.items():
            client_results = {}

            # Analyze current configuration
            current_limits = client._limits
            current_timeout = client._timeout

            client_results["current_config"] = {
                "max_keepalive_connections": current_limits.max_keepalive_connections,
                "max_connections": current_limits.max_connections,
                "timeout": current_timeout.timeout,
            }

            # Generate suggestions based on metrics
            suggestions = []

            # Check keep-alive effectiveness
            if self._http_metrics.keep_alive_effectiveness < 0.5:
                suggestions.append("Low keep-alive effectiveness detected")
                suggestions.append("Consider increasing max_keepalive_connections")

            # Check timeout rates
            if self._http_metrics.timeout_requests > 0:
                timeout_rate = self._http_metrics.timeout_requests / max(
                    self._http_metrics.total_requests, 1
                )
                if timeout_rate > 0.05:  # 5% timeout rate
                    suggestions.append(f"High timeout rate: {timeout_rate:.1%}")
                    suggestions.append("Consider increasing timeout values")

            # Check response times
            if self._http_metrics.average_response_time_ms > 5000:
                suggestions.append("High average response times detected")
                suggestions.append("Consider optimizing connection pool or timeouts")

            client_results["suggestions"] = suggestions
            optimization_results[client_name] = client_results

        return optimization_results

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._health_monitoring_enabled:
            return

        self._health_monitoring_enabled = True
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
        print("Connection health monitoring started")

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._health_monitoring_enabled = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        print("Connection health monitoring stopped")

    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while self._health_monitoring_enabled:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in connection health monitoring: {e}")
                await asyncio.sleep(self._health_check_interval)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on connections."""
        # Update metrics timestamp
        self._db_metrics.last_updated = datetime.utcnow()

        # Check for stale connections
        current_time = time.time()
        stale_connections = []

        for conn_id, conn_info in self._connection_tracking.items():
            age = current_time - conn_info["created_at"]
            if age > 3600:  # Connections older than 1 hour
                stale_connections.append(conn_id)

        # Log stale connections
        if stale_connections:
            print(f"Found {len(stale_connections)} stale database connections")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics."""
        stats = {
            "database_connections": {
                "total_created": self._db_metrics.total_connections_created,
                "total_closed": self._db_metrics.total_connections_closed,
                "active_connections": self._db_metrics.active_connections,
                "peak_connections": self._db_metrics.peak_connections,
                "connection_errors": self._db_metrics.connection_errors,
                "average_connection_time_ms": self._db_metrics.average_connection_time_ms,
                "connection_timeouts": self._db_metrics.connection_timeouts,
                "pool_hit_rate": (
                    self._db_metrics.connection_pool_hits
                    / max(
                        self._db_metrics.connection_pool_hits
                        + self._db_metrics.connection_pool_misses,
                        1,
                    )
                ),
                "last_updated": self._db_metrics.last_updated.isoformat(),
            },
            "http_connections": {
                "total_requests": self._http_metrics.total_requests,
                "successful_requests": self._http_metrics.successful_requests,
                "failed_requests": self._http_metrics.failed_requests,
                "timeout_requests": self._http_metrics.timeout_requests,
                "success_rate": (
                    self._http_metrics.successful_requests
                    / max(self._http_metrics.total_requests, 1)
                ),
                "average_response_time_ms": self._http_metrics.average_response_time_ms,
                "connections_reused": self._http_metrics.connections_reused,
                "keep_alive_effectiveness": self._http_metrics.keep_alive_effectiveness,
            },
            "active_tracking": {
                "tracked_connections": len(self._connection_tracking),
                "http_clients": len(self._optimized_http_clients),
            },
        }

        return stats

    async def create_optimized_http_client(
        self,
        name: str,
        base_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_connections: int = 100,
        max_keepalive: int = 20,
        enable_http2: bool = True,
    ) -> httpx.AsyncClient:
        """Create a custom optimized HTTP client."""
        config = {
            "timeout": httpx.Timeout(timeout_seconds, connect=timeout_seconds / 3),
            "limits": httpx.Limits(
                max_keepalive_connections=max_keepalive, max_connections=max_connections
            ),
            "http2": enable_http2,
            "verify": True,
        }

        if base_url:
            config["base_url"] = base_url

        client = httpx.AsyncClient(**config)
        self._optimized_http_clients[name] = client

        return client


# Global connection optimizer instance
_connection_optimizer: Optional[ConnectionOptimizer] = None


def get_connection_optimizer() -> ConnectionOptimizer:
    """Get global connection optimizer instance."""
    global _connection_optimizer
    if _connection_optimizer is None:
        _connection_optimizer = ConnectionOptimizer()
    return _connection_optimizer


async def init_connection_optimizer(
    metrics_collector: Optional[MetricsCollector] = None,
) -> ConnectionOptimizer:
    """Initialize global connection optimizer."""
    global _connection_optimizer
    _connection_optimizer = ConnectionOptimizer(metrics_collector)
    await _connection_optimizer.initialize()
    return _connection_optimizer
