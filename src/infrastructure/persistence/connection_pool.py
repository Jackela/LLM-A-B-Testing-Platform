"""Advanced connection pool management and monitoring."""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.pool import Pool

from .database import DatabaseConfig, DatabaseManager


@dataclass
class PoolMetrics:
    """Connection pool metrics."""

    pool_size: int
    checked_out: int
    checked_in: int
    overflow: int
    total_connections: int
    active_connections: int
    pool_hits: int
    pool_misses: int
    connection_errors: int
    average_checkout_time: float
    max_checkout_time: float
    last_updated: datetime


@dataclass
class ConnectionHealth:
    """Connection health status."""

    is_healthy: bool
    response_time_ms: float
    error_message: Optional[str] = None
    last_check: datetime = None


class ConnectionPoolManager:
    """Enhanced connection pool manager with monitoring and health checks."""

    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self._metrics = PoolMetrics(
            pool_size=0,
            checked_out=0,
            checked_in=0,
            overflow=0,
            total_connections=0,
            active_connections=0,
            pool_hits=0,
            pool_misses=0,
            connection_errors=0,
            average_checkout_time=0.0,
            max_checkout_time=0.0,
            last_updated=datetime.utcnow(),
        )
        self._checkout_times: List[float] = []
        self._max_checkout_times = 100  # Keep last 100 checkout times
        self._health_check_interval = 30  # seconds
        self._last_health_check = datetime.utcnow()
        self._connection_health = ConnectionHealth(
            is_healthy=True, response_time_ms=0.0, last_check=datetime.utcnow()
        )

    async def get_pool_metrics(self) -> PoolMetrics:
        """Get current connection pool metrics."""
        engine = self.database_manager.get_async_engine()
        pool = engine.pool

        if hasattr(pool, "size"):
            self._metrics.pool_size = pool.size()
            self._metrics.checked_out = pool.checkedout()
            self._metrics.checked_in = pool.checkedin()
            self._metrics.overflow = pool.overflow()
            self._metrics.total_connections = pool.size() + pool.overflow()

            # Calculate average checkout time
            if self._checkout_times:
                self._metrics.average_checkout_time = sum(self._checkout_times) / len(
                    self._checkout_times
                )
                self._metrics.max_checkout_time = max(self._checkout_times)

        self._metrics.last_updated = datetime.utcnow()
        return self._metrics

    @asynccontextmanager
    async def get_monitored_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with connection monitoring."""
        start_time = time.time()

        try:
            async with self.database_manager.get_session() as session:
                checkout_time = time.time() - start_time
                self._record_checkout_time(checkout_time)
                self._metrics.pool_hits += 1

                yield session

        except Exception as e:
            self._metrics.connection_errors += 1
            self._metrics.pool_misses += 1
            raise
        finally:
            total_time = time.time() - start_time
            self._record_checkout_time(total_time)

    def _record_checkout_time(self, checkout_time: float) -> None:
        """Record connection checkout time for metrics."""
        self._checkout_times.append(checkout_time)

        # Keep only recent checkout times
        if len(self._checkout_times) > self._max_checkout_times:
            self._checkout_times = self._checkout_times[-self._max_checkout_times :]

    async def health_check(self) -> ConnectionHealth:
        """Perform comprehensive connection health check."""
        now = datetime.utcnow()

        # Skip if recent check was performed
        if (now - self._last_health_check).seconds < self._health_check_interval:
            return self._connection_health

        start_time = time.time()

        try:
            async with self.get_monitored_session() as session:
                # Perform simple query to test connection
                await session.execute(text("SELECT 1"))

                # Test transaction capabilities
                async with session.begin():
                    await session.execute(text("SELECT current_timestamp"))

                response_time = (time.time() - start_time) * 1000  # Convert to ms

                self._connection_health = ConnectionHealth(
                    is_healthy=True, response_time_ms=response_time, last_check=now
                )

        except Exception as e:
            self._connection_health = ConnectionHealth(
                is_healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
                last_check=now,
            )

        self._last_health_check = now
        return self._connection_health

    async def get_connection_info(self) -> Dict[str, Any]:
        """Get comprehensive connection pool information."""
        metrics = await self.get_pool_metrics()
        health = await self.health_check()

        engine = self.database_manager.get_async_engine()

        return {
            "database_url": self.database_manager._mask_credentials(
                self.database_manager.config.database_url
            ),
            "pool_metrics": {
                "pool_size": metrics.pool_size,
                "checked_out": metrics.checked_out,
                "checked_in": metrics.checked_in,
                "overflow": metrics.overflow,
                "total_connections": metrics.total_connections,
                "pool_hits": metrics.pool_hits,
                "pool_misses": metrics.pool_misses,
                "connection_errors": metrics.connection_errors,
                "average_checkout_time_ms": metrics.average_checkout_time * 1000,
                "max_checkout_time_ms": metrics.max_checkout_time * 1000,
                "hit_rate": (
                    metrics.pool_hits / (metrics.pool_hits + metrics.pool_misses)
                    if (metrics.pool_hits + metrics.pool_misses) > 0
                    else 0
                ),
                "last_updated": metrics.last_updated.isoformat(),
            },
            "health_status": {
                "is_healthy": health.is_healthy,
                "response_time_ms": health.response_time_ms,
                "error_message": health.error_message,
                "last_check": health.last_check.isoformat(),
            },
            "engine_info": {
                "dialect": str(engine.dialect.name),
                "driver": str(engine.dialect.driver),
                "pool_class": type(engine.pool).__name__,
                "echo": engine.echo,
            },
        }

    async def optimize_pool_settings(self) -> Dict[str, Any]:
        """Analyze current usage and suggest pool optimizations."""
        metrics = await self.get_pool_metrics()

        suggestions = []
        current_config = self.database_manager.config

        # Analyze pool utilization
        if metrics.total_connections > 0:
            utilization = metrics.checked_out / metrics.total_connections

            if utilization > 0.8:
                suggestions.append(
                    {
                        "type": "pool_size",
                        "message": "High pool utilization detected. Consider increasing pool_size.",
                        "current_value": current_config.pool_size,
                        "suggested_value": current_config.pool_size + 5,
                    }
                )

            if utilization < 0.3 and current_config.pool_size > 10:
                suggestions.append(
                    {
                        "type": "pool_size",
                        "message": "Low pool utilization detected. Consider decreasing pool_size.",
                        "current_value": current_config.pool_size,
                        "suggested_value": max(10, current_config.pool_size - 5),
                    }
                )

        # Analyze checkout times
        if metrics.average_checkout_time > 5.0:  # 5 seconds
            suggestions.append(
                {
                    "type": "timeout",
                    "message": "High checkout times detected. Consider increasing pool_timeout.",
                    "current_value": current_config.pool_timeout,
                    "suggested_value": current_config.pool_timeout + 10,
                }
            )

        # Analyze overflow usage
        if metrics.overflow > metrics.pool_size * 0.5:
            suggestions.append(
                {
                    "type": "overflow",
                    "message": "High overflow usage detected. Consider increasing max_overflow.",
                    "current_value": current_config.max_overflow,
                    "suggested_value": current_config.max_overflow + 10,
                }
            )

        # Analyze error rate
        total_attempts = metrics.pool_hits + metrics.pool_misses
        if total_attempts > 0:
            error_rate = metrics.connection_errors / total_attempts
            if error_rate > 0.05:  # 5% error rate
                suggestions.append(
                    {
                        "type": "reliability",
                        "message": "High connection error rate detected. Check database connectivity.",
                        "error_rate": error_rate,
                        "total_errors": metrics.connection_errors,
                    }
                )

        return {
            "current_metrics": metrics.__dict__,
            "optimization_suggestions": suggestions,
            "recommended_actions": [
                "Monitor pool metrics regularly",
                "Adjust pool settings based on usage patterns",
                "Consider connection pooling at application level",
                "Implement connection retry logic for transient failures",
            ],
        }

    async def warm_up_pool(self, target_connections: Optional[int] = None) -> Dict[str, Any]:
        """Warm up connection pool by pre-creating connections."""
        if target_connections is None:
            target_connections = self.database_manager.config.pool_size

        start_time = time.time()
        successful_connections = 0
        failed_connections = 0

        # Create multiple concurrent connections to warm up the pool
        async def create_connection():
            try:
                async with self.get_monitored_session() as session:
                    await session.execute(text("SELECT 1"))
                return True
            except Exception:
                return False

        # Create connections concurrently
        tasks = [create_connection() for _ in range(target_connections)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if result is True:
                successful_connections += 1
            else:
                failed_connections += 1

        total_time = time.time() - start_time

        return {
            "target_connections": target_connections,
            "successful_connections": successful_connections,
            "failed_connections": failed_connections,
            "success_rate": (
                successful_connections / target_connections if target_connections > 0 else 0
            ),
            "warm_up_time_seconds": total_time,
            "average_time_per_connection": (
                total_time / target_connections if target_connections > 0 else 0
            ),
        }

    async def close_idle_connections(self) -> Dict[str, Any]:
        """Close idle connections and return statistics."""
        engine = self.database_manager.get_async_engine()

        # Get metrics before cleanup
        before_metrics = await self.get_pool_metrics()

        # Force garbage collection on pool
        if hasattr(engine.pool, "invalidate"):
            engine.pool.invalidate()

        # Give time for cleanup
        await asyncio.sleep(0.1)

        # Get metrics after cleanup
        after_metrics = await self.get_pool_metrics()

        return {
            "connections_before": before_metrics.total_connections,
            "connections_after": after_metrics.total_connections,
            "connections_closed": before_metrics.total_connections
            - after_metrics.total_connections,
            "cleanup_effective": before_metrics.total_connections > after_metrics.total_connections,
        }
