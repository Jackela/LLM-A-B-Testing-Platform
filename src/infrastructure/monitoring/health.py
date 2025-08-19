"""Health monitoring and status checks."""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import httpx
import psutil
import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .metrics import get_metrics_collector
from .structured_logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    message: str
    timestamp: datetime
    checks: List[HealthCheck]
    uptime_seconds: float
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["checks"] = [check.to_dict() for check in self.checks]
        return result


class HealthChecker:
    """Comprehensive health monitoring system."""

    def __init__(
        self,
        database_session_factory: Optional[Callable] = None,
        redis_url: str = "redis://localhost:6379",
        external_services: Dict[str, str] = None,
    ):
        self.database_session_factory = database_session_factory
        self.redis_url = redis_url
        self.external_services = external_services or {}
        self.metrics = get_metrics_collector()
        self.start_time = time.time()

        # Thresholds
        self.cpu_threshold = 90.0  # CPU usage percentage
        self.memory_threshold = 90.0  # Memory usage percentage
        self.disk_threshold = 95.0  # Disk usage percentage
        self.response_time_threshold = 5.0  # Response time in seconds

        # Health check registry
        self._health_checks: Dict[str, Callable] = {}
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks."""
        self._health_checks.update(
            {
                "system": self._check_system_resources,
                "database": self._check_database,
                "redis": self._check_redis,
                "external_services": self._check_external_services,
                "application": self._check_application_health,
            }
        )

    def register_check(self, name: str, check_function: Callable):
        """Register custom health check."""
        self._health_checks[name] = check_function

    async def check_health(self, include_checks: List[str] = None) -> SystemHealth:
        """Perform comprehensive health check."""
        start_time = time.time()

        # Determine which checks to run
        checks_to_run = include_checks or list(self._health_checks.keys())

        # Run health checks
        check_results = []
        overall_status = HealthStatus.HEALTHY

        for check_name in checks_to_run:
            if check_name in self._health_checks:
                try:
                    check_result = await self._run_health_check(
                        check_name, self._health_checks[check_name]
                    )
                    check_results.append(check_result)

                    # Update overall status
                    if check_result.status == HealthStatus.UNHEALTHY:
                        overall_status = HealthStatus.UNHEALTHY
                    elif (
                        check_result.status == HealthStatus.DEGRADED
                        and overall_status != HealthStatus.UNHEALTHY
                    ):
                        overall_status = HealthStatus.DEGRADED

                except Exception as e:
                    logger.error(f"Health check '{check_name}' failed: {e}")
                    check_results.append(
                        HealthCheck(
                            name=check_name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Check failed: {str(e)}",
                            duration_ms=(time.time() - start_time) * 1000,
                            timestamp=datetime.utcnow(),
                        )
                    )
                    overall_status = HealthStatus.UNHEALTHY

        # Calculate uptime
        uptime_seconds = time.time() - self.start_time

        # Create system health summary
        system_health = SystemHealth(
            status=overall_status,
            message=self._get_status_message(overall_status, check_results),
            timestamp=datetime.utcnow(),
            checks=check_results,
            uptime_seconds=uptime_seconds,
        )

        # Record metrics
        self.metrics.record_health_check(
            status=overall_status.value,
            duration=(time.time() - start_time) * 1000,
            checks_count=len(check_results),
        )

        logger.info(
            f"Health check completed: {overall_status.value}",
            event_type="system",
            health_status=overall_status.value,
            checks_count=len(check_results),
            duration_ms=(time.time() - start_time) * 1000,
        )

        return system_health

    async def _run_health_check(self, name: str, check_function: Callable) -> HealthCheck:
        """Run individual health check with timing."""
        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                result = check_function()

            duration_ms = (time.time() - start_time) * 1000

            if isinstance(result, HealthCheck):
                result.duration_ms = duration_ms
                return result
            else:
                # Assume healthy if no specific result
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="OK",
                    duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    details=result if isinstance(result, dict) else None,
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Health check '{name}' failed: {e}")

            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
            )

    def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk_usage = psutil.disk_usage("/")
            disk_percent = (disk_usage.used / disk_usage.total) * 100

            # Determine status
            status = HealthStatus.HEALTHY
            messages = []

            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.DEGRADED
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")

            if memory_percent > self.memory_threshold:
                status = HealthStatus.UNHEALTHY if memory_percent > 95 else HealthStatus.DEGRADED
                messages.append(f"High memory usage: {memory_percent:.1f}%")

            if disk_percent > self.disk_threshold:
                status = HealthStatus.UNHEALTHY
                messages.append(f"High disk usage: {disk_percent:.1f}%")

            message = "; ".join(messages) if messages else "System resources OK"

            return HealthCheck(
                name="system",
                status=status,
                message=message,
                duration_ms=0,  # Will be set by caller
                timestamp=datetime.utcnow(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk_usage.free / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheck(
                name="system",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {str(e)}",
                duration_ms=0,
                timestamp=datetime.utcnow(),
            )

    async def _check_database(self) -> HealthCheck:
        """Check database connectivity and performance."""
        if not self.database_session_factory:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNKNOWN,
                message="Database session factory not configured",
                duration_ms=0,
                timestamp=datetime.utcnow(),
            )

        try:
            async with self.database_session_factory() as session:
                # Test basic connectivity
                start_time = time.time()
                result = await session.execute(text("SELECT 1"))
                query_time = (time.time() - start_time) * 1000

                # Get database stats
                stats_result = await session.execute(
                    text(
                        """
                    SELECT 
                        count(*) as connection_count
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                """
                    )
                )

                stats = stats_result.fetchone()

                status = HealthStatus.HEALTHY
                message = "Database connection OK"

                if query_time > self.response_time_threshold * 1000:
                    status = HealthStatus.DEGRADED
                    message = f"Slow database response: {query_time:.0f}ms"

                return HealthCheck(
                    name="database",
                    status=status,
                    message=message,
                    duration_ms=0,
                    timestamp=datetime.utcnow(),
                    details={
                        "query_time_ms": query_time,
                        "active_connections": stats[0] if stats else 0,
                    },
                )

        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                duration_ms=0,
                timestamp=datetime.utcnow(),
            )

    async def _check_redis(self) -> HealthCheck:
        """Check Redis connectivity and performance."""
        try:
            redis_client = redis.from_url(self.redis_url)

            # Test connectivity with ping
            start_time = time.time()
            pong = await redis_client.ping()
            response_time = (time.time() - start_time) * 1000

            # Get Redis info
            info = await redis_client.info()

            await redis_client.close()

            status = HealthStatus.HEALTHY
            message = "Redis connection OK"

            if response_time > self.response_time_threshold * 1000:
                status = HealthStatus.DEGRADED
                message = f"Slow Redis response: {response_time:.0f}ms"

            # Check memory usage
            used_memory = info.get("used_memory", 0)
            max_memory = info.get("maxmemory", 0)

            if max_memory > 0:
                memory_percent = (used_memory / max_memory) * 100
                if memory_percent > 90:
                    status = HealthStatus.DEGRADED
                    message += f"; High memory usage: {memory_percent:.1f}%"

            return HealthCheck(
                name="redis",
                status=status,
                message=message,
                duration_ms=0,
                timestamp=datetime.utcnow(),
                details={
                    "response_time_ms": response_time,
                    "used_memory_mb": used_memory / (1024 * 1024),
                    "connected_clients": info.get("connected_clients", 0),
                    "version": info.get("redis_version", "unknown"),
                },
            )

        except Exception as e:
            return HealthCheck(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                duration_ms=0,
                timestamp=datetime.utcnow(),
            )

    async def _check_external_services(self) -> HealthCheck:
        """Check external service connectivity."""
        if not self.external_services:
            return HealthCheck(
                name="external_services",
                status=HealthStatus.HEALTHY,
                message="No external services configured",
                duration_ms=0,
                timestamp=datetime.utcnow(),
            )

        service_results = {}
        overall_status = HealthStatus.HEALTHY
        failed_services = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, service_url in self.external_services.items():
                try:
                    start_time = time.time()
                    response = await client.get(f"{service_url}/health")
                    response_time = (time.time() - start_time) * 1000

                    if response.status_code == 200:
                        service_results[service_name] = {
                            "status": "healthy",
                            "response_time_ms": response_time,
                        }
                    else:
                        service_results[service_name] = {
                            "status": "unhealthy",
                            "response_time_ms": response_time,
                            "status_code": response.status_code,
                        }
                        failed_services.append(service_name)
                        overall_status = HealthStatus.DEGRADED

                except Exception as e:
                    service_results[service_name] = {"status": "unhealthy", "error": str(e)}
                    failed_services.append(service_name)
                    overall_status = HealthStatus.DEGRADED

        if failed_services:
            message = f"Failed services: {', '.join(failed_services)}"
        else:
            message = "All external services OK"

        return HealthCheck(
            name="external_services",
            status=overall_status,
            message=message,
            duration_ms=0,
            timestamp=datetime.utcnow(),
            details=service_results,
        )

    def _check_application_health(self) -> HealthCheck:
        """Check application-specific health indicators."""
        try:
            # Check if critical components are initialized
            details = {"metrics_collector": "initialized", "logging_system": "initialized"}

            # Add any application-specific checks here
            # For example, check if background tasks are running

            return HealthCheck(
                name="application",
                status=HealthStatus.HEALTHY,
                message="Application components OK",
                duration_ms=0,
                timestamp=datetime.utcnow(),
                details=details,
            )

        except Exception as e:
            return HealthCheck(
                name="application",
                status=HealthStatus.UNHEALTHY,
                message=f"Application health check failed: {str(e)}",
                duration_ms=0,
                timestamp=datetime.utcnow(),
            )

    def _get_status_message(self, status: HealthStatus, checks: List[HealthCheck]) -> str:
        """Get overall status message based on check results."""
        if status == HealthStatus.HEALTHY:
            return "All systems operational"
        elif status == HealthStatus.DEGRADED:
            failed_checks = [c.name for c in checks if c.status != HealthStatus.HEALTHY]
            return f"System degraded - issues with: {', '.join(failed_checks)}"
        elif status == HealthStatus.UNHEALTHY:
            failed_checks = [c.name for c in checks if c.status == HealthStatus.UNHEALTHY]
            return f"System unhealthy - critical issues with: {', '.join(failed_checks)}"
        else:
            return "System status unknown"


class HealthMonitor:
    """Continuous health monitoring with alerting."""

    def __init__(
        self,
        health_checker: HealthChecker,
        check_interval: int = 60,
        alert_callback: Callable = None,
    ):
        self.health_checker = health_checker
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        self.is_running = False
        self.last_status = HealthStatus.UNKNOWN
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start continuous health monitoring."""
        if self.is_running:
            logger.warning("Health monitor is already running")
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    async def stop(self):
        """Stop health monitoring."""
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Perform health check
                health = await self.health_checker.check_health()

                # Check if status changed
                if health.status != self.last_status:
                    await self._handle_status_change(self.last_status, health)
                    self.last_status = health.status

                # Wait for next check
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(min(self.check_interval, 60))  # Wait at least 60s on error

    async def _handle_status_change(self, old_status: HealthStatus, health: SystemHealth):
        """Handle health status changes."""
        logger.info(
            f"Health status changed: {old_status.value} -> {health.status.value}",
            event_type="system",
            old_status=old_status.value,
            new_status=health.status.value,
            message=health.message,
        )

        # Call alert callback if configured
        if self.alert_callback:
            try:
                await self.alert_callback(old_status, health)
            except Exception as e:
                logger.error(f"Error in health alert callback: {e}")


# Global health checker instance
_health_checker: Optional[HealthChecker] = None
_health_monitor: Optional[HealthMonitor] = None


def get_health_checker(
    database_session_factory: Optional[Callable] = None,
    redis_url: str = "redis://localhost:6379",
    external_services: Dict[str, str] = None,
) -> HealthChecker:
    """Get global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(
            database_session_factory=database_session_factory,
            redis_url=redis_url,
            external_services=external_services,
        )
    return _health_checker


def get_health_monitor(
    health_checker: HealthChecker = None, check_interval: int = 60, alert_callback: Callable = None
) -> HealthMonitor:
    """Get global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        checker = health_checker or get_health_checker()
        _health_monitor = HealthMonitor(
            health_checker=checker, check_interval=check_interval, alert_callback=alert_callback
        )
    return _health_monitor
