"""Comprehensive performance management system integrating all optimization components."""

import asyncio
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

from .api_optimizer import APIOptimizer
from .cache_manager import CacheConfig, CacheLayer, CacheManager
from .circuit_breaker_manager import CircuitBreakerManager
from .connection_optimizer import ConnectionOptimizer
from .external_service_optimizer import ExternalServiceConfig, ExternalServiceOptimizer
from .memory_manager import MemoryManager
from .metrics_collector import MetricsCollector, PerformanceMetrics
from .monitoring_setup import (
    PerformanceMonitor,
    init_performance_monitor,
    shutdown_performance_monitor,
)
from .query_cache import QueryCacheManager


@dataclass
class PerformanceConfiguration:
    """Comprehensive performance configuration."""

    # Cache configuration
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    enable_query_cache: bool = True

    # Memory management
    enable_memory_monitoring: bool = True
    enable_object_pooling: bool = True
    gc_optimization: bool = True

    # Connection optimization
    enable_connection_optimization: bool = True
    enable_http_optimization: bool = True

    # Circuit breaker configuration
    enable_circuit_breakers: bool = True
    auto_recovery: bool = True

    # API optimization
    enable_api_compression: bool = True
    enable_api_caching: bool = True
    enable_request_batching: bool = True

    # External service optimization
    enable_external_service_optimization: bool = True

    # Monitoring
    enable_metrics: bool = True
    enable_prometheus: bool = True
    prometheus_port: int = 8000

    # Performance targets
    target_response_time_ms: int = 200
    target_cache_hit_rate: float = 0.8
    target_memory_usage_mb: int = 512
    target_uptime: float = 0.999


class PerformanceManager:
    """Comprehensive performance management system."""

    def __init__(self, config: PerformanceConfiguration):
        self.config = config

        # Core components
        self.metrics_collector: Optional[MetricsCollector] = None
        self.cache_manager: Optional[CacheManager] = None
        self.query_cache_manager: Optional[QueryCacheManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.connection_optimizer: Optional[ConnectionOptimizer] = None
        self.circuit_breaker_manager: Optional[CircuitBreakerManager] = None
        self.api_optimizer: Optional[APIOptimizer] = None
        self.external_service_optimizer: Optional[ExternalServiceOptimizer] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None

        # State management
        self._initialized = False
        self._monitoring_enabled = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._performance_alerts: List[Dict[str, Any]] = []

        # Performance tracking
        self._performance_history: List[PerformanceMetrics] = []
        self._alert_thresholds = {
            "response_time_ms": config.target_response_time_ms,
            "cache_hit_rate": config.target_cache_hit_rate,
            "memory_usage_mb": config.target_memory_usage_mb,
            "error_rate": 0.01,  # 1% error rate threshold
            "uptime": config.target_uptime,
        }

    async def initialize(self) -> None:
        """Initialize all performance components."""
        if self._initialized:
            return

        print("Initializing Performance Manager...")

        try:
            # Initialize metrics collector
            if self.config.enable_metrics:
                self.metrics_collector = MetricsCollector(
                    enable_prometheus=self.config.enable_prometheus,
                    prometheus_port=self.config.prometheus_port,
                )
                await self.metrics_collector.start_background_collection()
                print("✅ Metrics collector initialized")

            # Initialize cache manager
            self.cache_manager = CacheManager(self.config.cache_config)
            await self.cache_manager.initialize()
            print("✅ Cache manager initialized")

            # Initialize query cache manager
            if self.config.enable_query_cache:
                self.query_cache_manager = QueryCacheManager(self.cache_manager)
                print("✅ Query cache manager initialized")

            # Initialize memory manager
            if self.config.enable_memory_monitoring:
                self.memory_manager = MemoryManager(
                    enable_monitoring=self.config.enable_memory_monitoring
                )
                await self.memory_manager.initialize()
                print("✅ Memory manager initialized")

            # Initialize connection optimizer
            if self.config.enable_connection_optimization:
                self.connection_optimizer = ConnectionOptimizer(self.metrics_collector)
                await self.connection_optimizer.initialize()
                print("✅ Connection optimizer initialized")

            # Initialize circuit breaker manager
            if self.config.enable_circuit_breakers:
                self.circuit_breaker_manager = CircuitBreakerManager(self.metrics_collector)
                await self.circuit_breaker_manager.start_monitoring()
                print("✅ Circuit breaker manager initialized")

            # Initialize API optimizer
            if any([self.config.enable_api_compression, self.config.enable_api_caching]):
                self.api_optimizer = APIOptimizer(self.cache_manager, self.metrics_collector)
                await self.api_optimizer.initialize()
                print("✅ API optimizer initialized")

            # Initialize external service optimizer
            if self.config.enable_external_service_optimization:
                self.external_service_optimizer = ExternalServiceOptimizer(
                    connection_optimizer=self.connection_optimizer,
                    cache_manager=self.cache_manager,
                    circuit_breaker_manager=self.circuit_breaker_manager,
                    metrics_collector=self.metrics_collector,
                )

                # Register common external services
                self._register_external_services()
                print("✅ External service optimizer initialized")

            # Initialize performance monitoring
            if self.config.enable_metrics:
                self.performance_monitor = await init_performance_monitor(
                    metrics_collector=self.metrics_collector,
                    enable_console_alerts=True,
                    enable_logging_alerts=True,
                )
                print("✅ Performance monitoring initialized")

            self._initialized = True
            print("🚀 Performance Manager fully initialized")

        except Exception as e:
            print(f"❌ Performance Manager initialization failed: {e}")
            await self.shutdown()
            raise

    def _register_external_services(self) -> None:
        """Register common external services with the optimizer."""
        if not self.external_service_optimizer:
            return

        # OpenAI service
        openai_config = ExternalServiceConfig(
            service_name="openai",
            base_url="https://api.openai.com",
            timeout_seconds=60.0,
            max_retries=3,
            cache_enabled=True,
            cache_ttl_seconds=300,
        )
        self.external_service_optimizer.register_service(openai_config)

        # Anthropic service
        anthropic_config = ExternalServiceConfig(
            service_name="anthropic",
            base_url="https://api.anthropic.com",
            timeout_seconds=60.0,
            max_retries=3,
            cache_enabled=True,
            cache_ttl_seconds=300,
        )
        self.external_service_optimizer.register_service(anthropic_config)

        # Google AI service
        google_config = ExternalServiceConfig(
            service_name="google_ai",
            base_url="https://generativelanguage.googleapis.com",
            timeout_seconds=45.0,
            max_retries=3,
            cache_enabled=True,
            cache_ttl_seconds=300,
        )
        self.external_service_optimizer.register_service(google_config)

    async def shutdown(self) -> None:
        """Shutdown all performance components."""
        print("Shutting down Performance Manager...")

        # Stop monitoring
        await self.stop_monitoring()

        # Shutdown performance monitoring
        if self.performance_monitor:
            await shutdown_performance_monitor()
            print("  Performance monitoring stopped")

        # Shutdown components in reverse order
        if self.api_optimizer:
            print("  Shutting down API optimizer...")

        if self.external_service_optimizer:
            print("  External service optimizer stopped")

        if self.circuit_breaker_manager:
            await self.circuit_breaker_manager.stop_monitoring()
            print("  Circuit breaker manager stopped")

        if self.connection_optimizer:
            await self.connection_optimizer.shutdown()
            print("  Connection optimizer stopped")

        if self.memory_manager:
            await self.memory_manager.shutdown()
            print("  Memory manager stopped")

        if self.cache_manager:
            await self.cache_manager.close()
            print("  Cache manager stopped")

        self._initialized = False
        print("✅ Performance Manager shutdown complete")

    async def start_monitoring(self) -> None:
        """Start comprehensive performance monitoring."""
        if self._monitoring_enabled or not self._initialized:
            return

        self._monitoring_enabled = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        print("📊 Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring_enabled = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        print("📊 Performance monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_enabled:
            try:
                await self._collect_performance_metrics()
                await self._check_performance_alerts()
                await asyncio.sleep(30)  # Monitor every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                await asyncio.sleep(30)

    async def _collect_performance_metrics(self) -> None:
        """Collect comprehensive performance metrics."""
        if not self.metrics_collector:
            return

        current_metrics = self.metrics_collector.get_current_metrics()
        self._performance_history.append(current_metrics)

        # Keep only last 1000 metrics (about 8 hours at 30s intervals)
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]

    async def _check_performance_alerts(self) -> None:
        """Check for performance threshold violations."""
        if not self._performance_history:
            return

        latest_metrics = self._performance_history[-1]
        current_time = datetime.utcnow()

        # Check response time
        if latest_metrics.avg_response_time * 1000 > self._alert_thresholds["response_time_ms"]:
            await self._create_alert(
                "high_response_time",
                f"Response time {latest_metrics.avg_response_time * 1000:.1f}ms exceeds threshold",
                "warning",
            )

        # Check cache hit rate
        if latest_metrics.cache_hit_rate < self._alert_thresholds["cache_hit_rate"]:
            await self._create_alert(
                "low_cache_hit_rate",
                f"Cache hit rate {latest_metrics.cache_hit_rate:.1%} below threshold",
                "warning",
            )

        # Check memory usage
        if latest_metrics.memory_usage_mb > self._alert_thresholds["memory_usage_mb"]:
            await self._create_alert(
                "high_memory_usage",
                f"Memory usage {latest_metrics.memory_usage_mb:.1f}MB exceeds threshold",
                "critical",
            )

        # Check error rate
        error_rate = latest_metrics.failed_requests / max(latest_metrics.total_requests, 1)
        if error_rate > self._alert_thresholds["error_rate"]:
            await self._create_alert(
                "high_error_rate", f"Error rate {error_rate:.1%} exceeds threshold", "critical"
            )

    async def _create_alert(self, alert_type: str, message: str, severity: str) -> None:
        """Create performance alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "acknowledged": False,
        }

        self._performance_alerts.append(alert)

        # Keep only last 100 alerts
        if len(self._performance_alerts) > 100:
            self._performance_alerts = self._performance_alerts[-100:]

        print(f"🚨 PERFORMANCE ALERT [{severity.upper()}]: {message}")

    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        dashboard = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy" if self._initialized else "not_initialized",
            "configuration": {
                "cache_enabled": self.cache_manager is not None,
                "memory_monitoring": self.memory_manager is not None,
                "connection_optimization": self.connection_optimizer is not None,
                "circuit_breakers": self.circuit_breaker_manager is not None,
                "api_optimization": self.api_optimizer is not None,
                "external_service_optimization": self.external_service_optimizer is not None,
                "performance_monitoring": self.performance_monitor is not None,
            },
            "alerts": {
                "active_alerts": len(
                    [a for a in self._performance_alerts if not a["acknowledged"]]
                ),
                "recent_alerts": self._performance_alerts[-10:] if self._performance_alerts else [],
            },
        }

        # Cache statistics
        if self.cache_manager:
            dashboard["cache"] = await self.cache_manager.get_stats()

        # Memory statistics
        if self.memory_manager:
            dashboard["memory"] = self.memory_manager.get_memory_stats()

        # Connection statistics
        if self.connection_optimizer:
            dashboard["connections"] = self.connection_optimizer.get_connection_stats()

        # Circuit breaker statistics
        if self.circuit_breaker_manager:
            dashboard["circuit_breakers"] = (
                self.circuit_breaker_manager.get_circuit_breaker_status()
            )

        # API optimization statistics
        if self.api_optimizer:
            dashboard["api_optimization"] = self.api_optimizer.get_optimization_stats()

        # External service optimization statistics
        if self.external_service_optimizer:
            dashboard["external_services"] = self.external_service_optimizer.get_service_metrics()

        # Performance monitoring statistics
        if self.performance_monitor:
            dashboard["monitoring"] = self.performance_monitor.get_alert_summary()

        # Performance metrics
        if self.metrics_collector:
            dashboard["metrics"] = self.metrics_collector.get_metrics_summary()

        # Performance trends
        if len(self._performance_history) >= 2:
            dashboard["trends"] = self._calculate_performance_trends()

        return dashboard

    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends from historical data."""
        if len(self._performance_history) < 10:
            return {}

        recent_metrics = self._performance_history[-10:]
        older_metrics = (
            self._performance_history[-20:-10] if len(self._performance_history) >= 20 else []
        )

        trends = {}

        # Response time trend
        recent_avg_response = sum(m.avg_response_time for m in recent_metrics) / len(recent_metrics)
        if older_metrics:
            older_avg_response = sum(m.avg_response_time for m in older_metrics) / len(
                older_metrics
            )
            trends["response_time_trend"] = (
                recent_avg_response - older_avg_response
            ) / older_avg_response

        # Cache hit rate trend
        recent_cache_hit = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        if older_metrics:
            older_cache_hit = sum(m.cache_hit_rate for m in older_metrics) / len(older_metrics)
            if older_cache_hit > 0:
                trends["cache_hit_rate_trend"] = (
                    recent_cache_hit - older_cache_hit
                ) / older_cache_hit

        # Memory usage trend
        recent_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        if older_metrics:
            older_memory = sum(m.memory_usage_mb for m in older_metrics) / len(older_metrics)
            if older_memory > 0:
                trends["memory_usage_trend"] = (recent_memory - older_memory) / older_memory

        return trends

    @asynccontextmanager
    async def performance_context(
        self,
        operation_name: str,
        enable_caching: bool = True,
        enable_circuit_breaker: bool = True,
        cache_ttl: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Context manager for performance-optimized operations."""
        context = {
            "operation_name": operation_name,
            "start_time": time.time(),
            "cache_manager": self.cache_manager if enable_caching else None,
            "circuit_breaker_manager": (
                self.circuit_breaker_manager if enable_circuit_breaker else None
            ),
            "metrics": {},
        }

        try:
            yield context

            # Record successful operation
            duration = time.time() - context["start_time"]
            if self.metrics_collector:
                self.metrics_collector.record_custom_timer(f"operation_{operation_name}", duration)

            context["metrics"]["success"] = True
            context["metrics"]["duration"] = duration

        except Exception as e:
            # Record failed operation
            duration = time.time() - context["start_time"]
            if self.metrics_collector:
                self.metrics_collector.increment_custom_counter(
                    f"operation_{operation_name}_errors", labels={"error_type": type(e).__name__}
                )

            context["metrics"]["success"] = False
            context["metrics"]["duration"] = duration
            context["metrics"]["error"] = str(e)

            raise

    async def optimize_performance(self) -> Dict[str, Any]:
        """Run comprehensive performance optimizations."""
        optimization_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "optimizations_applied": [],
        }

        # Cache optimization
        if self.cache_manager:
            cache_stats = await self.cache_manager.get_stats()
            if cache_stats["memory_cache"]["utilization"] > 0.9:
                await self.cache_manager.clear("expired_entries")
                optimization_results["optimizations_applied"].append("cache_cleanup")

        # Memory optimization
        if self.memory_manager:
            memory_results = await self.memory_manager.optimize_memory_usage()
            optimization_results["memory_optimization"] = memory_results
            optimization_results["optimizations_applied"].append("memory_gc")

        # Connection optimization
        if self.connection_optimizer:
            conn_results = await self.connection_optimizer.optimize_http_clients()
            optimization_results["connection_optimization"] = conn_results
            optimization_results["optimizations_applied"].append("connection_tuning")

        # API optimization
        if self.api_optimizer:
            api_results = await self.api_optimizer.optimize_api_configuration()
            optimization_results["api_optimization"] = api_results
            optimization_results["optimizations_applied"].append("api_tuning")

        # External service optimization
        if self.external_service_optimizer:
            ext_results = await self.external_service_optimizer.optimize_services()
            optimization_results["external_service_optimization"] = ext_results
            optimization_results["optimizations_applied"].append("external_service_tuning")

        return optimization_results

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all performance components."""
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
        }

        # Check each component
        components_to_check = [
            ("cache_manager", self.cache_manager),
            ("memory_manager", self.memory_manager),
            ("connection_optimizer", self.connection_optimizer),
            ("circuit_breaker_manager", self.circuit_breaker_manager),
            ("metrics_collector", self.metrics_collector),
        ]

        for component_name, component in components_to_check:
            if component is None:
                health_status["components"][component_name] = {
                    "status": "disabled",
                    "message": "Component not initialized",
                }
                continue

            try:
                # Perform component-specific health check
                if hasattr(component, "health_check"):
                    component_health = await component.health_check()
                    health_status["components"][component_name] = component_health
                else:
                    health_status["components"][component_name] = {
                        "status": "healthy",
                        "message": "Component active",
                    }
            except Exception as e:
                health_status["components"][component_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_status["overall_status"] = "degraded"

        # Check performance thresholds
        if self._performance_history:
            latest_metrics = self._performance_history[-1]

            performance_checks = {
                "response_time": latest_metrics.avg_response_time * 1000
                < self._alert_thresholds["response_time_ms"],
                "cache_hit_rate": latest_metrics.cache_hit_rate
                > self._alert_thresholds["cache_hit_rate"],
                "memory_usage": latest_metrics.memory_usage_mb
                < self._alert_thresholds["memory_usage_mb"],
                "error_rate": (
                    latest_metrics.failed_requests / max(latest_metrics.total_requests, 1)
                )
                < self._alert_thresholds["error_rate"],
            }

            health_status["performance_checks"] = performance_checks

            if not all(performance_checks.values()):
                health_status["overall_status"] = "performance_issues"

        return health_status


# Global performance manager instance
_performance_manager: Optional[PerformanceManager] = None


def get_performance_manager() -> Optional[PerformanceManager]:
    """Get global performance manager instance."""
    return _performance_manager


async def init_performance_manager(config: PerformanceConfiguration) -> PerformanceManager:
    """Initialize global performance manager."""
    global _performance_manager
    _performance_manager = PerformanceManager(config)
    await _performance_manager.initialize()
    await _performance_manager.start_monitoring()
    return _performance_manager


async def shutdown_performance_manager() -> None:
    """Shutdown global performance manager."""
    global _performance_manager
    if _performance_manager:
        await _performance_manager.shutdown()
        _performance_manager = None
