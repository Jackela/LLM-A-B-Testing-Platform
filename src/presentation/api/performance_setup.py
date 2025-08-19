"""Performance optimization setup for FastAPI application."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from ...infrastructure.performance.cache_manager import CacheConfig, CacheLayer
from ...infrastructure.performance.config import PerformanceConfig
from ...infrastructure.performance.performance_manager import (
    PerformanceConfiguration,
    PerformanceManager,
    get_performance_manager,
    init_performance_manager,
    shutdown_performance_manager,
)


def create_performance_config() -> PerformanceConfiguration:
    """Create performance configuration from environment variables."""

    # Cache configuration
    cache_config = CacheConfig(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        redis_db=int(os.getenv("REDIS_DB", "0")),
        redis_pool_size=int(os.getenv("REDIS_POOL_SIZE", "20")),
        memory_max_size=int(os.getenv("MEMORY_CACHE_SIZE", "10000")),
        memory_ttl_default=int(os.getenv("MEMORY_CACHE_TTL", "300")),
        compression_enabled=os.getenv("CACHE_COMPRESSION", "true").lower() == "true",
        enable_metrics=os.getenv("CACHE_METRICS", "true").lower() == "true",
    )

    # Performance configuration
    perf_config = PerformanceConfiguration(
        cache_config=cache_config,
        enable_query_cache=os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true",
        enable_memory_monitoring=os.getenv("ENABLE_MEMORY_MONITORING", "true").lower() == "true",
        enable_connection_optimization=os.getenv("ENABLE_CONNECTION_OPT", "true").lower() == "true",
        enable_circuit_breakers=os.getenv("ENABLE_CIRCUIT_BREAKERS", "true").lower() == "true",
        enable_api_compression=os.getenv("ENABLE_API_COMPRESSION", "true").lower() == "true",
        enable_api_caching=os.getenv("ENABLE_API_CACHING", "true").lower() == "true",
        enable_external_service_optimization=os.getenv("ENABLE_EXT_SERVICE_OPT", "true").lower()
        == "true",
        enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
        enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "false").lower() == "true",
        prometheus_port=int(os.getenv("PROMETHEUS_PORT", "8000")),
        target_response_time_ms=int(os.getenv("TARGET_RESPONSE_TIME_MS", "200")),
        target_cache_hit_rate=float(os.getenv("TARGET_CACHE_HIT_RATE", "0.8")),
        target_memory_usage_mb=int(os.getenv("TARGET_MEMORY_MB", "512")),
    )

    return perf_config


@asynccontextmanager
async def performance_lifespan(app: FastAPI):
    """FastAPI lifespan context manager with performance optimization."""

    # Startup
    print("🚀 Initializing Performance Optimization System...")

    try:
        # Create performance configuration
        perf_config = create_performance_config()

        # Initialize performance manager
        performance_manager = await init_performance_manager(perf_config)

        # Store in app state for access in routes
        app.state.performance_manager = performance_manager

        print("✅ Performance optimization system initialized")
        print(
            f"   Cache: {'✓ Redis + Memory' if performance_manager.cache_manager else '✗ Disabled'}"
        )
        print(
            f"   Memory Monitoring: {'✓ Enabled' if performance_manager.memory_manager else '✗ Disabled'}"
        )
        print(
            f"   Circuit Breakers: {'✓ Enabled' if performance_manager.circuit_breaker_manager else '✗ Disabled'}"
        )
        print(
            f"   API Optimization: {'✓ Enabled' if performance_manager.api_optimizer else '✗ Disabled'}"
        )
        print(
            f"   Metrics Collection: {'✓ Enabled' if performance_manager.metrics_collector else '✗ Disabled'}"
        )

        # Run initial performance optimization
        optimization_results = await performance_manager.optimize_performance()
        print(
            f"   Initial Optimization: Applied {len(optimization_results.get('optimizations_applied', []))} optimizations"
        )

        yield

    except Exception as e:
        print(f"❌ Performance system initialization failed: {e}")
        # Continue without performance optimization
        app.state.performance_manager = None
        yield

    # Shutdown
    print("🔄 Shutting down Performance Optimization System...")
    try:
        await shutdown_performance_manager()
        print("✅ Performance optimization system shutdown complete")
    except Exception as e:
        print(f"⚠️  Error during performance system shutdown: {e}")


def add_performance_middleware(app: FastAPI) -> None:
    """Add performance optimization middleware to FastAPI app."""

    @app.middleware("http")
    async def performance_middleware(request, call_next):
        """Performance monitoring and optimization middleware."""

        # Get performance manager
        performance_manager = getattr(app.state, "performance_manager", None)
        if not performance_manager:
            return await call_next(request)

        # Performance context for the request
        operation_name = f"{request.method}_{request.url.path.replace('/', '_')}"

        async with performance_manager.performance_context(
            operation_name=operation_name, enable_caching=True, enable_circuit_breaker=True
        ) as context:

            # Add performance headers
            response = await call_next(request)

            # Add performance metrics to response headers
            if context.get("metrics"):
                response.headers["X-Response-Time"] = (
                    f"{context['metrics'].get('duration', 0) * 1000:.1f}ms"
                )
                response.headers["X-Cache-Status"] = "hit" if context.get("cache_hit") else "miss"
                response.headers["X-Performance-Optimized"] = "true"

            return response


def add_performance_routes(app: FastAPI) -> None:
    """Add performance monitoring routes to FastAPI app."""

    @app.get("/health/performance", tags=["Performance"])
    async def performance_health():
        """Get performance system health status."""
        performance_manager = getattr(app.state, "performance_manager", None)
        if not performance_manager:
            return {"status": "disabled", "message": "Performance optimization not enabled"}

        return await performance_manager.health_check()

    @app.get("/admin/performance/dashboard", tags=["Admin", "Performance"])
    async def performance_dashboard():
        """Get comprehensive performance dashboard data."""
        performance_manager = getattr(app.state, "performance_manager", None)
        if not performance_manager:
            return {"error": "Performance optimization not enabled"}

        return await performance_manager.get_performance_dashboard()

    @app.post("/admin/performance/optimize", tags=["Admin", "Performance"])
    async def optimize_performance():
        """Run comprehensive performance optimizations."""
        performance_manager = getattr(app.state, "performance_manager", None)
        if not performance_manager:
            return {"error": "Performance optimization not enabled"}

        return await performance_manager.optimize_performance()

    @app.get("/admin/cache/stats", tags=["Admin", "Cache"])
    async def cache_stats():
        """Get cache statistics."""
        performance_manager = getattr(app.state, "performance_manager", None)
        if not performance_manager or not performance_manager.cache_manager:
            return {"error": "Cache not enabled"}

        return await performance_manager.cache_manager.get_stats()

    @app.post("/admin/cache/clear", tags=["Admin", "Cache"])
    async def clear_cache(namespace: str = ""):
        """Clear cache with optional namespace."""
        performance_manager = getattr(app.state, "performance_manager", None)
        if not performance_manager or not performance_manager.cache_manager:
            return {"error": "Cache not enabled"}

        await performance_manager.cache_manager.clear(namespace)
        return {"message": f"Cache cleared{f' for namespace: {namespace}' if namespace else ''}"}


# Cache decorator for API routes
def cached_response(
    ttl: int = 300,
    namespace: str = "api",
    layer: CacheLayer = CacheLayer.HYBRID,
    key_func: Optional[callable] = None,
):
    """Decorator for caching API response data."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get FastAPI app from somewhere (could be dependency injected)
            # For now, we'll use the global performance manager
            performance_manager = get_performance_manager()

            if not performance_manager or not performance_manager.cache_manager:
                return await func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                import hashlib
                import json

                # Create key from function name and arguments
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": json.dumps(kwargs, sort_keys=True, default=str),
                }
                cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

            # Try cache first
            cached_result = await performance_manager.cache_manager.get(cache_key, namespace, layer)
            if cached_result is not None:
                return cached_result

            # Compute result and cache it
            result = await func(*args, **kwargs)
            await performance_manager.cache_manager.set(cache_key, result, ttl, namespace, layer)

            return result

        return wrapper

    return decorator
