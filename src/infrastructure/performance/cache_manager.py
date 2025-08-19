"""Advanced multi-layer caching infrastructure with Redis and in-memory caching."""

import asyncio
import hashlib
import json
import pickle
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

import redis.asyncio as redis
from pydantic import BaseModel


class CacheLayer(Enum):
    """Cache layer types."""

    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


@dataclass
class CacheConfig:
    """Cache configuration settings."""

    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_pool_size: int = 20
    redis_max_connections: int = 50

    # Memory cache configuration
    memory_max_size: int = 10000  # Max number of items
    memory_ttl_default: int = 300  # 5 minutes default TTL

    # Hybrid cache configuration
    memory_first: bool = True  # Check memory before Redis
    promote_to_memory: bool = True  # Promote frequently accessed items

    # Performance settings
    compression_enabled: bool = True
    compression_threshold: int = 1024  # Compress items larger than 1KB
    serialization_format: str = "pickle"  # pickle or json

    # Cache warming
    enable_cache_warming: bool = True
    warm_up_batch_size: int = 100

    # Metrics
    enable_metrics: bool = True
    metrics_sample_rate: float = 1.0


class CacheMetrics:
    """Cache performance metrics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.total_time = 0.0
        self.last_reset = time.time()

    def record_hit(self, duration: float = 0) -> None:
        """Record cache hit."""
        self.hits += 1
        self.total_time += duration

    def record_miss(self, duration: float = 0) -> None:
        """Record cache miss."""
        self.misses += 1
        self.total_time += duration

    def record_set(self, duration: float = 0) -> None:
        """Record cache set operation."""
        self.sets += 1
        self.total_time += duration

    def record_delete(self, duration: float = 0) -> None:
        """Record cache delete operation."""
        self.deletes += 1
        self.total_time += duration

    def record_eviction(self) -> None:
        """Record cache eviction."""
        self.evictions += 1

    def record_error(self) -> None:
        """Record cache error."""
        self.errors += 1

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_average_time(self) -> float:
        """Calculate average operation time."""
        total_ops = self.hits + self.misses + self.sets + self.deletes
        return self.total_time / total_ops if total_ops > 0 else 0.0

    def reset(self) -> None:
        """Reset metrics."""
        self.__init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "errors": self.errors,
            "hit_rate": self.get_hit_rate(),
            "average_time_ms": self.get_average_time() * 1000,
            "total_time": self.total_time,
            "uptime_seconds": time.time() - self.last_reset,
        }


class MemoryCache:
    """High-performance in-memory cache with LRU eviction."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._data: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get item from memory cache."""
        async with self._lock:
            if key in self._data:
                # Check TTL
                if key in self._timestamps:
                    if time.time() > self._timestamps[key]:
                        await self._remove_key(key)
                        return None

                # Update access order for LRU
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)

                return self._data[key]

            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in memory cache."""
        async with self._lock:
            # Set TTL
            if ttl is not None:
                self._timestamps[key] = time.time() + ttl
            elif self.default_ttl > 0:
                self._timestamps[key] = time.time() + self.default_ttl

            # Add/update data
            self._data[key] = value

            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            # Evict if necessary
            await self._evict_if_needed()

    async def delete(self, key: str) -> bool:
        """Delete item from memory cache."""
        async with self._lock:
            if key in self._data:
                await self._remove_key(key)
                return True
            return False

    async def clear(self) -> None:
        """Clear all items from memory cache."""
        async with self._lock:
            self._data.clear()
            self._timestamps.clear()
            self._access_order.clear()

    async def _remove_key(self, key: str) -> None:
        """Remove key from all internal structures."""
        self._data.pop(key, None)
        self._timestamps.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    async def _evict_if_needed(self) -> int:
        """Evict items if cache is over capacity."""
        evicted = 0

        # First, remove expired items
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items() if current_time > timestamp
        ]

        for key in expired_keys:
            await self._remove_key(key)
            evicted += 1

        # Then, evict LRU items if still over capacity
        while len(self._data) > self.max_size and self._access_order:
            oldest_key = self._access_order[0]
            await self._remove_key(oldest_key)
            evicted += 1

        return evicted

    def size(self) -> int:
        """Get current cache size."""
        return len(self._data)

    def capacity(self) -> int:
        """Get cache capacity."""
        return self.max_size


class CacheManager:
    """Advanced multi-layer cache manager with Redis and in-memory caching."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCache(config.memory_max_size, config.memory_ttl_default)
        self.redis_client: Optional[redis.Redis] = None
        self.metrics = CacheMetrics()
        self._compression_enabled = config.compression_enabled
        self._ready = False

    async def initialize(self) -> None:
        """Initialize cache manager and connections."""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis.from_url(
                self.config.redis_url,
                db=self.config.redis_db,
                max_connections=self.config.redis_max_connections,
                health_check_interval=30,
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                decode_responses=False,  # We handle encoding ourselves
            )

            # Test Redis connection
            await self.redis_client.ping()
            self._ready = True

        except Exception as e:
            print(f"Failed to initialize Redis cache: {e}")
            # Continue with memory-only cache
            self.redis_client = None
            self._ready = True

    async def close(self) -> None:
        """Close cache connections."""
        if self.redis_client:
            await self.redis_client.close()
        await self.memory_cache.clear()
        self._ready = False

    def _generate_cache_key(self, key: str, namespace: str = "") -> str:
        """Generate cache key with namespace."""
        if namespace:
            return f"{namespace}:{key}"
        return key

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.config.serialization_format == "json":
            try:
                return json.dumps(value).encode("utf-8")
            except (TypeError, ValueError):
                # Fallback to pickle for non-JSON serializable objects
                return pickle.dumps(value)
        else:
            return pickle.dumps(value)

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try pickle first (more reliable)
            return pickle.loads(data)
        except (pickle.PickleError, TypeError):
            try:
                # Fallback to JSON
                return json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return data  # Return raw data if can't deserialize

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if enabled and beneficial."""
        if self._compression_enabled and len(data) > self.config.compression_threshold:
            import zlib

            return zlib.compress(data)
        return data

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if it was compressed."""
        if self._compression_enabled:
            try:
                import zlib

                return zlib.decompress(data)
            except zlib.error:
                # Data wasn't compressed
                return data
        return data

    async def get(
        self, key: str, namespace: str = "", layer: CacheLayer = CacheLayer.HYBRID
    ) -> Optional[Any]:
        """Get value from cache."""
        if not self._ready:
            return None

        start_time = time.time()
        cache_key = self._generate_cache_key(key, namespace)

        try:
            # Try memory cache first (if hybrid or memory only)
            if layer in [CacheLayer.MEMORY, CacheLayer.HYBRID]:
                value = await self.memory_cache.get(cache_key)
                if value is not None:
                    duration = time.time() - start_time
                    self.metrics.record_hit(duration)
                    return value

            # Try Redis cache (if hybrid or Redis only)
            if layer in [CacheLayer.REDIS, CacheLayer.HYBRID] and self.redis_client:
                try:
                    data = await self.redis_client.get(cache_key)
                    if data is not None:
                        # Decompress and deserialize
                        decompressed_data = self._decompress_data(data)
                        value = self._deserialize_value(decompressed_data)

                        # Promote to memory cache if hybrid and enabled
                        if layer == CacheLayer.HYBRID and self.config.promote_to_memory:
                            await self.memory_cache.set(cache_key, value)

                        duration = time.time() - start_time
                        self.metrics.record_hit(duration)
                        return value

                except Exception as e:
                    self.metrics.record_error()
                    print(f"Redis get error: {e}")

            # Cache miss
            duration = time.time() - start_time
            self.metrics.record_miss(duration)
            return None

        except Exception as e:
            self.metrics.record_error()
            print(f"Cache get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "",
        layer: CacheLayer = CacheLayer.HYBRID,
    ) -> bool:
        """Set value in cache."""
        if not self._ready:
            return False

        start_time = time.time()
        cache_key = self._generate_cache_key(key, namespace)
        success = False

        try:
            # Set in memory cache
            if layer in [CacheLayer.MEMORY, CacheLayer.HYBRID]:
                await self.memory_cache.set(cache_key, value, ttl)
                success = True

            # Set in Redis cache
            if layer in [CacheLayer.REDIS, CacheLayer.HYBRID] and self.redis_client:
                try:
                    # Serialize and compress
                    serialized_data = self._serialize_value(value)
                    compressed_data = self._compress_data(serialized_data)

                    if ttl is not None:
                        await self.redis_client.setex(cache_key, ttl, compressed_data)
                    else:
                        await self.redis_client.set(cache_key, compressed_data)

                    success = True

                except Exception as e:
                    self.metrics.record_error()
                    print(f"Redis set error: {e}")

            duration = time.time() - start_time
            self.metrics.record_set(duration)
            return success

        except Exception as e:
            self.metrics.record_error()
            print(f"Cache set error: {e}")
            return False

    async def delete(
        self, key: str, namespace: str = "", layer: CacheLayer = CacheLayer.HYBRID
    ) -> bool:
        """Delete value from cache."""
        if not self._ready:
            return False

        start_time = time.time()
        cache_key = self._generate_cache_key(key, namespace)
        success = False

        try:
            # Delete from memory cache
            if layer in [CacheLayer.MEMORY, CacheLayer.HYBRID]:
                await self.memory_cache.delete(cache_key)
                success = True

            # Delete from Redis cache
            if layer in [CacheLayer.REDIS, CacheLayer.HYBRID] and self.redis_client:
                try:
                    result = await self.redis_client.delete(cache_key)
                    success = success or (result > 0)
                except Exception as e:
                    self.metrics.record_error()
                    print(f"Redis delete error: {e}")

            duration = time.time() - start_time
            self.metrics.record_delete(duration)
            return success

        except Exception as e:
            self.metrics.record_error()
            print(f"Cache delete error: {e}")
            return False

    async def clear(self, namespace: str = "") -> None:
        """Clear cache items."""
        try:
            if namespace:
                # Clear namespace in Redis
                if self.redis_client:
                    pattern = f"{namespace}:*"
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)

                # Clear namespace in memory cache (simplified - clear all)
                await self.memory_cache.clear()
            else:
                # Clear all
                if self.redis_client:
                    await self.redis_client.flushdb()
                await self.memory_cache.clear()

        except Exception as e:
            self.metrics.record_error()
            print(f"Cache clear error: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache": {
                "size": self.memory_cache.size(),
                "capacity": self.memory_cache.capacity(),
                "utilization": self.memory_cache.size() / self.memory_cache.capacity(),
            },
            "redis_cache": {"connected": self.redis_client is not None, "info": {}},
            "metrics": self.metrics.to_dict(),
        }

        # Get Redis info if available
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info("memory")
                stats["redis_cache"]["info"] = {
                    "used_memory": redis_info.get("used_memory", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "used_memory_peak": redis_info.get("used_memory_peak", 0),
                    "used_memory_peak_human": redis_info.get("used_memory_peak_human", "0B"),
                    "mem_fragmentation_ratio": redis_info.get("mem_fragmentation_ratio", 0),
                }
            except Exception as e:
                stats["redis_cache"]["error"] = str(e)

        return stats

    @asynccontextmanager
    async def cached_result(
        self, key: str, ttl: int = 300, namespace: str = "", layer: CacheLayer = CacheLayer.HYBRID
    ) -> AsyncGenerator[Any, None]:
        """Context manager for cached function results."""
        # Try to get from cache first
        cached_value = await self.get(key, namespace, layer)
        if cached_value is not None:
            yield cached_value
            return

        # Cache miss - compute and cache result
        class ResultCapture:
            def __init__(self):
                self.value = None

            def __call__(self, value):
                self.value = value
                return value

        capture = ResultCapture()

        try:
            yield capture

            # Cache the result if it was set
            if capture.value is not None:
                await self.set(key, capture.value, ttl, namespace, layer)

        except Exception:
            # Don't cache errors
            raise


# Decorator for caching function results
def cached(
    ttl: int = 300,
    namespace: str = "",
    layer: CacheLayer = CacheLayer.HYBRID,
    key_func: Optional[Callable] = None,
):
    """Decorator for caching function results."""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Generate key from function name and arguments
                arg_str = "_".join(str(arg) for arg in args)
                kwarg_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key_parts = [func.__name__, arg_str, kwarg_str]
                cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()

            # Get cache manager from somewhere (dependency injection in real app)
            # For now, assume it's available globally
            cache_manager = getattr(func, "_cache_manager", None)
            if not cache_manager:
                return await func(*args, **kwargs)

            # Try cache first
            cached_result = await cache_manager.get(cache_key, namespace, layer)
            if cached_result is not None:
                return cached_result

            # Compute result and cache it
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl, namespace, layer)
            return result

        return wrapper

    return decorator
