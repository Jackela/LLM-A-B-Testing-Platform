"""Advanced memory management and object pooling for optimal performance."""

import asyncio
import gc
import threading
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar

import psutil

T = TypeVar("T")


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""

    current_usage_mb: float = 0.0
    peak_usage_mb: float = 0.0
    gc_collections: int = 0
    gc_time_ms: float = 0.0
    object_pool_stats: Dict[str, Any] = field(default_factory=dict)
    memory_leaks_detected: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ObjectPoolStats:
    """Statistics for object pools."""

    pool_name: str
    total_created: int = 0
    total_acquired: int = 0
    total_released: int = 0
    current_pool_size: int = 0
    current_in_use: int = 0
    peak_pool_size: int = 0
    average_lifetime_ms: float = 0.0
    cache_hit_rate: float = 0.0


class ObjectPool(Generic[T]):
    """Generic object pool for reusing expensive objects."""

    def __init__(
        self,
        name: str,
        factory: Callable[[], T],
        reset_func: Optional[Callable[[T], None]] = None,
        validator: Optional[Callable[[T], bool]] = None,
        max_size: int = 100,
        min_size: int = 5,
        max_idle_time: int = 300,  # 5 minutes
        enable_metrics: bool = True,
    ):
        self.name = name
        self.factory = factory
        self.reset_func = reset_func
        self.validator = validator
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.enable_metrics = enable_metrics

        # Pool storage
        self._pool: deque = deque()
        self._in_use: Set[T] = set()
        self._creation_times: Dict[T, float] = {}
        self._last_used_times: Dict[T, float] = {}

        # Thread safety
        self._lock = asyncio.Lock()

        # Statistics
        self.stats = ObjectPoolStats(pool_name=name)

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_running = False

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_running:
            return

        self._cleanup_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        self._cleanup_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Background cleanup of idle objects."""
        while self._cleanup_running:
            try:
                await self._cleanup_idle_objects()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in object pool cleanup: {e}")
                await asyncio.sleep(60)

    async def _cleanup_idle_objects(self) -> None:
        """Remove objects that have been idle too long."""
        async with self._lock:
            current_time = time.time()
            objects_to_remove = []

            # Check pooled objects
            for obj in list(self._pool):
                last_used = self._last_used_times.get(obj, current_time)
                if current_time - last_used > self.max_idle_time:
                    objects_to_remove.append(obj)

            # Remove idle objects
            for obj in objects_to_remove:
                self._pool.remove(obj)
                self._creation_times.pop(obj, None)
                self._last_used_times.pop(obj, None)

            # Ensure minimum pool size
            while len(self._pool) < self.min_size:
                try:
                    obj = self.factory()
                    self._pool.append(obj)
                    self._creation_times[obj] = current_time
                    self._last_used_times[obj] = current_time
                    self.stats.total_created += 1
                except Exception as e:
                    print(f"Failed to create object in pool {self.name}: {e}")
                    break

            # Update stats
            self.stats.current_pool_size = len(self._pool)
            if self.stats.current_pool_size > self.stats.peak_pool_size:
                self.stats.peak_pool_size = self.stats.current_pool_size

    async def acquire(self) -> T:
        """Acquire an object from the pool."""
        async with self._lock:
            current_time = time.time()

            # Try to get from pool
            while self._pool:
                obj = self._pool.popleft()

                # Validate object if validator provided
                if self.validator and not self.validator(obj):
                    self._creation_times.pop(obj, None)
                    self._last_used_times.pop(obj, None)
                    continue

                # Reset object if reset function provided
                if self.reset_func:
                    try:
                        self.reset_func(obj)
                    except Exception as e:
                        print(f"Failed to reset object in pool {self.name}: {e}")
                        self._creation_times.pop(obj, None)
                        self._last_used_times.pop(obj, None)
                        continue

                # Mark as in use
                self._in_use.add(obj)
                self._last_used_times[obj] = current_time

                # Update stats
                self.stats.total_acquired += 1
                self.stats.current_in_use = len(self._in_use)
                self.stats.current_pool_size = len(self._pool)
                if self.enable_metrics:
                    total_attempts = self.stats.total_acquired + len(self._pool)
                    self.stats.cache_hit_rate = self.stats.total_acquired / max(total_attempts, 1)

                return obj

            # Pool is empty, create new object
            try:
                obj = self.factory()
                self._in_use.add(obj)
                self._creation_times[obj] = current_time
                self._last_used_times[obj] = current_time

                # Update stats
                self.stats.total_created += 1
                self.stats.total_acquired += 1
                self.stats.current_in_use = len(self._in_use)

                return obj

            except Exception as e:
                print(f"Failed to create object in pool {self.name}: {e}")
                raise

    async def release(self, obj: T) -> None:
        """Release an object back to the pool."""
        async with self._lock:
            if obj not in self._in_use:
                return  # Object not from this pool or already released

            self._in_use.remove(obj)
            current_time = time.time()

            # Update stats
            self.stats.total_released += 1
            self.stats.current_in_use = len(self._in_use)

            # Calculate lifetime
            if obj in self._creation_times and self.enable_metrics:
                lifetime = (current_time - self._creation_times[obj]) * 1000  # Convert to ms
                # Update running average
                if self.stats.average_lifetime_ms == 0:
                    self.stats.average_lifetime_ms = lifetime
                else:
                    self.stats.average_lifetime_ms = (
                        self.stats.average_lifetime_ms * 0.9 + lifetime * 0.1
                    )

            # Return to pool if there's space
            if len(self._pool) < self.max_size:
                self._pool.append(obj)
                self._last_used_times[obj] = current_time
                self.stats.current_pool_size = len(self._pool)
            else:
                # Pool is full, discard object
                self._creation_times.pop(obj, None)
                self._last_used_times.pop(obj, None)

    @asynccontextmanager
    async def borrow(self) -> AsyncGenerator[T, None]:
        """Context manager for borrowing objects from the pool."""
        obj = await self.acquire()
        try:
            yield obj
        finally:
            await self.release(obj)

    async def clear(self) -> None:
        """Clear the entire pool."""
        async with self._lock:
            self._pool.clear()
            self._in_use.clear()
            self._creation_times.clear()
            self._last_used_times.clear()

            # Reset stats
            self.stats.current_pool_size = 0
            self.stats.current_in_use = 0

    def get_stats(self) -> ObjectPoolStats:
        """Get current pool statistics."""
        return self.stats


class MemoryManager:
    """Advanced memory management with leak detection and optimization."""

    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.metrics = MemoryMetrics()

        # Object pools
        self._object_pools: Dict[str, ObjectPool] = {}

        # Memory monitoring
        self._monitoring_enabled = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = 30  # seconds

        # Garbage collection tracking
        self._gc_stats = {"collections": [0, 0, 0], "total_time": 0.0}  # counts for each generation

        # Memory leak detection
        self._enable_tracemalloc = False
        self._memory_snapshots: deque = deque(maxlen=10)
        self._leak_detection_threshold = 50 * 1024 * 1024  # 50MB increase

        # Weak references to track object lifecycles
        self._tracked_objects: weakref.WeakSet = weakref.WeakSet()

        # Memory optimization settings
        self._gc_optimization_enabled = True
        self._gc_threshold_multiplier = 2.0

    async def initialize(self) -> None:
        """Initialize memory manager."""
        if self.enable_monitoring:
            await self.start_monitoring()

        # Enable tracemalloc for leak detection
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._enable_tracemalloc = True

        # Optimize garbage collection
        if self._gc_optimization_enabled:
            self._optimize_gc_thresholds()

        print("Memory manager initialized")

    async def shutdown(self) -> None:
        """Shutdown memory manager."""
        await self.stop_monitoring()

        # Stop all object pool cleanup tasks
        for pool in self._object_pools.values():
            await pool.stop_cleanup_task()

        # Clear all pools
        for pool in self._object_pools.values():
            await pool.clear()

        # Stop tracemalloc if we started it
        if self._enable_tracemalloc:
            tracemalloc.stop()

        print("Memory manager shutdown complete")

    def create_object_pool(
        self,
        name: str,
        factory: Callable[[], T],
        reset_func: Optional[Callable[[T], None]] = None,
        validator: Optional[Callable[[T], bool]] = None,
        max_size: int = 100,
        min_size: int = 5,
        max_idle_time: int = 300,
    ) -> ObjectPool[T]:
        """Create a new object pool."""
        if name in self._object_pools:
            raise ValueError(f"Object pool '{name}' already exists")

        pool = ObjectPool(
            name=name,
            factory=factory,
            reset_func=reset_func,
            validator=validator,
            max_size=max_size,
            min_size=min_size,
            max_idle_time=max_idle_time,
            enable_metrics=self.enable_monitoring,
        )

        self._object_pools[name] = pool

        # Start cleanup task
        asyncio.create_task(pool.start_cleanup_task())

        return pool

    def get_object_pool(self, name: str) -> Optional[ObjectPool]:
        """Get an existing object pool."""
        return self._object_pools.get(name)

    async def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self._monitoring_enabled:
            return

        self._monitoring_enabled = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        print("Memory monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring_enabled = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        print("Memory monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background memory monitoring loop."""
        while self._monitoring_enabled:
            try:
                await self._collect_memory_metrics()
                await self._detect_memory_leaks()
                await asyncio.sleep(self._monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in memory monitoring: {e}")
                await asyncio.sleep(self._monitoring_interval)

    async def _collect_memory_metrics(self) -> None:
        """Collect current memory metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            # Update metrics
            current_mb = memory_info.rss / (1024 * 1024)
            self.metrics.current_usage_mb = current_mb

            if current_mb > self.metrics.peak_usage_mb:
                self.metrics.peak_usage_mb = current_mb

            # Collect GC stats
            gc_stats = gc.get_stats()
            for i, stats in enumerate(gc_stats):
                self.metrics.gc_collections += stats.get("collections", 0)

            # Update object pool stats
            self.metrics.object_pool_stats = {
                name: pool.get_stats().__dict__ for name, pool in self._object_pools.items()
            }

            self.metrics.last_updated = datetime.utcnow()

        except Exception as e:
            print(f"Error collecting memory metrics: {e}")

    async def _detect_memory_leaks(self) -> None:
        """Detect potential memory leaks."""
        if not self._enable_tracemalloc:
            return

        try:
            current_snapshot = tracemalloc.take_snapshot()
            self._memory_snapshots.append(current_snapshot)

            # Need at least 2 snapshots to compare
            if len(self._memory_snapshots) < 2:
                return

            # Compare with previous snapshot
            previous_snapshot = self._memory_snapshots[-2]
            top_stats = current_snapshot.compare_to(previous_snapshot, "lineno")

            # Check for significant memory increase
            total_increase = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

            if total_increase > self._leak_detection_threshold:
                self.metrics.memory_leaks_detected += 1

                print(f"POTENTIAL MEMORY LEAK DETECTED:")
                print(f"  Memory increase: {total_increase / (1024 * 1024):.2f} MB")
                print(f"  Top allocations:")

                for i, stat in enumerate(top_stats[:5]):
                    if stat.size_diff > 0:
                        print(
                            f"    {i+1}. {stat.traceback.format()[-1]}: "
                            f"+{stat.size_diff / 1024:.1f} KB"
                        )

        except Exception as e:
            print(f"Error in leak detection: {e}")

    def _optimize_gc_thresholds(self) -> None:
        """Optimize garbage collection thresholds for better performance."""
        # Get current thresholds
        threshold0, threshold1, threshold2 = gc.get_threshold()

        # Increase thresholds to reduce GC frequency
        new_threshold0 = int(threshold0 * self._gc_threshold_multiplier)
        new_threshold1 = int(threshold1 * self._gc_threshold_multiplier)
        new_threshold2 = int(threshold2 * self._gc_threshold_multiplier)

        gc.set_threshold(new_threshold0, new_threshold1, new_threshold2)

        print(f"Optimized GC thresholds: {gc.get_threshold()}")

    async def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics."""
        start_time = time.time()

        # Run garbage collection for all generations
        collected = [gc.collect(i) for i in range(3)]

        duration = time.time() - start_time
        self.metrics.gc_time_ms += duration * 1000

        return {
            "objects_collected": sum(collected),
            "by_generation": collected,
            "duration_ms": duration * 1000,
            "current_objects": len(gc.get_objects()),
            "gc_stats": gc.get_stats(),
        }

    def track_object(self, obj: Any) -> None:
        """Track an object for lifecycle monitoring."""
        self._tracked_objects.add(obj)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            stats = {
                "current_usage": {
                    "rss_mb": memory_info.rss / (1024 * 1024),
                    "vms_mb": memory_info.vms / (1024 * 1024),
                    "percent": process.memory_percent(),
                },
                "peak_usage_mb": self.metrics.peak_usage_mb,
                "garbage_collection": {
                    "total_collections": self.metrics.gc_collections,
                    "total_time_ms": self.metrics.gc_time_ms,
                    "current_objects": len(gc.get_objects()),
                    "thresholds": gc.get_threshold(),
                    "stats": gc.get_stats(),
                },
                "object_pools": self.metrics.object_pool_stats,
                "tracked_objects": len(self._tracked_objects),
                "memory_leaks_detected": self.metrics.memory_leaks_detected,
                "last_updated": self.metrics.last_updated.isoformat(),
            }

            # Add tracemalloc stats if available
            if self._enable_tracemalloc:
                current, peak = tracemalloc.get_traced_memory()
                stats["tracemalloc"] = {
                    "current_mb": current / (1024 * 1024),
                    "peak_mb": peak / (1024 * 1024),
                    "snapshots_taken": len(self._memory_snapshots),
                }

            return stats

        except Exception as e:
            return {"error": str(e)}

    @asynccontextmanager
    async def memory_profiling(self, operation_name: str) -> AsyncGenerator[None, None]:
        """Context manager for profiling memory usage of operations."""
        if not self._enable_tracemalloc:
            yield
            return

        # Take snapshot before operation
        snapshot_before = tracemalloc.take_snapshot()
        start_time = time.time()

        try:
            yield
        finally:
            # Take snapshot after operation
            snapshot_after = tracemalloc.take_snapshot()
            duration = time.time() - start_time

            # Compare snapshots
            top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
            total_allocated = sum(stat.size for stat in top_stats if stat.size > 0)

            print(f"Memory Profile - {operation_name}:")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Memory allocated: {total_allocated / 1024:.1f} KB")

            if top_stats:
                print(f"  Top allocations:")
                for i, stat in enumerate(top_stats[:3]):
                    if stat.size > 0:
                        print(
                            f"    {i+1}. {stat.size / 1024:.1f} KB - "
                            f"{stat.traceback.format()[-1]}"
                        )

    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Perform memory optimization operations."""
        optimization_results = {}

        # Force garbage collection
        gc_results = await self.force_garbage_collection()
        optimization_results["garbage_collection"] = gc_results

        # Clear caches in object pools
        cleared_pools = 0
        for name, pool in self._object_pools.items():
            if pool.stats.current_pool_size > pool.min_size * 2:
                # Pool is significantly larger than minimum, clear some objects
                await pool._cleanup_idle_objects()
                cleared_pools += 1

        optimization_results["object_pools_optimized"] = cleared_pools

        # System-level optimizations
        if hasattr(gc, "set_debug"):
            # Disable debug output to reduce memory overhead
            gc.set_debug(0)

        optimization_results["system_optimizations"] = ["gc_debug_disabled"]

        return optimization_results


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


async def init_memory_manager(enable_monitoring: bool = True) -> MemoryManager:
    """Initialize global memory manager."""
    global _memory_manager
    _memory_manager = MemoryManager(enable_monitoring)
    await _memory_manager.initialize()
    return _memory_manager
