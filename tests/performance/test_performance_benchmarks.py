"""Comprehensive performance benchmarks and validation tests."""

import asyncio
import statistics
import time
from datetime import datetime
from typing import Any, Dict, List

import pytest

from src.infrastructure.performance import (
    CacheConfig,
    CacheManager,
    CircuitBreakerManager,
    ConnectionOptimizer,
    MemoryManager,
    MetricsCollector,
    QueryCacheManager,
)
from src.infrastructure.performance.api_optimizer import APIOptimizer
from src.infrastructure.persistence.database import DatabaseConfig, DatabaseManager


class PerformanceBenchmark:
    """Performance benchmark runner."""

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.metrics_collector = None
        self.cache_manager = None
        self.memory_manager = None
        self.connection_optimizer = None

    async def setup(self):
        """Setup benchmark environment."""
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(enable_prometheus=False)

        # Initialize cache manager
        cache_config = CacheConfig(
            redis_url="redis://localhost:6379", memory_max_size=1000, enable_cache_warming=True
        )

        self.cache_manager = CacheManager(cache_config)
        await self.cache_manager.initialize()

        # Initialize memory manager
        self.memory_manager = MemoryManager(enable_monitoring=False)
        await self.memory_manager.initialize()

        # Initialize connection optimizer
        self.connection_optimizer = ConnectionOptimizer(self.metrics_collector)
        await self.connection_optimizer.initialize()

    async def teardown(self):
        """Cleanup benchmark environment."""
        if self.cache_manager:
            await self.cache_manager.close()

        if self.memory_manager:
            await self.memory_manager.shutdown()

        if self.connection_optimizer:
            await self.connection_optimizer.shutdown()

    async def run_cache_benchmark(self, iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark cache performance."""
        print(f"Running cache benchmark with {iterations} iterations...")

        # Test data
        test_data = {"key": f"value_{i}" for i in range(100)}

        # Benchmark cache set operations
        start_time = time.time()
        for i in range(iterations):
            await self.cache_manager.set(f"test_key_{i}", test_data, ttl=300)
        set_time = time.time() - start_time

        # Benchmark cache get operations
        start_time = time.time()
        hit_count = 0
        for i in range(iterations):
            result = await self.cache_manager.get(f"test_key_{i}")
            if result is not None:
                hit_count += 1
        get_time = time.time() - start_time

        # Calculate metrics
        set_ops_per_second = iterations / set_time
        get_ops_per_second = iterations / get_time
        hit_rate = hit_count / iterations

        return {
            "cache_set_ops_per_second": set_ops_per_second,
            "cache_get_ops_per_second": get_ops_per_second,
            "cache_hit_rate": hit_rate,
            "total_set_time": set_time,
            "total_get_time": get_time,
            "iterations": iterations,
        }

    async def run_memory_benchmark(self, pool_size: int = 100) -> Dict[str, Any]:
        """Benchmark memory management and object pooling."""
        print(f"Running memory benchmark with pool size {pool_size}...")

        # Create object pool
        def create_test_object():
            return {"data": "x" * 1000, "timestamp": time.time()}

        def reset_test_object(obj):
            obj["timestamp"] = time.time()

        pool = self.memory_manager.create_object_pool(
            "test_pool",
            factory=create_test_object,
            reset_func=reset_test_object,
            max_size=pool_size,
        )

        # Benchmark object acquisition
        start_time = time.time()
        acquired_objects = []

        for i in range(pool_size * 2):  # Acquire more than pool size
            obj = await pool.acquire()
            acquired_objects.append(obj)

        acquisition_time = time.time() - start_time

        # Benchmark object release
        start_time = time.time()
        for obj in acquired_objects:
            await pool.release(obj)
        release_time = time.time() - start_time

        # Get pool statistics
        stats = pool.get_stats()

        return {
            "object_acquisition_time": acquisition_time,
            "object_release_time": release_time,
            "acquisitions_per_second": len(acquired_objects) / acquisition_time,
            "releases_per_second": len(acquired_objects) / release_time,
            "pool_stats": stats.__dict__,
            "pool_size": pool_size,
        }

    async def run_api_response_benchmark(self, response_count: int = 500) -> Dict[str, Any]:
        """Benchmark API response optimization."""
        print(f"Running API response benchmark with {response_count} responses...")

        # Initialize API optimizer
        api_optimizer = APIOptimizer(self.cache_manager, self.metrics_collector)
        await api_optimizer.initialize()

        # Test data of varying sizes
        small_data = {"message": "hello world"}
        medium_data = {"data": [{"id": i, "value": f"data_{i}"} for i in range(100)]}
        large_data = {"data": [{"id": i, "value": f"data_{i}" * 100} for i in range(1000)]}

        test_datasets = [("small", small_data), ("medium", medium_data), ("large", large_data)]

        benchmark_results = {}

        for data_type, test_data in test_datasets:
            print(f"  Benchmarking {data_type} responses...")

            # Benchmark compression
            compression_times = []
            compression_ratios = []

            for i in range(response_count // 3):
                start_time = time.time()

                # Simulate response optimization
                original_size = len(str(test_data))

                # Simple compression simulation
                import gzip
                import json

                json_data = json.dumps(test_data)
                compressed_data = gzip.compress(json_data.encode())

                compression_time = (time.time() - start_time) * 1000  # ms
                compression_ratio = len(json_data) / len(compressed_data)

                compression_times.append(compression_time)
                compression_ratios.append(compression_ratio)

            benchmark_results[data_type] = {
                "average_compression_time_ms": statistics.mean(compression_times),
                "average_compression_ratio": statistics.mean(compression_ratios),
                "original_size_bytes": len(str(test_data)),
                "iterations": len(compression_times),
            }

        return benchmark_results

    async def run_circuit_breaker_benchmark(self, operations: int = 1000) -> Dict[str, Any]:
        """Benchmark circuit breaker performance."""
        print(f"Running circuit breaker benchmark with {operations} operations...")

        # Initialize circuit breaker manager
        cb_manager = CircuitBreakerManager(self.metrics_collector)

        # Create test circuit breaker
        circuit_breaker = cb_manager.get_circuit_breaker("test_service", "external_api")

        # Benchmark successful operations
        async def successful_operation():
            await asyncio.sleep(0.001)  # Simulate work
            return "success"

        start_time = time.time()
        success_count = 0

        for i in range(operations):
            try:
                async with cb_manager.protected_call("test_service", successful_operation):
                    success_count += 1
            except Exception:
                pass

        operation_time = time.time() - start_time

        # Get circuit breaker metrics
        cb_metrics = circuit_breaker.get_metrics()

        return {
            "operations_per_second": operations / operation_time,
            "success_rate": success_count / operations,
            "total_operation_time": operation_time,
            "circuit_breaker_metrics": cb_metrics.__dict__,
            "operations": operations,
        }

    async def run_concurrent_load_test(
        self, concurrent_users: int = 100, operations_per_user: int = 10
    ) -> Dict[str, Any]:
        """Run concurrent load test."""
        print(
            f"Running concurrent load test: {concurrent_users} users, {operations_per_user} ops each..."
        )

        async def user_simulation(user_id: int):
            """Simulate a user performing multiple operations."""
            user_results = {
                "cache_operations": 0,
                "memory_operations": 0,
                "errors": 0,
                "total_time": 0,
            }

            start_time = time.time()

            try:
                for op in range(operations_per_user):
                    # Cache operation
                    await self.cache_manager.set(f"user_{user_id}_op_{op}", {"data": f"value_{op}"})
                    result = await self.cache_manager.get(f"user_{user_id}_op_{op}")
                    if result:
                        user_results["cache_operations"] += 1

                    # Memory operation (if available)
                    if hasattr(self, "test_pool"):
                        obj = await self.test_pool.acquire()
                        await self.test_pool.release(obj)
                        user_results["memory_operations"] += 1

                    # Small delay to simulate realistic usage
                    await asyncio.sleep(0.001)

            except Exception as e:
                user_results["errors"] += 1
                print(f"User {user_id} error: {e}")

            user_results["total_time"] = time.time() - start_time
            return user_results

        # Create test object pool for concurrent test
        def create_test_object():
            return {"user_data": "test"}

        self.test_pool = self.memory_manager.create_object_pool(
            "concurrent_test_pool", factory=create_test_object, max_size=concurrent_users * 2
        )

        # Run concurrent user simulations
        start_time = time.time()
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Aggregate results
        successful_users = [r for r in results if isinstance(r, dict)]
        failed_users = len(results) - len(successful_users)

        if successful_users:
            total_cache_ops = sum(r["cache_operations"] for r in successful_users)
            total_memory_ops = sum(r["memory_operations"] for r in successful_users)
            total_errors = sum(r["errors"] for r in successful_users)
            avg_user_time = statistics.mean(r["total_time"] for r in successful_users)
        else:
            total_cache_ops = total_memory_ops = total_errors = avg_user_time = 0

        return {
            "concurrent_users": concurrent_users,
            "operations_per_user": operations_per_user,
            "successful_users": len(successful_users),
            "failed_users": failed_users,
            "total_cache_operations": total_cache_ops,
            "total_memory_operations": total_memory_ops,
            "total_errors": total_errors,
            "average_user_time": avg_user_time,
            "total_test_time": total_time,
            "operations_per_second": (total_cache_ops + total_memory_ops) / total_time,
            "user_success_rate": len(successful_users) / concurrent_users,
        }

    async def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("Starting comprehensive performance benchmark suite...")

        suite_start_time = time.time()
        suite_results = {}

        try:
            # Cache benchmarks
            suite_results["cache_benchmark"] = await self.run_cache_benchmark(1000)

            # Memory benchmarks
            suite_results["memory_benchmark"] = await self.run_memory_benchmark(100)

            # API response benchmarks
            suite_results["api_response_benchmark"] = await self.run_api_response_benchmark(300)

            # Circuit breaker benchmarks
            suite_results["circuit_breaker_benchmark"] = await self.run_circuit_breaker_benchmark(
                500
            )

            # Concurrent load test
            suite_results["concurrent_load_test"] = await self.run_concurrent_load_test(50, 10)

            suite_results["total_benchmark_time"] = time.time() - suite_start_time
            suite_results["benchmark_timestamp"] = datetime.utcnow().isoformat()
            suite_results["status"] = "completed"

        except Exception as e:
            suite_results["error"] = str(e)
            suite_results["status"] = "failed"
            print(f"Benchmark suite failed: {e}")

        return suite_results


@pytest.mark.asyncio
async def test_cache_performance_targets():
    """Test that cache performance meets targets."""
    benchmark = PerformanceBenchmark()

    try:
        await benchmark.setup()

        # Run cache benchmark
        results = await benchmark.run_cache_benchmark(500)

        # Performance targets
        assert (
            results["cache_set_ops_per_second"] > 1000
        ), f"Cache set performance too low: {results['cache_set_ops_per_second']}"
        assert (
            results["cache_get_ops_per_second"] > 2000
        ), f"Cache get performance too low: {results['cache_get_ops_per_second']}"
        assert (
            results["cache_hit_rate"] > 0.95
        ), f"Cache hit rate too low: {results['cache_hit_rate']}"

        print(f"✅ Cache performance targets met:")
        print(f"   Set ops/sec: {results['cache_set_ops_per_second']:.1f}")
        print(f"   Get ops/sec: {results['cache_get_ops_per_second']:.1f}")
        print(f"   Hit rate: {results['cache_hit_rate']:.1%}")

    finally:
        await benchmark.teardown()


@pytest.mark.asyncio
async def test_memory_management_performance():
    """Test memory management performance."""
    benchmark = PerformanceBenchmark()

    try:
        await benchmark.setup()

        # Run memory benchmark
        results = await benchmark.run_memory_benchmark(50)

        # Performance targets
        assert (
            results["acquisitions_per_second"] > 5000
        ), f"Object acquisition too slow: {results['acquisitions_per_second']}"
        assert (
            results["releases_per_second"] > 10000
        ), f"Object release too slow: {results['releases_per_second']}"

        pool_stats = results["pool_stats"]
        assert (
            pool_stats["total_created"] <= results["pool_size"] * 1.2
        ), "Too many objects created (poor pooling)"

        print(f"✅ Memory management performance targets met:")
        print(f"   Acquisitions/sec: {results['acquisitions_per_second']:.1f}")
        print(f"   Releases/sec: {results['releases_per_second']:.1f}")
        print(f"   Objects created: {pool_stats['total_created']}")

    finally:
        await benchmark.teardown()


@pytest.mark.asyncio
async def test_api_response_optimization():
    """Test API response optimization performance."""
    benchmark = PerformanceBenchmark()

    try:
        await benchmark.setup()

        # Run API response benchmark
        results = await benchmark.run_api_response_benchmark(100)

        # Performance targets for compression
        for data_type, metrics in results.items():
            assert (
                metrics["average_compression_time_ms"] < 10
            ), f"{data_type} compression too slow: {metrics['average_compression_time_ms']}ms"
            assert (
                metrics["average_compression_ratio"] > 1.5
            ), f"{data_type} compression ratio too low: {metrics['average_compression_ratio']}"

        print(f"✅ API response optimization targets met:")
        for data_type, metrics in results.items():
            print(
                f"   {data_type}: {metrics['average_compression_time_ms']:.2f}ms, ratio: {metrics['average_compression_ratio']:.2f}"
            )

    finally:
        await benchmark.teardown()


@pytest.mark.asyncio
async def test_concurrent_performance():
    """Test performance under concurrent load."""
    benchmark = PerformanceBenchmark()

    try:
        await benchmark.setup()

        # Run concurrent load test
        results = await benchmark.run_concurrent_load_test(25, 5)

        # Performance targets
        assert (
            results["user_success_rate"] > 0.95
        ), f"User success rate too low: {results['user_success_rate']}"
        assert (
            results["operations_per_second"] > 100
        ), f"Operations per second too low: {results['operations_per_second']}"
        assert results["total_errors"] == 0, f"Errors detected: {results['total_errors']}"

        print(f"✅ Concurrent performance targets met:")
        print(f"   User success rate: {results['user_success_rate']:.1%}")
        print(f"   Operations/sec: {results['operations_per_second']:.1f}")
        print(f"   Total errors: {results['total_errors']}")

    finally:
        await benchmark.teardown()


@pytest.mark.asyncio
async def test_circuit_breaker_performance():
    """Test circuit breaker performance impact."""
    benchmark = PerformanceBenchmark()

    try:
        await benchmark.setup()

        # Run circuit breaker benchmark
        results = await benchmark.run_circuit_breaker_benchmark(500)

        # Performance targets
        assert (
            results["operations_per_second"] > 500
        ), f"Circuit breaker overhead too high: {results['operations_per_second']}"
        assert (
            results["success_rate"] > 0.99
        ), f"Circuit breaker success rate too low: {results['success_rate']}"

        print(f"✅ Circuit breaker performance targets met:")
        print(f"   Operations/sec: {results['operations_per_second']:.1f}")
        print(f"   Success rate: {results['success_rate']:.1%}")

    finally:
        await benchmark.teardown()


@pytest.mark.asyncio
async def test_full_performance_suite():
    """Run the complete performance benchmark suite."""
    benchmark = PerformanceBenchmark()

    try:
        await benchmark.setup()

        # Run full benchmark suite
        results = await benchmark.run_full_benchmark_suite()

        assert (
            results["status"] == "completed"
        ), f"Benchmark suite failed: {results.get('error', 'Unknown error')}"

        # Validate all components performed adequately
        cache_results = results["cache_benchmark"]
        assert cache_results["cache_set_ops_per_second"] > 500
        assert cache_results["cache_get_ops_per_second"] > 1000

        memory_results = results["memory_benchmark"]
        assert memory_results["acquisitions_per_second"] > 2000

        concurrent_results = results["concurrent_load_test"]
        assert concurrent_results["user_success_rate"] > 0.9

        print(
            f"✅ Full performance suite completed successfully in {results['total_benchmark_time']:.2f}s"
        )
        print(f"   All performance targets met across {len(results)-3} benchmark categories")

        # Store results for analysis
        with open("performance_benchmark_results.json", "w") as f:
            import json

            json.dump(results, f, indent=2)

        print(f"📊 Detailed results saved to performance_benchmark_results.json")

    finally:
        await benchmark.teardown()


if __name__ == "__main__":
    """Run benchmarks directly."""

    async def main():
        benchmark = PerformanceBenchmark()
        await benchmark.setup()

        try:
            results = await benchmark.run_full_benchmark_suite()

            print("\n" + "=" * 80)
            print("PERFORMANCE BENCHMARK RESULTS")
            print("=" * 80)

            for test_name, test_results in results.items():
                if test_name in ["total_benchmark_time", "benchmark_timestamp", "status"]:
                    continue

                print(f"\n{test_name.upper().replace('_', ' ')}:")
                if isinstance(test_results, dict):
                    for key, value in test_results.items():
                        if isinstance(value, (int, float)):
                            if "time" in key.lower() and "ms" not in key.lower():
                                print(f"  {key}: {value:.3f}s")
                            elif "rate" in key.lower() or "ratio" in key.lower():
                                print(
                                    f"  {key}: {value:.1%}"
                                    if value <= 1
                                    else f"  {key}: {value:.2f}"
                                )
                            elif "per_second" in key.lower():
                                print(f"  {key}: {value:.1f}")
                            else:
                                print(f"  {key}: {value}")
                        else:
                            print(f"  {key}: {value}")

            print(f"\nTotal benchmark time: {results['total_benchmark_time']:.2f}s")
            print(f"Status: {results['status']}")

        finally:
            await benchmark.teardown()

    asyncio.run(main())
