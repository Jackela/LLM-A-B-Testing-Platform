"""Benchmark validation and performance regression tests."""

import asyncio
import time
from datetime import datetime
from statistics import mean, median, stdev
from typing import Any, Dict, List

import pytest

from tests.factories import (
    EvaluationResultFactory,
    ModelResponseFactory,
    PerformanceTestDataFactory,
    TestFactory,
)


@pytest.mark.performance
class TestBenchmarkValidation:
    """Benchmark validation and performance regression tests."""

    @pytest.fixture
    def benchmark_thresholds(self):
        """Define performance benchmark thresholds."""
        return {
            "api_response_time": {
                "excellent": 0.1,  # < 100ms
                "good": 0.5,  # < 500ms
                "acceptable": 1.0,  # < 1s
                "poor": 2.0,  # < 2s
            },
            "database_query_time": {
                "excellent": 0.01,  # < 10ms
                "good": 0.05,  # < 50ms
                "acceptable": 0.1,  # < 100ms
                "poor": 0.5,  # < 500ms
            },
            "throughput": {
                "excellent": 1000,  # > 1000 req/s
                "good": 500,  # > 500 req/s
                "acceptable": 100,  # > 100 req/s
                "poor": 50,  # > 50 req/s
            },
            "memory_usage": {
                "excellent": 100,  # < 100MB
                "good": 250,  # < 250MB
                "acceptable": 500,  # < 500MB
                "poor": 1000,  # < 1GB
            },
        }

    def classify_performance(self, value: float, thresholds: Dict[str, float]) -> str:
        """Classify performance based on thresholds."""
        if value <= thresholds["excellent"]:
            return "excellent"
        elif value <= thresholds["good"]:
            return "good"
        elif value <= thresholds["acceptable"]:
            return "acceptable"
        elif value <= thresholds["poor"]:
            return "poor"
        else:
            return "unacceptable"

    @pytest.mark.asyncio
    async def test_api_endpoint_benchmarks(self, async_client, auth_headers, benchmark_thresholds):
        """Benchmark all API endpoints for performance."""
        endpoints = [
            ("GET", "/api/v1/tests/"),
            ("GET", "/api/v1/providers/"),
            ("GET", "/health"),
            ("GET", "/api/v1/info"),
        ]

        benchmark_results = {}

        for method, endpoint in endpoints:
            response_times = []

            # Warm up
            for _ in range(3):
                if method == "GET":
                    await async_client.get(endpoint, headers=auth_headers)

            # Benchmark
            for _ in range(20):
                start_time = time.perf_counter()

                if method == "GET":
                    response = await async_client.get(endpoint, headers=auth_headers)

                end_time = time.perf_counter()
                response_time = end_time - start_time

                if response.status_code in [200, 404]:  # 404 is acceptable for some endpoints
                    response_times.append(response_time)

            if response_times:
                avg_time = mean(response_times)
                median_time = median(response_times)
                std_time = stdev(response_times) if len(response_times) > 1 else 0
                min_time = min(response_times)
                max_time = max(response_times)

                performance_class = self.classify_performance(
                    avg_time, benchmark_thresholds["api_response_time"]
                )

                benchmark_results[f"{method} {endpoint}"] = {
                    "avg_time": avg_time,
                    "median_time": median_time,
                    "std_time": std_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "performance_class": performance_class,
                    "samples": len(response_times),
                }

        # Print benchmark results
        print(f"\nAPI Endpoint Benchmarks:")
        print(
            f"{'Endpoint':<30} {'Avg (ms)':<10} {'Med (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Class':<12}"
        )
        print("-" * 90)

        for endpoint, stats in benchmark_results.items():
            print(
                f"{endpoint:<30} {stats['avg_time']*1000:<10.2f} {stats['median_time']*1000:<10.2f} "
                f"{stats['min_time']*1000:<10.2f} {stats['max_time']*1000:<10.2f} {stats['performance_class']:<12}"
            )

        # Validate benchmarks
        for endpoint, stats in benchmark_results.items():
            assert (
                stats["performance_class"] != "unacceptable"
            ), f"Endpoint {endpoint} has unacceptable performance: {stats['avg_time']*1000:.2f}ms"

    @pytest.mark.asyncio
    async def test_database_operation_benchmarks(
        self, async_session, test_repository, benchmark_thresholds
    ):
        """Benchmark database operations."""
        from tests.factories import TestFactory

        # Prepare test data
        tests = [TestFactory() for _ in range(100)]
        await test_repository.bulk_save(tests)
        await async_session.commit()

        benchmark_operations = {
            "single_insert": lambda: test_repository.save(TestFactory()),
            "single_select": lambda: test_repository.get_by_id(tests[0].id),
            "paginated_select": lambda: test_repository.get_paginated(page=1, page_size=20),
            "filtered_select": lambda: test_repository.get_by_status("CONFIGURED"),
            "count_query": lambda: test_repository.count_by_status("CONFIGURED"),
        }

        benchmark_results = {}

        for operation_name, operation_func in benchmark_operations.items():
            response_times = []

            # Warm up
            for _ in range(3):
                await operation_func()

            # Benchmark
            for _ in range(50):
                start_time = time.perf_counter()
                result = await operation_func()
                end_time = time.perf_counter()

                response_time = end_time - start_time
                response_times.append(response_time)

                # Verify operation succeeded
                assert result is not None or operation_name == "count_query"

            avg_time = mean(response_times)
            median_time = median(response_times)
            std_time = stdev(response_times) if len(response_times) > 1 else 0
            min_time = min(response_times)
            max_time = max(response_times)

            performance_class = self.classify_performance(
                avg_time, benchmark_thresholds["database_query_time"]
            )

            benchmark_results[operation_name] = {
                "avg_time": avg_time,
                "median_time": median_time,
                "std_time": std_time,
                "min_time": min_time,
                "max_time": max_time,
                "performance_class": performance_class,
                "samples": len(response_times),
            }

        # Print database benchmark results
        print(f"\nDatabase Operation Benchmarks:")
        print(
            f"{'Operation':<20} {'Avg (ms)':<10} {'Med (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Class':<12}"
        )
        print("-" * 80)

        for operation, stats in benchmark_results.items():
            print(
                f"{operation:<20} {stats['avg_time']*1000:<10.2f} {stats['median_time']*1000:<10.2f} "
                f"{stats['min_time']*1000:<10.2f} {stats['max_time']*1000:<10.2f} {stats['performance_class']:<12}"
            )

        # Validate database benchmarks
        for operation, stats in benchmark_results.items():
            assert (
                stats["performance_class"] != "unacceptable"
            ), f"Database operation {operation} has unacceptable performance: {stats['avg_time']*1000:.2f}ms"

    @pytest.mark.asyncio
    async def test_throughput_benchmarks(self, async_client, auth_headers, benchmark_thresholds):
        """Benchmark system throughput under various loads."""
        test_scenarios = [
            {"concurrent_users": 1, "duration": 10},
            {"concurrent_users": 5, "duration": 10},
            {"concurrent_users": 10, "duration": 10},
            {"concurrent_users": 20, "duration": 10},
        ]

        throughput_results = []

        for scenario in test_scenarios:
            concurrent_users = scenario["concurrent_users"]
            duration = scenario["duration"]

            results = []
            start_time = time.time()

            async def user_session():
                """Simulate user session."""
                session_requests = 0
                session_start = time.time()

                while (time.time() - session_start) < duration:
                    try:
                        response = await async_client.get("/api/v1/tests/", headers=auth_headers)
                        if response.status_code == 200:
                            session_requests += 1
                    except Exception:
                        pass  # Count errors but continue

                    await asyncio.sleep(0.1)  # 100ms between requests

                return session_requests

            # Run concurrent user sessions
            tasks = [user_session() for _ in range(concurrent_users)]
            session_results = await asyncio.gather(*tasks)

            total_time = time.time() - start_time
            total_requests = sum(session_results)
            throughput = total_requests / total_time

            performance_class = self.classify_performance(
                throughput, benchmark_thresholds["throughput"]
            )

            throughput_results.append(
                {
                    "concurrent_users": concurrent_users,
                    "duration": duration,
                    "total_requests": total_requests,
                    "throughput": throughput,
                    "performance_class": performance_class,
                }
            )

        # Print throughput results
        print(f"\nThroughput Benchmarks:")
        print(f"{'Users':<6} {'Duration':<10} {'Requests':<10} {'Throughput':<12} {'Class':<12}")
        print("-" * 60)

        for result in throughput_results:
            print(
                f"{result['concurrent_users']:<6} {result['duration']:<10} {result['total_requests']:<10} "
                f"{result['throughput']:<12.2f} {result['performance_class']:<12}"
            )

        # Validate throughput benchmarks
        for result in throughput_results:
            if result["concurrent_users"] <= 10:  # Be lenient with higher loads
                assert (
                    result["performance_class"] != "unacceptable"
                ), f"Throughput with {result['concurrent_users']} users is unacceptable: {result['throughput']:.2f} req/s"

    @pytest.mark.asyncio
    async def test_memory_benchmark(self, async_client, auth_headers, benchmark_thresholds):
        """Benchmark memory usage patterns."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Baseline memory measurement
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_measurements = []

        # Test memory usage under different loads
        test_loads = [
            {"operations": 10, "description": "Light load"},
            {"operations": 50, "description": "Medium load"},
            {"operations": 100, "description": "Heavy load"},
        ]

        for load in test_loads:
            operations = load["operations"]

            # Measure memory before load
            before_memory = process.memory_info().rss / 1024 / 1024

            # Apply load
            tasks = []
            for i in range(operations):
                task = async_client.get("/api/v1/tests/", headers=auth_headers)
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            successful_responses = [
                r for r in responses if not isinstance(r, Exception) and r.status_code == 200
            ]

            # Measure memory after load
            after_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = after_memory - before_memory

            # Force garbage collection
            import gc

            gc.collect()

            # Measure memory after GC
            gc_memory = process.memory_info().rss / 1024 / 1024

            performance_class = self.classify_performance(
                after_memory, benchmark_thresholds["memory_usage"]
            )

            memory_measurements.append(
                {
                    "description": load["description"],
                    "operations": operations,
                    "successful_ops": len(successful_responses),
                    "before_memory": before_memory,
                    "after_memory": after_memory,
                    "gc_memory": gc_memory,
                    "memory_increase": memory_increase,
                    "performance_class": performance_class,
                }
            )

        # Print memory benchmark results
        print(f"\nMemory Usage Benchmarks:")
        print(
            f"{'Load':<12} {'Ops':<5} {'Before':<8} {'After':<8} {'GC':<8} {'Increase':<10} {'Class':<12}"
        )
        print("-" * 70)

        for measurement in memory_measurements:
            print(
                f"{measurement['description']:<12} {measurement['operations']:<5} "
                f"{measurement['before_memory']:<8.1f} {measurement['after_memory']:<8.1f} "
                f"{measurement['gc_memory']:<8.1f} {measurement['memory_increase']:<10.1f} "
                f"{measurement['performance_class']:<12}"
            )

        # Validate memory benchmarks
        for measurement in memory_measurements:
            assert (
                measurement["performance_class"] != "unacceptable"
            ), f"Memory usage under {measurement['description']} is unacceptable: {measurement['after_memory']:.1f}MB"

            # Memory should not increase excessively
            assert (
                measurement["memory_increase"] < 100
            ), f"Memory increase too high: {measurement['memory_increase']:.1f}MB"

    @pytest.mark.asyncio
    async def test_latency_percentiles(self, async_client, auth_headers):
        """Test latency percentiles for SLA validation."""
        response_times = []

        # Collect response times
        for _ in range(100):
            start_time = time.perf_counter()
            response = await async_client.get("/api/v1/tests/", headers=auth_headers)
            end_time = time.perf_counter()

            if response.status_code == 200:
                response_times.append((end_time - start_time) * 1000)  # Convert to ms

        if not response_times:
            pytest.skip("No successful responses to analyze")

        # Calculate percentiles
        response_times.sort()

        percentiles = {
            "p50": response_times[int(0.50 * len(response_times))],
            "p90": response_times[int(0.90 * len(response_times))],
            "p95": response_times[int(0.95 * len(response_times))],
            "p99": response_times[int(0.99 * len(response_times))],
            "p99.9": (
                response_times[int(0.999 * len(response_times))]
                if len(response_times) >= 1000
                else response_times[-1]
            ),
        }

        avg_latency = mean(response_times)
        max_latency = max(response_times)
        min_latency = min(response_times)

        print(f"\nLatency Percentiles:")
        print(f"Samples: {len(response_times)}")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"Minimum: {min_latency:.2f}ms")
        print(f"Maximum: {max_latency:.2f}ms")
        print(f"P50 (median): {percentiles['p50']:.2f}ms")
        print(f"P90: {percentiles['p90']:.2f}ms")
        print(f"P95: {percentiles['p95']:.2f}ms")
        print(f"P99: {percentiles['p99']:.2f}ms")
        print(f"P99.9: {percentiles['p99.9']:.2f}ms")

        # SLA validation (typical web service SLAs)
        assert percentiles["p50"] < 200, f"P50 latency {percentiles['p50']:.2f}ms exceeds 200ms SLA"
        assert percentiles["p90"] < 500, f"P90 latency {percentiles['p90']:.2f}ms exceeds 500ms SLA"
        assert (
            percentiles["p95"] < 1000
        ), f"P95 latency {percentiles['p95']:.2f}ms exceeds 1000ms SLA"
        assert (
            percentiles["p99"] < 2000
        ), f"P99 latency {percentiles['p99']:.2f}ms exceeds 2000ms SLA"

    @pytest.mark.asyncio
    async def test_scalability_benchmarks(self, async_client, auth_headers):
        """Test system scalability characteristics."""
        user_loads = [1, 2, 4, 8, 16]
        scalability_results = []

        for user_count in user_loads:

            async def user_load():
                """Single user load test."""
                request_count = 0
                start_time = time.time()
                test_duration = 5  # 5 seconds per load test

                while (time.time() - start_time) < test_duration:
                    try:
                        response = await async_client.get("/api/v1/tests/", headers=auth_headers)
                        if response.status_code == 200:
                            request_count += 1
                    except Exception:
                        pass

                    await asyncio.sleep(0.05)  # 50ms between requests

                return request_count

            # Run load test
            tasks = [user_load() for _ in range(user_count)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            total_requests = sum(results)
            throughput = total_requests / total_time
            avg_requests_per_user = total_requests / user_count

            scalability_results.append(
                {
                    "users": user_count,
                    "total_requests": total_requests,
                    "throughput": throughput,
                    "avg_per_user": avg_requests_per_user,
                    "efficiency": throughput / user_count,  # Throughput per user
                }
            )

        # Print scalability results
        print(f"\nScalability Benchmarks:")
        print(
            f"{'Users':<6} {'Requests':<10} {'Throughput':<12} {'Per User':<10} {'Efficiency':<12}"
        )
        print("-" * 60)

        for result in scalability_results:
            print(
                f"{result['users']:<6} {result['total_requests']:<10} {result['throughput']:<12.2f} "
                f"{result['avg_per_user']:<10.2f} {result['efficiency']:<12.2f}"
            )

        # Analyze scalability
        # System should maintain reasonable efficiency as load increases
        base_efficiency = scalability_results[0]["efficiency"]

        for i, result in enumerate(scalability_results[1:], 1):
            efficiency_ratio = result["efficiency"] / base_efficiency

            # Allow for some degradation but not complete breakdown
            if result["users"] <= 8:
                assert (
                    efficiency_ratio > 0.5
                ), f"Efficiency degradation too severe at {result['users']} users: {efficiency_ratio:.2%} of baseline"

    def test_performance_regression_detection(self, benchmark_thresholds):
        """Test performance regression detection mechanisms."""
        # This test would typically compare current benchmarks against historical data
        # For this implementation, we'll demonstrate the concept

        # Simulated historical benchmarks
        historical_benchmarks = {
            "api_response_time": 0.150,  # 150ms
            "database_query_time": 0.020,  # 20ms
            "throughput": 800,  # 800 req/s
            "memory_usage": 120,  # 120MB
        }

        # Simulated current benchmarks (slightly worse)
        current_benchmarks = {
            "api_response_time": 0.180,  # 180ms (+20%)
            "database_query_time": 0.025,  # 25ms (+25%)
            "throughput": 720,  # 720 req/s (-10%)
            "memory_usage": 140,  # 140MB (+17%)
        }

        regression_threshold = 0.20  # 20% degradation threshold
        regressions_detected = []

        for metric, current_value in current_benchmarks.items():
            historical_value = historical_benchmarks[metric]

            if metric == "throughput":
                # For throughput, lower is worse
                degradation = (historical_value - current_value) / historical_value
            else:
                # For other metrics, higher is worse
                degradation = (current_value - historical_value) / historical_value

            if degradation > regression_threshold:
                regressions_detected.append(
                    {
                        "metric": metric,
                        "historical": historical_value,
                        "current": current_value,
                        "degradation": degradation,
                    }
                )

        print(f"\nPerformance Regression Analysis:")
        print(f"Regression Threshold: {regression_threshold:.1%}")

        if regressions_detected:
            print("Regressions Detected:")
            for regression in regressions_detected:
                print(
                    f"  {regression['metric']}: {regression['degradation']:.1%} degradation "
                    f"({regression['historical']} -> {regression['current']})"
                )
        else:
            print("No significant regressions detected")

        # In a real implementation, this would fail the test if regressions are detected
        # For this demo, we'll just log the results
        assert (
            len(regressions_detected) == 0
        ), f"Performance regressions detected: {regressions_detected}"
