"""Comprehensive performance testing framework for the LLM A/B Testing Platform."""

import asyncio
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import httpx
import psutil

from .metrics_collector import MetricsCollector
from .performance_manager import get_performance_manager


class TestResult(NamedTuple):
    """Result of a single performance test."""

    name: str
    success: bool
    duration_ms: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""

    concurrent_users: int = 100
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    requests_per_user: Optional[int] = None
    target_rps: Optional[int] = None
    think_time_ms: int = 0
    failure_threshold: float = 0.05  # 5% failure rate threshold
    response_time_threshold_ms: int = 2000


@dataclass
class PerformanceTestResult:
    """Comprehensive performance test results."""

    test_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    requests_per_second: float
    failure_rate: float
    throughput_mb_per_second: float

    # System metrics
    peak_cpu_usage: float = 0.0
    peak_memory_usage_mb: float = 0.0
    peak_memory_percentage: float = 0.0

    # Application metrics
    cache_hit_rate: float = 0.0
    database_query_count: int = 0
    database_slow_queries: int = 0

    # Errors
    errors: List[str] = field(default_factory=list)

    def passed(self, config: LoadTestConfig) -> bool:
        """Check if test passed based on configuration thresholds."""
        return (
            self.failure_rate <= config.failure_threshold
            and self.p95_response_time_ms <= config.response_time_threshold_ms
        )


class PerformanceBenchmark:
    """Performance benchmark runner with comprehensive metrics."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.metrics_collector = None
        self.results: List[TestResult] = []
        self.system_metrics: List[Dict[str, float]] = []
        self._monitoring_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize performance benchmark."""
        # Get performance manager
        perf_manager = get_performance_manager()
        if perf_manager:
            self.metrics_collector = perf_manager.metrics_collector

    @asynccontextmanager
    async def system_monitoring(self, interval_seconds: float = 1.0):
        """Context manager for system resource monitoring."""
        self._monitoring_task = asyncio.create_task(
            self._monitor_system_resources(interval_seconds)
        )

        try:
            yield
        finally:
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

    async def _monitor_system_resources(self, interval: float) -> None:
        """Monitor system resources during testing."""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)

                # Memory usage
                memory = psutil.virtual_memory()

                # Network I/O
                network = psutil.net_io_counters()

                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()

                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_mb": memory.used / 1024 / 1024,
                    "memory_available_mb": memory.available / 1024 / 1024,
                    "process_memory_mb": process_memory.rss / 1024 / 1024,
                    "network_bytes_sent": network.bytes_sent,
                    "network_bytes_recv": network.bytes_recv,
                }

                self.system_metrics.append(metrics)
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error monitoring system resources: {e}")
                await asyncio.sleep(interval)

    async def run_api_load_test(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[LoadTestConfig] = None,
    ) -> PerformanceTestResult:
        """Run comprehensive API load test."""
        if config is None:
            config = LoadTestConfig()

        print(f"🚀 Starting load test: {method} {endpoint}")
        print(
            f"📊 Configuration: {config.concurrent_users} users, {config.duration_seconds}s duration"
        )

        # Reset metrics
        self.results.clear()
        self.system_metrics.clear()

        start_time = datetime.utcnow()

        async with self.system_monitoring():
            # Run load test
            await self._execute_load_test(endpoint, method, payload, headers, config)

        end_time = datetime.utcnow()

        # Analyze results
        return await self._analyze_results(
            test_name=f"{method} {endpoint}",
            start_time=start_time,
            end_time=end_time,
            config=config,
        )

    async def _execute_load_test(
        self,
        endpoint: str,
        method: str,
        payload: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        config: LoadTestConfig,
    ) -> None:
        """Execute the actual load test."""

        # Calculate request distribution
        total_duration = config.duration_seconds
        ramp_up_duration = min(config.ramp_up_seconds, total_duration)
        sustained_duration = total_duration - ramp_up_duration

        # Create HTTP client with optimized settings
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(
                max_keepalive_connections=config.concurrent_users,
                max_connections=config.concurrent_users * 2,
            ),
        ) as client:

            # Ramp-up phase
            if ramp_up_duration > 0:
                await self._ramp_up_phase(
                    client, endpoint, method, payload, headers, config, ramp_up_duration
                )

            # Sustained load phase
            if sustained_duration > 0:
                await self._sustained_load_phase(
                    client, endpoint, method, payload, headers, config, sustained_duration
                )

    async def _ramp_up_phase(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        method: str,
        payload: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        config: LoadTestConfig,
        duration: int,
    ) -> None:
        """Execute ramp-up phase with gradually increasing load."""
        print(f"📈 Ramp-up phase: {duration}s")

        tasks = []

        for i in range(config.concurrent_users):
            # Stagger user start times during ramp-up
            delay = (i / config.concurrent_users) * duration

            task = asyncio.create_task(
                self._user_session(
                    client,
                    endpoint,
                    method,
                    payload,
                    headers,
                    config,
                    delay,
                    duration + (config.duration_seconds - duration),
                )
            )
            tasks.append(task)

        # Wait for ramp-up to complete
        await asyncio.sleep(duration)

    async def _sustained_load_phase(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        method: str,
        payload: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        config: LoadTestConfig,
        duration: int,
    ) -> None:
        """Execute sustained load phase."""
        print(f"⚖️ Sustained load phase: {duration}s")

        # All users should already be running from ramp-up
        # Just wait for sustained phase to complete
        await asyncio.sleep(duration)

    async def _user_session(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        method: str,
        payload: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        config: LoadTestConfig,
        initial_delay: float,
        total_duration: float,
    ) -> None:
        """Simulate a single user session."""

        # Wait for initial delay
        await asyncio.sleep(initial_delay)

        session_start = time.time()
        request_count = 0

        while (time.time() - session_start) < total_duration:
            try:
                # Make request
                result = await self._make_single_request(client, endpoint, method, payload, headers)
                self.results.append(result)

                request_count += 1

                # Check if we've hit the requests per user limit
                if config.requests_per_user and request_count >= config.requests_per_user:
                    break

                # Think time between requests
                if config.think_time_ms > 0:
                    await asyncio.sleep(config.think_time_ms / 1000.0)

            except Exception as e:
                # Record failed request
                error_result = TestResult(
                    name=f"{method} {endpoint}",
                    success=False,
                    duration_ms=0.0,
                    timestamp=datetime.utcnow(),
                    metadata={"error": str(e)},
                )
                self.results.append(error_result)

    async def _make_single_request(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        method: str,
        payload: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
    ) -> TestResult:
        """Make a single HTTP request and measure performance."""

        url = f"{self.base_url.rstrip('/')}{endpoint}"
        start_time = time.time()

        try:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(url, json=payload, headers=headers)
            elif method.upper() == "PUT":
                response = await client.put(url, json=payload, headers=headers)
            elif method.upper() == "DELETE":
                response = await client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            duration_ms = (time.time() - start_time) * 1000

            # Check response
            success = 200 <= response.status_code < 400

            metadata = {
                "status_code": response.status_code,
                "response_size": len(response.content),
                "content_type": response.headers.get("content-type", ""),
            }

            return TestResult(
                name=f"{method} {endpoint}",
                success=success,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                metadata=metadata,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return TestResult(
                name=f"{method} {endpoint}",
                success=False,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                metadata={"error": str(e)},
            )

    async def _analyze_results(
        self, test_name: str, start_time: datetime, end_time: datetime, config: LoadTestConfig
    ) -> PerformanceTestResult:
        """Analyze test results and generate comprehensive report."""

        if not self.results:
            return PerformanceTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time_ms=0.0,
                p50_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                max_response_time_ms=0.0,
                min_response_time_ms=0.0,
                requests_per_second=0.0,
                failure_rate=0.0,
                throughput_mb_per_second=0.0,
                errors=["No requests completed"],
            )

        # Basic metrics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests

        # Response time analysis
        successful_durations = [r.duration_ms for r in self.results if r.success]
        all_durations = [r.duration_ms for r in self.results]

        if successful_durations:
            avg_response_time = statistics.mean(successful_durations)
            p50_response_time = statistics.median(successful_durations)
            p95_response_time = self._percentile(successful_durations, 95)
            p99_response_time = self._percentile(successful_durations, 99)
            max_response_time = max(successful_durations)
            min_response_time = min(successful_durations)
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0.0
            max_response_time = min_response_time = 0.0

        # Throughput calculation
        duration_seconds = (end_time - start_time).total_seconds()
        requests_per_second = total_requests / duration_seconds if duration_seconds > 0 else 0

        # Calculate throughput in MB/s
        total_response_size = sum(
            r.metadata.get("response_size", 0) for r in self.results if r.success
        )
        throughput_mb_per_second = (
            (total_response_size / (1024 * 1024)) / duration_seconds if duration_seconds > 0 else 0
        )

        # Failure rate
        failure_rate = failed_requests / total_requests if total_requests > 0 else 0

        # System metrics analysis
        peak_cpu_usage = max((m.get("cpu_percent", 0) for m in self.system_metrics), default=0)
        peak_memory_usage_mb = max(
            (m.get("process_memory_mb", 0) for m in self.system_metrics), default=0
        )
        peak_memory_percentage = max(
            (m.get("memory_percent", 0) for m in self.system_metrics), default=0
        )

        # Application metrics
        cache_hit_rate = 0.0
        database_query_count = 0
        database_slow_queries = 0

        # Try to get application metrics from performance manager
        perf_manager = get_performance_manager()
        if perf_manager and perf_manager.cache_manager:
            try:
                cache_stats = await perf_manager.cache_manager.get_stats()
                if cache_stats.get("metrics"):
                    cache_metrics = cache_stats["metrics"]
                    total_cache_ops = cache_metrics.get("hits", 0) + cache_metrics.get("misses", 0)
                    if total_cache_ops > 0:
                        cache_hit_rate = cache_metrics.get("hits", 0) / total_cache_ops
            except Exception:
                pass

        # Collect errors
        errors = []
        error_counts = {}

        for result in self.results:
            if not result.success:
                error = result.metadata.get(
                    "error", f"HTTP {result.metadata.get('status_code', 'Unknown')}"
                )
                error_counts[error] = error_counts.get(error, 0) + 1

        # Format errors with counts
        for error, count in error_counts.items():
            errors.append(f"{error} ({count} times)")

        result = PerformanceTestResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time_ms=avg_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            max_response_time_ms=max_response_time,
            min_response_time_ms=min_response_time,
            requests_per_second=requests_per_second,
            failure_rate=failure_rate,
            throughput_mb_per_second=throughput_mb_per_second,
            peak_cpu_usage=peak_cpu_usage,
            peak_memory_usage_mb=peak_memory_usage_mb,
            peak_memory_percentage=peak_memory_percentage,
            cache_hit_rate=cache_hit_rate,
            database_query_count=database_query_count,
            database_slow_queries=database_slow_queries,
            errors=errors,
        )

        # Print results
        self._print_results(result, config)

        return result

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100.0)
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]

    def _print_results(self, result: PerformanceTestResult, config: LoadTestConfig) -> None:
        """Print test results to console."""
        print(f"\n📊 Performance Test Results: {result.test_name}")
        print("=" * 60)

        # Basic metrics
        print(f"🕒 Duration: {(result.end_time - result.start_time).total_seconds():.1f}s")
        print(f"📈 Total Requests: {result.total_requests}")
        print(
            f"✅ Successful: {result.successful_requests} ({result.successful_requests/result.total_requests*100:.1f}%)"
        )
        print(f"❌ Failed: {result.failed_requests} ({result.failure_rate*100:.1f}%)")
        print(f"⚡ Requests/sec: {result.requests_per_second:.1f}")
        print(f"📊 Throughput: {result.throughput_mb_per_second:.2f} MB/s")

        # Response time metrics
        print(f"\n⏱️ Response Times (ms):")
        print(f"  Average: {result.average_response_time_ms:.1f}")
        print(f"  Median (P50): {result.p50_response_time_ms:.1f}")
        print(f"  P95: {result.p95_response_time_ms:.1f}")
        print(f"  P99: {result.p99_response_time_ms:.1f}")
        print(f"  Min: {result.min_response_time_ms:.1f}")
        print(f"  Max: {result.max_response_time_ms:.1f}")

        # System metrics
        print(f"\n💻 System Resources:")
        print(f"  Peak CPU: {result.peak_cpu_usage:.1f}%")
        print(
            f"  Peak Memory: {result.peak_memory_usage_mb:.1f} MB ({result.peak_memory_percentage:.1f}%)"
        )

        # Application metrics
        print(f"\n🎯 Application Metrics:")
        print(f"  Cache Hit Rate: {result.cache_hit_rate*100:.1f}%")

        # Test status
        passed = result.passed(config)
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\n🏁 Test Status: {status}")

        if not passed:
            if result.failure_rate > config.failure_threshold:
                print(
                    f"  ❌ Failure rate {result.failure_rate*100:.1f}% exceeds threshold {config.failure_threshold*100:.1f}%"
                )
            if result.p95_response_time_ms > config.response_time_threshold_ms:
                print(
                    f"  ❌ P95 response time {result.p95_response_time_ms:.1f}ms exceeds threshold {config.response_time_threshold_ms}ms"
                )

        # Errors
        if result.errors:
            print(f"\n🚨 Errors:")
            for error in result.errors:
                print(f"  • {error}")


class PerformanceTestSuite:
    """Comprehensive performance test suite for the LLM A/B Testing Platform."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.benchmark = PerformanceBenchmark(base_url)
        self.results: List[PerformanceTestResult] = []

    async def run_full_test_suite(self) -> List[PerformanceTestResult]:
        """Run comprehensive performance test suite."""
        print("🚀 Starting Full Performance Test Suite")
        print("=" * 60)

        await self.benchmark.initialize()

        # Test configurations
        light_load = LoadTestConfig(concurrent_users=10, duration_seconds=30)
        medium_load = LoadTestConfig(concurrent_users=50, duration_seconds=60)
        heavy_load = LoadTestConfig(concurrent_users=100, duration_seconds=120)

        # Health endpoint tests
        print("\n🏥 Testing Health Endpoints...")

        result = await self.benchmark.run_api_load_test("/health", "GET", config=light_load)
        self.results.append(result)

        result = await self.benchmark.run_api_load_test(
            "/health/detailed", "GET", config=light_load
        )
        self.results.append(result)

        # API endpoint tests
        print("\n🔧 Testing API Endpoints...")

        # Test endpoints (mock data)
        test_payload = {
            "name": "Performance Test",
            "description": "Load testing scenario",
            "providers": ["openai", "anthropic"],
        }

        result = await self.benchmark.run_api_load_test("/api/v1/tests", "GET", config=medium_load)
        self.results.append(result)

        result = await self.benchmark.run_api_load_test(
            "/api/v1/tests", "POST", test_payload, config=light_load
        )
        self.results.append(result)

        # Provider endpoints
        result = await self.benchmark.run_api_load_test(
            "/api/v1/providers", "GET", config=medium_load
        )
        self.results.append(result)

        # Analytics endpoints
        result = await self.benchmark.run_api_load_test(
            "/api/v1/analytics", "GET", config=medium_load
        )
        self.results.append(result)

        # Performance monitoring endpoints
        result = await self.benchmark.run_api_load_test(
            "/metrics/performance", "GET", config=light_load
        )
        self.results.append(result)

        # Stress test
        print("\n💪 Running Stress Test...")
        result = await self.benchmark.run_api_load_test("/health", "GET", config=heavy_load)
        self.results.append(result)

        self._print_summary()

        return self.results

    def _print_summary(self) -> None:
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE TEST SUITE SUMMARY")
        print("=" * 80)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed(LoadTestConfig()))
        failed_tests = total_tests - passed_tests

        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"❌ Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")

        # Overall metrics
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        overall_failure_rate = (
            (total_requests - total_successful) / total_requests if total_requests > 0 else 0
        )

        print(f"\n🎯 Overall Metrics:")
        print(f"  Total Requests: {total_requests:,}")
        print(f"  Success Rate: {(1-overall_failure_rate)*100:.1f}%")
        print(
            f"  Average RPS: {statistics.mean([r.requests_per_second for r in self.results]):.1f}"
        )
        print(
            f"  Average P95 Response Time: {statistics.mean([r.p95_response_time_ms for r in self.results]):.1f}ms"
        )

        # Performance recommendations
        print(f"\n💡 Performance Recommendations:")
        high_latency_tests = [r for r in self.results if r.p95_response_time_ms > 1000]
        if high_latency_tests:
            print(f"  • {len(high_latency_tests)} endpoints have high P95 latency (>1000ms)")

        low_cache_tests = [r for r in self.results if r.cache_hit_rate < 0.5]
        if low_cache_tests:
            print(f"  • {len(low_cache_tests)} tests have low cache hit rates (<50%)")

        high_error_tests = [r for r in self.results if r.failure_rate > 0.01]
        if high_error_tests:
            print(f"  • {len(high_error_tests)} endpoints have elevated error rates (>1%)")

        print(f"\n🏁 Suite Status: {'✅ PASSED' if failed_tests == 0 else '❌ FAILED'}")


# Convenience functions for quick testing


async def quick_load_test(
    endpoint: str = "/health",
    concurrent_users: int = 50,
    duration_seconds: int = 30,
    base_url: str = "http://localhost:8000",
) -> PerformanceTestResult:
    """Run a quick load test on a single endpoint."""
    benchmark = PerformanceBenchmark(base_url)
    await benchmark.initialize()

    config = LoadTestConfig(concurrent_users=concurrent_users, duration_seconds=duration_seconds)

    return await benchmark.run_api_load_test(endpoint, "GET", config=config)


async def run_performance_validation() -> bool:
    """Run performance validation suite and return True if all tests pass."""
    suite = PerformanceTestSuite()
    results = await suite.run_full_test_suite()

    # Check if all tests passed
    config = LoadTestConfig()  # Default thresholds
    return all(result.passed(config) for result in results)
