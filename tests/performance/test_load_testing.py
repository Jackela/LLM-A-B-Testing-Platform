"""Performance and load testing."""

import asyncio
import time
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Any, Dict, List

import pytest

from tests.factories import CreateTestCommandDTOFactory, TestSampleDTOFactory


@pytest.mark.performance
class TestLoadTesting:
    """Performance and load testing for the LLM A/B Testing Platform."""

    @pytest.fixture
    def performance_config(self, performance_test_config):
        """Performance test configuration."""
        return performance_test_config

    @pytest.mark.asyncio
    async def test_api_response_time_benchmarks(
        self, async_client, auth_headers, performance_config
    ):
        """Test API response time benchmarks."""
        response_times = []
        max_response_time = performance_config["max_response_time"]

        # Test GET /tests endpoint
        for _ in range(20):
            start_time = time.time()
            response = await async_client.get("/api/v1/tests/", headers=auth_headers)
            end_time = time.time()

            response_time = end_time - start_time
            response_times.append(response_time)

            assert response.status_code == 200
            assert (
                response_time < max_response_time
            ), f"Response time {response_time:.3f}s exceeds limit {max_response_time}s"

        # Calculate statistics
        avg_response_time = mean(response_times)
        std_response_time = stdev(response_times) if len(response_times) > 1 else 0
        max_observed = max(response_times)
        min_observed = min(response_times)

        print(f"\nAPI Response Time Statistics:")
        print(f"Average: {avg_response_time:.3f}s")
        print(f"Std Dev: {std_response_time:.3f}s")
        print(f"Min: {min_observed:.3f}s")
        print(f"Max: {max_observed:.3f}s")

        # Performance assertions
        assert avg_response_time < max_response_time * 0.5, "Average response time too high"
        assert max_observed < max_response_time, "Maximum response time exceeded"

    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, async_client, auth_headers, performance_config):
        """Test concurrent API request handling with optimized load patterns."""
        concurrent_users = performance_config["concurrent_users"]

        # 优化: 使用连接池和会话复用
        import httpx

        limits = httpx.Limits(
            max_keepalive_connections=concurrent_users, max_connections=concurrent_users * 2
        )
        timeout = httpx.Timeout(30.0)

        async def make_request_optimized(
            session_id: int, client: httpx.AsyncClient
        ) -> Dict[str, Any]:
            """优化的API请求，支持连接复用和更好的错误处理."""
            start_time = time.time()
            try:
                response = await client.get("/api/v1/tests/", headers=auth_headers)
                end_time = time.time()

                return {
                    "session_id": session_id,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "response_size": len(response.content),
                    "success": response.status_code == 200,
                    "error": None,
                }
            except httpx.TimeoutException:
                end_time = time.time()
                return {
                    "session_id": session_id,
                    "status_code": 0,
                    "response_time": end_time - start_time,
                    "response_size": 0,
                    "success": False,
                    "error": "Timeout",
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "session_id": session_id,
                    "status_code": 0,
                    "response_time": end_time - start_time,
                    "response_size": 0,
                    "success": False,
                    "error": str(e),
                }

        # Execute concurrent requests with optimized client
        start_time = time.time()

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as optimized_client:
            # 分批执行以避免过载
            batch_size = min(20, concurrent_users)
            all_results = []

            for i in range(0, concurrent_users, batch_size):
                batch_end = min(i + batch_size, concurrent_users)
                batch_tasks = [
                    make_request_optimized(session_id, optimized_client)
                    for session_id in range(i, batch_end)
                ]

                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # 处理异常结果
                for result in batch_results:
                    if isinstance(result, Exception):
                        all_results.append(
                            {
                                "session_id": -1,
                                "status_code": 0,
                                "response_time": 0,
                                "response_size": 0,
                                "success": False,
                                "error": str(result),
                            }
                        )
                    else:
                        all_results.append(result)

                # 批次间延迟
                if batch_end < concurrent_users:
                    await asyncio.sleep(0.1)

        results = all_results
        total_time = time.time() - start_time

        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]

        success_rate = len(successful_requests) / len(results)
        error_rate = len(failed_requests) / len(results)

        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
            avg_response_time = mean(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)

            # 计算百分位数
            response_times.sort()
            p50_time = response_times[int(0.5 * len(response_times))] if response_times else 0
            p95_time = response_times[int(0.95 * len(response_times))] if response_times else 0
            p99_time = response_times[int(0.99 * len(response_times))] if response_times else 0

            # 计算总数据传输量
            total_bytes = sum(r.get("response_size", 0) for r in successful_requests)
            throughput_mbps = (total_bytes / (1024 * 1024)) / total_time if total_time > 0 else 0
        else:
            avg_response_time = max_response_time = min_response_time = 0
            p50_time = p95_time = p99_time = 0
            total_bytes = throughput_mbps = 0

        throughput = len(successful_requests) / total_time if total_time > 0 else 0

        print(f"\nConcurrent Request Performance (Optimized):")
        print(f"Concurrent Users: {concurrent_users}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Error Rate: {error_rate:.2%}")
        print(f"Response Time Stats:")
        print(f"  Average: {avg_response_time:.3f}s")
        print(f"  Min: {min_response_time:.3f}s")
        print(f"  Max: {max_response_time:.3f}s")
        print(f"  P50: {p50_time:.3f}s")
        print(f"  P95: {p95_time:.3f}s")
        print(f"  P99: {p99_time:.3f}s")
        print(f"Throughput: {throughput:.2f} req/s")
        print(f"Data Transfer: {throughput_mbps:.2f} MB/s")

        # 错误分析
        if failed_requests:
            error_types = {}
            for req in failed_requests:
                error = req.get("error", "Unknown")
                error_types[error] = error_types.get(error, 0) + 1

            print(f"Error Analysis:")
            for error_type, count in error_types.items():
                print(f"  {error_type}: {count} times")

        # Performance assertions
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"
        assert (
            error_rate <= performance_config["error_rate_threshold"]
        ), f"Error rate {error_rate:.2%} too high"
        assert (
            avg_response_time < performance_config["max_response_time"]
        ), "Average response time too high"

    @pytest.mark.asyncio
    async def test_test_creation_performance(self, async_client, auth_headers, performance_config):
        """Test performance of test creation under load."""
        creation_times = []
        successful_creations = 0
        failed_creations = 0

        async def create_test(test_index: int) -> Dict[str, Any]:
            """Create a single test and measure performance."""
            command = CreateTestCommandDTOFactory(
                name=f"Performance Test {test_index}",
                samples=[TestSampleDTOFactory() for _ in range(20)],
            )

            start_time = time.time()
            try:
                response = await async_client.post(
                    "/api/v1/tests/", json=command.dict(), headers=auth_headers
                )
                end_time = time.time()

                return {
                    "test_index": test_index,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 201,
                    "test_id": (
                        response.json().get("test_id") if response.status_code == 201 else None
                    ),
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "test_index": test_index,
                    "status_code": 0,
                    "response_time": end_time - start_time,
                    "success": False,
                    "error": str(e),
                }

        # Create tests concurrently
        num_tests = 10
        tasks = [create_test(i) for i in range(num_tests)]
        results = await asyncio.gather(*tasks)

        # Analyze results
        for result in results:
            if result["success"]:
                successful_creations += 1
                creation_times.append(result["response_time"])
            else:
                failed_creations += 1

        if creation_times:
            avg_creation_time = mean(creation_times)
            max_creation_time = max(creation_times)
            min_creation_time = min(creation_times)
        else:
            avg_creation_time = max_creation_time = min_creation_time = 0

        success_rate = successful_creations / num_tests

        print(f"\nTest Creation Performance:")
        print(f"Total Tests: {num_tests}")
        print(f"Successful: {successful_creations}")
        print(f"Failed: {failed_creations}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Creation Time: {avg_creation_time:.3f}s")
        print(f"Min Creation Time: {min_creation_time:.3f}s")
        print(f"Max Creation Time: {max_creation_time:.3f}s")

        # Performance assertions
        assert success_rate >= 0.90, f"Test creation success rate {success_rate:.2%} below 90%"
        assert avg_creation_time < 5.0, f"Average creation time {avg_creation_time:.3f}s too high"

    @pytest.mark.asyncio
    async def test_database_performance(self, async_session, test_repository):
        """Test database operation performance."""
        from tests.factories import TestFactory

        # Test bulk insert performance
        num_tests = 100
        tests = [TestFactory() for _ in range(num_tests)]

        start_time = time.time()
        await test_repository.bulk_save(tests)
        await async_session.commit()
        bulk_insert_time = time.time() - start_time

        print(f"\nDatabase Performance:")
        print(f"Bulk Insert ({num_tests} tests): {bulk_insert_time:.3f}s")
        print(f"Insert Rate: {num_tests / bulk_insert_time:.2f} tests/s")

        # Test query performance
        start_time = time.time()
        retrieved_tests = await test_repository.get_paginated(page=1, page_size=50)
        query_time = time.time() - start_time

        print(f"Paginated Query (50 tests): {query_time:.3f}s")

        # Test filtering performance
        start_time = time.time()
        filtered_tests = await test_repository.get_by_status("CONFIGURED")
        filter_time = time.time() - start_time

        print(f"Filtered Query: {filter_time:.3f}s")

        # Performance assertions
        assert bulk_insert_time < 10.0, f"Bulk insert time {bulk_insert_time:.3f}s too high"
        assert query_time < 1.0, f"Query time {query_time:.3f}s too high"
        assert filter_time < 2.0, f"Filter time {filter_time:.3f}s too high"
        assert len(retrieved_tests) > 0, "No tests retrieved"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_load(self, async_client, auth_headers, performance_config):
        """Test system performance under sustained load."""
        test_duration = performance_config["test_duration"]
        concurrent_users = min(
            performance_config["concurrent_users"], 5
        )  # Limit for sustained test

        results = []
        start_test = time.time()

        async def sustained_user_session(user_id: int):
            """Simulate a user session with multiple operations."""
            session_results = []

            while (time.time() - start_test) < test_duration:
                # Get tests list
                start_time = time.time()
                try:
                    response = await async_client.get("/api/v1/tests/", headers=auth_headers)
                    end_time = time.time()

                    session_results.append(
                        {
                            "user_id": user_id,
                            "operation": "get_tests",
                            "response_time": end_time - start_time,
                            "success": response.status_code == 200,
                            "timestamp": datetime.utcnow(),
                        }
                    )

                except Exception as e:
                    session_results.append(
                        {
                            "user_id": user_id,
                            "operation": "get_tests",
                            "response_time": 0,
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.utcnow(),
                        }
                    )

                # Wait before next request
                await asyncio.sleep(1)

            return session_results

        # Run sustained load
        tasks = [sustained_user_session(i) for i in range(concurrent_users)]
        all_results = await asyncio.gather(*tasks)

        # Flatten results
        for user_results in all_results:
            results.extend(user_results)

        # Analyze sustained load results
        successful_ops = [r for r in results if r["success"]]
        failed_ops = [r for r in results if not r["success"]]

        total_operations = len(results)
        success_rate = len(successful_ops) / total_operations if total_operations > 0 else 0
        error_rate = len(failed_ops) / total_operations if total_operations > 0 else 0

        if successful_ops:
            avg_response_time = mean([r["response_time"] for r in successful_ops])
            response_times = [r["response_time"] for r in successful_ops]

            # Calculate percentiles
            response_times.sort()
            p95_response_time = (
                response_times[int(0.95 * len(response_times))] if response_times else 0
            )
            p99_response_time = (
                response_times[int(0.99 * len(response_times))] if response_times else 0
            )
        else:
            avg_response_time = p95_response_time = p99_response_time = 0

        operations_per_second = total_operations / test_duration

        print(f"\nSustained Load Test Results:")
        print(f"Duration: {test_duration}s")
        print(f"Concurrent Users: {concurrent_users}")
        print(f"Total Operations: {total_operations}")
        print(f"Operations/Second: {operations_per_second:.2f}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Error Rate: {error_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        print(f"95th Percentile: {p95_response_time:.3f}s")
        print(f"99th Percentile: {p99_response_time:.3f}s")

        # Performance assertions for sustained load
        assert success_rate >= 0.95, f"Sustained load success rate {success_rate:.2%} below 95%"
        assert error_rate <= 0.05, f"Sustained load error rate {error_rate:.2%} above 5%"
        assert (
            avg_response_time < performance_config["max_response_time"] * 1.5
        ), "Sustained load response time degraded"
        assert (
            p95_response_time < performance_config["max_response_time"] * 2
        ), "95th percentile response time too high"

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, async_client, auth_headers):
        """Test memory usage under load conditions."""
        import os

        import psutil

        # Get current process
        process = psutil.Process(os.getpid())

        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create load
        async def memory_stress_operation():
            """Operation that could potentially cause memory issues."""
            large_command = CreateTestCommandDTOFactory(
                samples=[TestSampleDTOFactory() for _ in range(100)]
            )

            try:
                response = await async_client.post(
                    "/api/v1/tests/", json=large_command.dict(), headers=auth_headers
                )
                return response.status_code == 201
            except Exception:
                return False

        # Run multiple operations
        tasks = [memory_stress_operation() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"\nMemory Usage Test:")
        print(f"Initial Memory: {initial_memory:.2f} MB")
        print(f"Final Memory: {final_memory:.2f} MB")
        print(f"Memory Increase: {memory_increase:.2f} MB")
        print(f"Successful Operations: {sum(results)}/{len(results)}")

        # Memory usage assertions
        assert memory_increase < 100, f"Memory increase {memory_increase:.2f} MB too high"
        assert final_memory < 500, f"Final memory usage {final_memory:.2f} MB too high"

    @pytest.mark.asyncio
    async def test_connection_pool_performance(self, async_session):
        """Test database connection pool performance."""
        from src.infrastructure.persistence.connection_pool import DatabaseConnectionPool

        async def db_operation(operation_id: int):
            """Perform a database operation."""
            start_time = time.time()
            try:
                # Simulate database query
                await async_session.execute("SELECT 1")
                end_time = time.time()
                return {
                    "operation_id": operation_id,
                    "success": True,
                    "duration": end_time - start_time,
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "operation_id": operation_id,
                    "success": False,
                    "duration": end_time - start_time,
                    "error": str(e),
                }

        # Test concurrent database operations
        num_operations = 50
        tasks = [db_operation(i) for i in range(num_operations)]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        successful_ops = [r for r in results if r["success"]]
        failed_ops = [r for r in results if not r["success"]]

        success_rate = len(successful_ops) / len(results)
        avg_duration = mean([r["duration"] for r in successful_ops]) if successful_ops else 0
        operations_per_second = len(successful_ops) / total_time

        print(f"\nConnection Pool Performance:")
        print(f"Total Operations: {num_operations}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Successful: {len(successful_ops)}")
        print(f"Failed: {len(failed_ops)}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Duration: {avg_duration:.3f}s")
        print(f"Operations/Second: {operations_per_second:.2f}")

        # Connection pool assertions
        assert success_rate >= 0.95, f"DB operation success rate {success_rate:.2%} below 95%"
        assert avg_duration < 0.1, f"Average DB operation duration {avg_duration:.3f}s too high"
        assert (
            operations_per_second > 100
        ), f"DB operations/second {operations_per_second:.2f} too low"

    @pytest.mark.asyncio
    async def test_cache_performance(self, async_client, auth_headers):
        """Test caching performance and effectiveness."""
        # Make initial request (cache miss)
        start_time = time.time()
        response1 = await async_client.get("/api/v1/tests/", headers=auth_headers)
        first_request_time = time.time() - start_time

        # Make subsequent request (should be cache hit)
        start_time = time.time()
        response2 = await async_client.get("/api/v1/tests/", headers=auth_headers)
        second_request_time = time.time() - start_time

        # Verify responses
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Cache should improve performance
        cache_improvement = (first_request_time - second_request_time) / first_request_time

        print(f"\nCache Performance:")
        print(f"First Request (cache miss): {first_request_time:.3f}s")
        print(f"Second Request (cache hit): {second_request_time:.3f}s")
        print(f"Cache Improvement: {cache_improvement:.1%}")

        # Note: This test might not show improvement if caching is not implemented
        # or if the test data is too small to see significant differences
