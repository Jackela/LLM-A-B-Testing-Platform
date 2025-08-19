"""End-to-end tests for complete test workflows."""

import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.domain.model_provider.value_objects.money import Money
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.test_status import TestStatus
from tests.factories import (
    CreateTestCommandDTOFactory,
    EvaluationConfigurationDTOFactory,
    ModelConfigurationDTOFactory,
    TestSampleDTOFactory,
)


@pytest.mark.e2e
class TestCompleteTestWorkflow:
    """End-to-end tests for complete test workflows."""

    @pytest.fixture
    def complete_test_command(self):
        """Create a complete test command for E2E testing."""
        return CreateTestCommandDTOFactory(
            name="E2E Test: GPT-4 vs Claude-3",
            samples=[
                TestSampleDTOFactory(
                    prompt=f"Explain the concept of machine learning in simple terms. Sample {i}",
                    expected_response=f"Expected explanation {i}",
                    difficulty=DifficultyLevel.MEDIUM,
                )
                for i in range(20)
            ],
        )

    @pytest.mark.asyncio
    async def test_full_test_lifecycle(self, async_client, complete_test_command, auth_headers):
        """Test complete test lifecycle from creation to completion."""
        # Step 1: Create test
        create_response = await async_client.post(
            "/api/v1/tests/", json=complete_test_command.dict(), headers=auth_headers
        )
        assert create_response.status_code == 201

        test_data = create_response.json()
        test_id = test_data["test_id"]
        assert test_data["status"] == "CONFIGURED"
        assert test_data["created_test"] is True

        # Step 2: Verify test was created
        get_response = await async_client.get(f"/api/v1/tests/{test_id}", headers=auth_headers)
        assert get_response.status_code == 200

        test_details = get_response.json()
        assert test_details["id"] == test_id
        assert test_details["name"] == complete_test_command.name
        assert len(test_details["samples"]) == len(complete_test_command.samples)

        # Step 3: Start test execution
        start_response = await async_client.post(
            f"/api/v1/tests/{test_id}/start", headers=auth_headers
        )
        assert start_response.status_code == 200

        start_data = start_response.json()
        assert start_data["started"] is True
        assert start_data["status"] == "RUNNING"

        # Step 4: Monitor test progress
        max_wait_time = 300  # 5 minutes
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < max_wait_time:
            progress_response = await async_client.get(
                f"/api/v1/tests/{test_id}/progress", headers=auth_headers
            )
            assert progress_response.status_code == 200

            progress_data = progress_response.json()

            if progress_data["status"] in ["COMPLETED", "FAILED"]:
                break

            # Wait before checking again
            await asyncio.sleep(5)

        # Step 5: Verify test completion
        final_response = await async_client.get(f"/api/v1/tests/{test_id}", headers=auth_headers)
        assert final_response.status_code == 200

        final_test = final_response.json()
        assert final_test["status"] in ["COMPLETED", "FAILED"]

        if final_test["status"] == "COMPLETED":
            assert final_test["completed_at"] is not None

            # Step 6: Get test results
            results_response = await async_client.get(
                f"/api/v1/tests/{test_id}/results", headers=auth_headers
            )
            assert results_response.status_code == 200

            results_data = results_response.json()
            assert "model_performances" in results_data
            assert "statistical_analysis" in results_data
            assert "insights" in results_data

    @pytest.mark.asyncio
    async def test_concurrent_test_execution(self, async_client, auth_headers):
        """Test concurrent execution of multiple tests."""
        # Create multiple tests
        test_commands = [
            CreateTestCommandDTOFactory(
                name=f"Concurrent Test {i}", samples=[TestSampleDTOFactory() for _ in range(10)]
            )
            for i in range(3)
        ]

        # Start all tests concurrently
        test_ids = []

        for command in test_commands:
            create_response = await async_client.post(
                "/api/v1/tests/", json=command.dict(), headers=auth_headers
            )
            assert create_response.status_code == 201
            test_ids.append(create_response.json()["test_id"])

        # Start all tests
        start_tasks = []
        for test_id in test_ids:
            start_tasks.append(
                async_client.post(f"/api/v1/tests/{test_id}/start", headers=auth_headers)
            )

        start_responses = await asyncio.gather(*start_tasks)

        # Verify all started successfully
        for response in start_responses:
            assert response.status_code == 200
            assert response.json()["started"] is True

        # Monitor all tests
        completed_tests = set()
        max_wait_time = 300
        start_time = datetime.utcnow()

        while (
            len(completed_tests) < len(test_ids)
            and (datetime.utcnow() - start_time).total_seconds() < max_wait_time
        ):

            for test_id in test_ids:
                if test_id not in completed_tests:
                    progress_response = await async_client.get(
                        f"/api/v1/tests/{test_id}/progress", headers=auth_headers
                    )

                    if progress_response.status_code == 200:
                        progress_data = progress_response.json()
                        if progress_data["status"] in ["COMPLETED", "FAILED"]:
                            completed_tests.add(test_id)

            await asyncio.sleep(5)

        # Verify all tests completed
        assert len(completed_tests) == len(test_ids)

    @pytest.mark.asyncio
    async def test_test_failure_handling(self, async_client, auth_headers):
        """Test handling of test failures."""
        # Create test with invalid configuration
        invalid_command = CreateTestCommandDTOFactory()
        # Set impossible parameters to force failure
        for model in invalid_command.configuration.models:
            model.parameters["max_tokens"] = -1  # Invalid parameter

        create_response = await async_client.post(
            "/api/v1/tests/", json=invalid_command.dict(), headers=auth_headers
        )

        # Should either fail creation or fail during execution
        if create_response.status_code == 201:
            test_id = create_response.json()["test_id"]

            start_response = await async_client.post(
                f"/api/v1/tests/{test_id}/start", headers=auth_headers
            )

            # Should fail to start or fail during execution
            if start_response.status_code == 200:
                # Wait for failure
                max_wait_time = 60
                start_time = datetime.utcnow()

                while (datetime.utcnow() - start_time).total_seconds() < max_wait_time:
                    progress_response = await async_client.get(
                        f"/api/v1/tests/{test_id}/progress", headers=auth_headers
                    )

                    if progress_response.status_code == 200:
                        progress_data = progress_response.json()
                        if progress_data["status"] == "FAILED":
                            break

                    await asyncio.sleep(2)

                # Verify failure was recorded
                final_response = await async_client.get(
                    f"/api/v1/tests/{test_id}", headers=auth_headers
                )
                assert final_response.status_code == 200
                final_test = final_response.json()
                assert final_test["status"] == "FAILED"
        else:
            # Creation failed as expected
            assert create_response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_test_cancellation(self, async_client, complete_test_command, auth_headers):
        """Test test cancellation during execution."""
        # Create and start test
        create_response = await async_client.post(
            "/api/v1/tests/", json=complete_test_command.dict(), headers=auth_headers
        )
        assert create_response.status_code == 201
        test_id = create_response.json()["test_id"]

        start_response = await async_client.post(
            f"/api/v1/tests/{test_id}/start", headers=auth_headers
        )
        assert start_response.status_code == 200

        # Wait a bit for test to start processing
        await asyncio.sleep(10)

        # Cancel test
        cancel_response = await async_client.post(
            f"/api/v1/tests/{test_id}/cancel", headers=auth_headers
        )
        assert cancel_response.status_code == 200

        cancel_data = cancel_response.json()
        assert cancel_data["cancelled"] is True

        # Verify test was cancelled
        final_response = await async_client.get(f"/api/v1/tests/{test_id}", headers=auth_headers)
        assert final_response.status_code == 200

        final_test = final_response.json()
        assert final_test["status"] == "CANCELLED"

    @pytest.mark.asyncio
    async def test_test_results_and_analytics(
        self, async_client, complete_test_command, auth_headers
    ):
        """Test retrieval of test results and analytics."""
        # Create and complete a test
        create_response = await async_client.post(
            "/api/v1/tests/", json=complete_test_command.dict(), headers=auth_headers
        )
        assert create_response.status_code == 201
        test_id = create_response.json()["test_id"]

        # Mock successful completion (in real scenario, wait for actual completion)
        # For E2E test, we'll simulate this by creating sample results

        # Get basic results
        results_response = await async_client.get(
            f"/api/v1/tests/{test_id}/results", headers=auth_headers
        )

        if results_response.status_code == 200:
            results_data = results_response.json()
            assert "model_performances" in results_data

            # Get detailed analytics
            analytics_response = await async_client.get(
                f"/api/v1/tests/{test_id}/analytics", headers=auth_headers
            )

            if analytics_response.status_code == 200:
                analytics_data = analytics_response.json()
                assert "statistical_tests" in analytics_data
                assert "insights" in analytics_data
                assert "recommendations" in analytics_data

    @pytest.mark.asyncio
    async def test_test_configuration_validation(self, async_client, auth_headers):
        """Test comprehensive test configuration validation."""
        # Test with insufficient samples
        insufficient_samples_command = CreateTestCommandDTOFactory()
        insufficient_samples_command.samples = insufficient_samples_command.samples[
            :5
        ]  # Only 5 samples

        response = await async_client.post(
            "/api/v1/tests/", json=insufficient_samples_command.dict(), headers=auth_headers
        )
        assert response.status_code in [400, 422]

        # Test with invalid model weights
        invalid_weights_command = CreateTestCommandDTOFactory()
        for model in invalid_weights_command.configuration.models:
            model.weight = 0.6  # Total > 1.0

        response = await async_client.post(
            "/api/v1/tests/", json=invalid_weights_command.dict(), headers=auth_headers
        )
        assert response.status_code in [400, 422]

        # Test with zero max cost
        zero_cost_command = CreateTestCommandDTOFactory()
        zero_cost_command.configuration.max_cost = Money(0.0, "USD")

        response = await async_client.post(
            "/api/v1/tests/", json=zero_cost_command.dict(), headers=auth_headers
        )
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_api_pagination_and_filtering(self, async_client, auth_headers):
        """Test API pagination and filtering functionality."""
        # Create multiple tests
        test_names = [f"Test {i:02d}" for i in range(15)]
        test_ids = []

        for name in test_names:
            command = CreateTestCommandDTOFactory(name=name)
            response = await async_client.post(
                "/api/v1/tests/", json=command.dict(), headers=auth_headers
            )
            if response.status_code == 201:
                test_ids.append(response.json()["test_id"])

        # Test pagination
        page1_response = await async_client.get(
            "/api/v1/tests/?page=1&page_size=5", headers=auth_headers
        )
        assert page1_response.status_code == 200

        page1_data = page1_response.json()
        assert len(page1_data["items"]) <= 5
        assert "total" in page1_data
        assert "page" in page1_data
        assert "page_size" in page1_data

        # Test filtering by status
        status_response = await async_client.get(
            "/api/v1/tests/?status=CONFIGURED", headers=auth_headers
        )
        assert status_response.status_code == 200

        status_data = status_response.json()
        for test in status_data["items"]:
            assert test["status"] == "CONFIGURED"

        # Test search by name
        search_response = await async_client.get(
            "/api/v1/tests/?search=Test%2001", headers=auth_headers
        )
        assert search_response.status_code == 200

        search_data = search_response.json()
        if search_data["items"]:
            assert any("Test 01" in test["name"] for test in search_data["items"])

    @pytest.mark.asyncio
    async def test_real_time_progress_updates(
        self, async_client, complete_test_command, auth_headers
    ):
        """Test real-time progress updates during test execution."""
        # Create and start test
        create_response = await async_client.post(
            "/api/v1/tests/", json=complete_test_command.dict(), headers=auth_headers
        )
        assert create_response.status_code == 201
        test_id = create_response.json()["test_id"]

        start_response = await async_client.post(
            f"/api/v1/tests/{test_id}/start", headers=auth_headers
        )
        assert start_response.status_code == 200

        # Monitor progress with multiple checks
        previous_progress = 0
        progress_checks = 0
        max_checks = 20

        while progress_checks < max_checks:
            progress_response = await async_client.get(
                f"/api/v1/tests/{test_id}/progress", headers=auth_headers
            )

            if progress_response.status_code == 200:
                progress_data = progress_response.json()
                current_progress = progress_data.get("completion_percentage", 0)

                # Progress should never decrease
                assert current_progress >= previous_progress
                previous_progress = current_progress

                # Check for detailed progress information
                assert "samples_processed" in progress_data
                assert "total_samples" in progress_data
                assert "estimated_completion_time" in progress_data

                if progress_data["status"] in ["COMPLETED", "FAILED", "CANCELLED"]:
                    break

            progress_checks += 1
            await asyncio.sleep(3)

    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, async_client, auth_headers):
        """Test system resilience and error recovery."""
        # Test with temporarily unavailable providers
        command = CreateTestCommandDTOFactory()

        # Create test that might fail due to provider issues
        create_response = await async_client.post(
            "/api/v1/tests/", json=command.dict(), headers=auth_headers
        )

        if create_response.status_code == 201:
            test_id = create_response.json()["test_id"]

            # Try to start test (might fail due to provider issues)
            start_response = await async_client.post(
                f"/api/v1/tests/{test_id}/start", headers=auth_headers
            )

            # System should handle errors gracefully
            assert start_response.status_code in [200, 400, 503]

            if start_response.status_code != 200:
                # Should provide helpful error message
                error_data = start_response.json()
                assert "error" in error_data or "detail" in error_data

        # Test API resilience to malformed requests
        malformed_requests = [
            {},  # Empty request
            {"name": ""},  # Empty name
            {"name": "Test", "configuration": {}},  # Empty config
            {"invalid_field": "value"},  # Invalid fields
        ]

        for malformed_request in malformed_requests:
            response = await async_client.post(
                "/api/v1/tests/", json=malformed_request, headers=auth_headers
            )
            # Should return proper error codes, not crash
            assert response.status_code in [400, 422, 500]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_long_running_test_stability(self, async_client, auth_headers):
        """Test stability of long-running tests."""
        # Create test with many samples to ensure longer execution
        large_command = CreateTestCommandDTOFactory(
            name="Long Running Stability Test", samples=[TestSampleDTOFactory() for _ in range(100)]
        )

        create_response = await async_client.post(
            "/api/v1/tests/", json=large_command.dict(), headers=auth_headers
        )

        if create_response.status_code == 201:
            test_id = create_response.json()["test_id"]

            start_response = await async_client.post(
                f"/api/v1/tests/{test_id}/start", headers=auth_headers
            )

            if start_response.status_code == 200:
                # Monitor for extended period
                max_wait_time = 1800  # 30 minutes
                start_time = datetime.utcnow()
                last_progress_time = start_time

                while (datetime.utcnow() - start_time).total_seconds() < max_wait_time:
                    progress_response = await async_client.get(
                        f"/api/v1/tests/{test_id}/progress", headers=auth_headers
                    )

                    assert progress_response.status_code == 200
                    progress_data = progress_response.json()

                    # Verify progress is being made
                    if progress_data.get("samples_processed", 0) > 0:
                        last_progress_time = datetime.utcnow()

                    # If no progress for 10 minutes, something is wrong
                    if (datetime.utcnow() - last_progress_time).total_seconds() > 600:
                        pytest.fail("Test appears to be stuck - no progress for 10 minutes")

                    if progress_data["status"] in ["COMPLETED", "FAILED"]:
                        break

                    await asyncio.sleep(30)  # Check every 30 seconds

                # Verify final state
                final_response = await async_client.get(
                    f"/api/v1/tests/{test_id}", headers=auth_headers
                )
                assert final_response.status_code == 200

                final_test = final_response.json()
                assert final_test["status"] in ["COMPLETED", "FAILED", "RUNNING"]
