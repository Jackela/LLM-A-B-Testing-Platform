"""Integration tests for TestOrchestrationService."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.application.dto.test_configuration_dto import (
    CreateTestCommandDTO,
    EvaluationConfigurationDTO,
    ModelConfigurationDTO,
    TestConfigurationDTO,
    TestMonitoringResultDTO,
    TestResultDTO,
    TestSampleDTO,
)
from src.application.services.test_orchestration_service import TestOrchestrationService
from src.domain.model_provider.value_objects.money import Money
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.test_status import TestStatus


@pytest.fixture
def mock_uow():
    """Mock unit of work."""
    uow = AsyncMock()
    uow.tests = AsyncMock()
    uow.providers = AsyncMock()
    return uow


@pytest.fixture
def mock_event_publisher():
    """Mock domain event publisher."""
    return AsyncMock()


@pytest.fixture
def mock_use_cases():
    """Mock all use cases."""
    return {
        "create_test": AsyncMock(),
        "start_test": AsyncMock(),
        "monitor_test": AsyncMock(),
        "complete_test": AsyncMock(),
        "process_samples": AsyncMock(),
    }


@pytest.fixture
def orchestration_service(mock_uow, mock_event_publisher, mock_use_cases):
    """Test orchestration service instance."""
    return TestOrchestrationService(
        uow=mock_uow,
        event_publisher=mock_event_publisher,
        create_test_use_case=mock_use_cases["create_test"],
        start_test_use_case=mock_use_cases["start_test"],
        monitor_test_use_case=mock_use_cases["monitor_test"],
        complete_test_use_case=mock_use_cases["complete_test"],
        process_samples_use_case=mock_use_cases["process_samples"],
    )


@pytest.fixture
def valid_create_command():
    """Valid test creation command."""
    return CreateTestCommandDTO(
        name="Integration Test",
        configuration=TestConfigurationDTO(
            models=[
                ModelConfigurationDTO(
                    model_id="gpt-4",
                    provider_name="openai",
                    parameters={"temperature": 0.7},
                    weight=0.5,
                ),
                ModelConfigurationDTO(
                    model_id="claude-3",
                    provider_name="anthropic",
                    parameters={"temperature": 0.7},
                    weight=0.5,
                ),
            ],
            evaluation=EvaluationConfigurationDTO(template_id="standard"),
            max_cost=Money(100.0, "USD"),
        ),
        samples=[
            TestSampleDTO(prompt=f"Test prompt {i}", difficulty=DifficultyLevel.MEDIUM)
            for i in range(50)
        ],
    )


class TestOrchestrationService:
    """Test cases for TestOrchestrationService."""

    @pytest.mark.asyncio
    async def test_create_and_start_test_success(
        self, orchestration_service, valid_create_command, mock_use_cases
    ):
        """Test successful create and start workflow."""
        # Arrange
        test_id = uuid4()
        mock_use_cases["create_test"].execute.return_value = TestResultDTO(
            test_id=test_id,
            status="CONFIGURED",
            created_test=True,
            estimated_cost=Money(25.0, "USD"),
            estimated_duration=300.0,
        )
        mock_use_cases["start_test"].execute.return_value = TestResultDTO(
            test_id=test_id, status="RUNNING", created_test=False
        )

        # Act
        result = await orchestration_service.create_and_start_test(valid_create_command)

        # Assert
        assert result["success"] is True
        assert result["test_id"] == test_id
        assert result["status"] == "RUNNING"
        assert result["estimated_cost"] == Money(25.0, "USD")
        assert result["estimated_duration"] == 300.0

        # Verify use cases were called
        mock_use_cases["create_test"].execute.assert_called_once_with(valid_create_command)
        mock_use_cases["start_test"].execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_and_start_test_creation_failure(
        self, orchestration_service, valid_create_command, mock_use_cases
    ):
        """Test create and start workflow when creation fails."""
        # Arrange
        mock_use_cases["create_test"].execute.return_value = TestResultDTO(
            test_id=uuid4(),
            status="validation_failed",
            created_test=False,
            errors=["Invalid configuration"],
        )

        # Act
        result = await orchestration_service.create_and_start_test(valid_create_command)

        # Assert
        assert result["success"] is False
        assert result["stage"] == "creation"
        assert result["test_id"] is None
        assert "Invalid configuration" in result["errors"]

        # Verify start was not called
        mock_use_cases["start_test"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_and_start_test_start_failure(
        self, orchestration_service, valid_create_command, mock_use_cases
    ):
        """Test create and start workflow when start fails."""
        # Arrange
        test_id = uuid4()
        mock_use_cases["create_test"].execute.return_value = TestResultDTO(
            test_id=test_id, status="CONFIGURED", created_test=True
        )
        mock_use_cases["start_test"].execute.return_value = TestResultDTO(
            test_id=test_id,
            status="validation_failed",
            created_test=False,
            errors=["Cannot start test"],
        )

        # Act
        result = await orchestration_service.create_and_start_test(valid_create_command)

        # Assert
        assert result["success"] is False
        assert result["stage"] == "starting"
        assert result["test_id"] == test_id
        assert "Cannot start test" in result["errors"]

    @pytest.mark.asyncio
    async def test_schedule_sample_processing(self, orchestration_service):
        """Test scheduling sample processing."""
        # Arrange
        test_id = uuid4()

        # Act
        result = await orchestration_service.schedule_sample_processing(test_id)

        # Assert
        assert result is True
        assert test_id in orchestration_service._active_processing_tasks

        # Clean up
        task = orchestration_service._active_processing_tasks[test_id]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_schedule_sample_processing_already_active(self, orchestration_service):
        """Test scheduling when processing already active."""
        # Arrange
        test_id = uuid4()
        await orchestration_service.schedule_sample_processing(test_id)

        # Act
        result = await orchestration_service.schedule_sample_processing(test_id)

        # Assert
        assert result is False

        # Clean up
        task = orchestration_service._active_processing_tasks[test_id]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_cancel_test_processing(self, orchestration_service, mock_uow):
        """Test cancelling test processing."""
        # Arrange
        test_id = uuid4()
        mock_test = MagicMock()
        mock_test.status.is_terminal.return_value = False
        mock_uow.tests.find_by_id.return_value = mock_test

        await orchestration_service.schedule_sample_processing(test_id)

        # Act
        result = await orchestration_service.cancel_test_processing(test_id, "User request")

        # Assert
        assert result is True
        mock_test.cancel.assert_called_once_with("User request")
        mock_uow.tests.save.assert_called_once_with(mock_test)
        mock_uow.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_active_tests(self, orchestration_service, mock_uow, mock_use_cases):
        """Test getting all active tests."""
        # Arrange
        test_id1, test_id2 = uuid4(), uuid4()
        mock_test1 = MagicMock()
        mock_test1.id = test_id1
        mock_test2 = MagicMock()
        mock_test2.id = test_id2

        mock_uow.tests.find_active_tests.return_value = [mock_test1, mock_test2]

        mock_use_cases["monitor_test"].execute.side_effect = [
            TestMonitoringResultDTO(
                test_id=test_id1,
                status="RUNNING",
                progress=0.5,
                total_samples=100,
                evaluated_samples=50,
                model_scores={},
                estimated_remaining_time=150.0,
                current_cost=Money(5.0, "USD"),
                errors=[],
            ),
            TestMonitoringResultDTO(
                test_id=test_id2,
                status="RUNNING",
                progress=0.8,
                total_samples=200,
                evaluated_samples=160,
                model_scores={},
                estimated_remaining_time=60.0,
                current_cost=Money(12.0, "USD"),
                errors=[],
            ),
        ]

        # Act
        results = await orchestration_service.get_all_active_tests()

        # Assert
        assert len(results) == 2
        assert results[0].test_id == test_id1
        assert results[1].test_id == test_id2
        assert results[0].progress == 0.5
        assert results[1].progress == 0.8

    @pytest.mark.asyncio
    async def test_get_processing_status(self, orchestration_service, mock_uow):
        """Test getting overall processing status."""
        # Arrange
        test_id = uuid4()
        await orchestration_service.schedule_sample_processing(test_id)

        mock_uow.tests.count_by_status.side_effect = lambda status: {
            TestStatus.DRAFT: 5,
            TestStatus.CONFIGURED: 3,
            TestStatus.RUNNING: 2,
            TestStatus.COMPLETED: 10,
            TestStatus.FAILED: 1,
            TestStatus.CANCELLED: 0,
        }.get(status, 0)

        # Act
        status = await orchestration_service.get_processing_status()

        # Assert
        assert status["active_processing_tasks"] == 1
        assert str(test_id) in status["active_test_ids"]
        assert status["test_status_counts"]["RUNNING"] == 2
        assert status["test_status_counts"]["COMPLETED"] == 10
        assert status["system_healthy"] is True

        # Clean up
        task = orchestration_service._active_processing_tasks[test_id]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_health_check_processing_tasks(self, orchestration_service):
        """Test health check of processing tasks."""
        # Arrange
        test_id = uuid4()
        await orchestration_service.schedule_sample_processing(test_id)

        # Act
        health_report = await orchestration_service.health_check_processing_tasks()

        # Assert
        assert health_report["healthy_tasks"] >= 1
        assert health_report["failed_tasks"] >= 0
        assert len(health_report["task_details"]) >= 1

        task_detail = next(
            (t for t in health_report["task_details"] if t["test_id"] == str(test_id)), None
        )
        assert task_detail is not None
        assert task_detail["status"] == "running"

        # Clean up
        task = orchestration_service._active_processing_tasks[test_id]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_cleanup_completed_tasks(self, orchestration_service):
        """Test cleanup of completed tasks."""
        # Arrange
        test_id = uuid4()

        # Create a completed task
        completed_task = AsyncMock()
        completed_task.done.return_value = True
        completed_task.cancelled.return_value = False
        completed_task.exception.return_value = None

        orchestration_service._active_processing_tasks[test_id] = completed_task

        # Act
        cleaned_count = await orchestration_service.cleanup_completed_tasks()

        # Assert
        assert cleaned_count == 1
        assert test_id not in orchestration_service._active_processing_tasks

    @pytest.mark.asyncio
    async def test_force_complete_test(self, orchestration_service, mock_use_cases, mock_uow):
        """Test force completing a test."""
        # Arrange
        test_id = uuid4()
        mock_test = MagicMock()
        mock_test.status.is_terminal.return_value = False
        mock_uow.tests.find_by_id.return_value = mock_test

        mock_use_cases["complete_test"].execute.return_value = TestResultDTO(
            test_id=test_id, status="COMPLETED", created_test=False
        )

        # Act
        result = await orchestration_service.force_complete_test(test_id, "Emergency completion")

        # Assert
        assert result["success"] is True
        assert result["test_id"] == test_id
        assert result["status"] == "COMPLETED"

        # Verify complete use case was called with force flag
        mock_use_cases["complete_test"].execute.assert_called_once_with(
            test_id, force_completion=True
        )

    @pytest.mark.asyncio
    async def test_process_samples_with_monitoring_success(
        self, orchestration_service, mock_use_cases
    ):
        """Test the private monitoring method with successful processing."""
        # Arrange
        test_id = uuid4()

        # Mock successful processing
        mock_use_cases["process_samples"].execute.return_value = {
            "success": True,
            "processed_samples": 10,
            "total_samples": 100,
            "failed_samples": 0,
        }

        # Mock monitoring showing incomplete progress
        mock_use_cases["monitor_test"].execute.side_effect = [
            TestMonitoringResultDTO(
                test_id=test_id,
                status="RUNNING",
                progress=0.1,  # 10% complete
                total_samples=100,
                evaluated_samples=10,
                model_scores={},
                estimated_remaining_time=450.0,
                current_cost=Money(2.0, "USD"),
                errors=[],
            ),
            TestMonitoringResultDTO(
                test_id=test_id,
                status="RUNNING",
                progress=1.0,  # 100% complete
                total_samples=100,
                evaluated_samples=100,
                model_scores={"gpt-4": 0.85, "claude-3": 0.82},
                estimated_remaining_time=0.0,
                current_cost=Money(20.0, "USD"),
                errors=[],
            ),
        ]

        mock_use_cases["complete_test"].execute.return_value = TestResultDTO(
            test_id=test_id, status="COMPLETED", created_test=False
        )

        # Act
        await orchestration_service.schedule_sample_processing(test_id)

        # Wait a bit for processing to start
        await asyncio.sleep(0.1)

        # Clean up
        await orchestration_service.cancel_test_processing(test_id)

    @pytest.mark.asyncio
    async def test_error_handling_in_orchestration(
        self, orchestration_service, valid_create_command, mock_use_cases
    ):
        """Test error handling in orchestration workflows."""
        # Arrange
        mock_use_cases["create_test"].execute.side_effect = Exception("Database error")

        # Act
        result = await orchestration_service.create_and_start_test(valid_create_command)

        # Assert
        assert result["success"] is False
        assert result["stage"] == "system_error"
        assert "System error" in str(result["errors"])

    @pytest.mark.asyncio
    async def test_concurrent_processing_limits(self, orchestration_service):
        """Test that processing respects concurrency limits."""
        # Arrange
        test_ids = [uuid4() for _ in range(5)]

        # Act - Schedule multiple processing tasks
        results = []
        for test_id in test_ids:
            result = await orchestration_service.schedule_sample_processing(test_id, batch_size=5)
            results.append(result)

        # Assert
        assert all(results)  # All should succeed
        assert len(orchestration_service._active_processing_tasks) == 5

        # Clean up
        for test_id in test_ids:
            task = orchestration_service._active_processing_tasks[test_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
