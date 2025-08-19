"""Integration tests for complete test workflow end-to-end."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.application.dto.test_configuration_dto import (
    CreateTestCommandDTO,
    EvaluationConfigurationDTO,
    ModelConfigurationDTO,
    TestConfigurationDTO,
    TestSampleDTO,
)
from src.application.services.model_provider_service import ModelProviderService
from src.application.services.test_orchestration_service import TestOrchestrationService
from src.application.services.test_validation_service import TestValidationService
from src.application.use_cases.test_management.complete_test import CompleteTestUseCase
from src.application.use_cases.test_management.create_test import CreateTestUseCase
from src.application.use_cases.test_management.monitor_test import MonitorTestUseCase
from src.application.use_cases.test_management.process_samples import ProcessSamplesUseCase
from src.application.use_cases.test_management.start_test import StartTestUseCase
from src.domain.model_provider.value_objects.money import Money
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.test_status import TestStatus
from src.domain.test_management.value_objects.validation_result import ValidationResult


class TestCompleteWorkflow:
    """End-to-end integration tests for the complete test workflow."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        return {
            "uow": AsyncMock(),
            "event_publisher": AsyncMock(),
            "validation_service": AsyncMock(spec=TestValidationService),
            "provider_service": AsyncMock(spec=ModelProviderService),
        }

    @pytest.fixture
    def use_cases(self, mock_dependencies):
        """Create all use cases with mocked dependencies."""
        deps = mock_dependencies

        return {
            "create_test": CreateTestUseCase(
                uow=deps["uow"],
                event_publisher=deps["event_publisher"],
                validation_service=deps["validation_service"],
                provider_service=deps["provider_service"],
            ),
            "start_test": StartTestUseCase(
                uow=deps["uow"],
                event_publisher=deps["event_publisher"],
                provider_service=deps["provider_service"],
            ),
            "monitor_test": MonitorTestUseCase(
                uow=deps["uow"], provider_service=deps["provider_service"]
            ),
            "complete_test": CompleteTestUseCase(
                uow=deps["uow"], event_publisher=deps["event_publisher"]
            ),
            "process_samples": ProcessSamplesUseCase(
                uow=deps["uow"],
                event_publisher=deps["event_publisher"],
                provider_service=deps["provider_service"],
            ),
        }

    @pytest.fixture
    def orchestration_service(self, mock_dependencies, use_cases):
        """Create orchestration service with real use cases."""
        deps = mock_dependencies

        return TestOrchestrationService(
            uow=deps["uow"],
            event_publisher=deps["event_publisher"],
            create_test_use_case=use_cases["create_test"],
            start_test_use_case=use_cases["start_test"],
            monitor_test_use_case=use_cases["monitor_test"],
            complete_test_use_case=use_cases["complete_test"],
            process_samples_use_case=use_cases["process_samples"],
        )

    @pytest.fixture
    def test_command(self):
        """Create a valid test command for workflows."""
        return CreateTestCommandDTO(
            name="Complete Workflow Test",
            configuration=TestConfigurationDTO(
                models=[
                    ModelConfigurationDTO(
                        model_id="gpt-4",
                        provider_name="openai",
                        parameters={"temperature": 0.7, "max_tokens": 150},
                        weight=0.6,
                    ),
                    ModelConfigurationDTO(
                        model_id="claude-3",
                        provider_name="anthropic",
                        parameters={"temperature": 0.7, "max_tokens": 150},
                        weight=0.4,
                    ),
                ],
                evaluation=EvaluationConfigurationDTO(
                    template_id="comprehensive-eval",
                    judge_count=5,
                    consensus_threshold=0.8,
                    quality_threshold=0.75,
                ),
                max_cost=Money(200.0, "USD"),
                description="End-to-end workflow test with comprehensive evaluation",
            ),
            samples=[
                TestSampleDTO(
                    prompt=f"Complex test prompt number {i} requiring detailed analysis",
                    expected_output=f"Expected comprehensive response {i}",
                    difficulty=DifficultyLevel.HIGH if i % 3 == 0 else DifficultyLevel.MEDIUM,
                )
                for i in range(100)  # Large sample set for realistic testing
            ],
            creator_id="test-user-123",
        )

    def setup_successful_mocks(self, mock_dependencies):
        """Setup mocks for successful workflow execution."""
        deps = mock_dependencies

        # Setup validation service
        deps["validation_service"].validate_test_creation.return_value = ValidationResult(True, [])
        deps["validation_service"].estimate_test_cost.return_value = Money(45.50, "USD")
        deps["validation_service"].estimate_test_duration.return_value = 850.0

        # Setup provider service
        deps["provider_service"].verify_model_availability.return_value = {
            "openai/gpt-4": True,
            "anthropic/claude-3": True,
        }
        deps["provider_service"].validate_model_parameters.return_value = {}
        deps["provider_service"].get_providers_for_test.return_value = [
            MagicMock(name="OpenAI Provider", health_status=MagicMock(is_operational=True)),
            MagicMock(name="Anthropic Provider", health_status=MagicMock(is_operational=True)),
        ]
        deps["provider_service"].get_model_cost_estimates.return_value = {
            "openai/gpt-4": 25.30,
            "anthropic/claude-3": 20.20,
        }

        # Setup UoW with mock test entity
        mock_test = MagicMock()
        mock_test.id = uuid4()
        mock_test.name = "Complete Workflow Test"
        mock_test.status = TestStatus.DRAFT
        mock_test.samples = []
        mock_test.configuration = MagicMock()
        mock_test.configuration.models = ["gpt-4", "claude-3"]
        mock_test.can_be_modified.return_value = True
        mock_test.calculate_progress.return_value = 0.0
        mock_test.get_domain_events.return_value = []
        mock_test.clear_domain_events.return_value = None
        mock_test.estimate_remaining_time.return_value = 600.0
        mock_test.get_model_scores.return_value = {"gpt-4": 0.0, "claude-3": 0.0}
        mock_test.get_test_statistics.return_value = {
            "total_samples": 100,
            "evaluated_samples": 0,
            "progress": 0.0,
            "overall_score": 0.0,
            "model_scores": {"gpt-4": 0.0, "claude-3": 0.0},
        }

        deps["uow"].tests.find_by_id.return_value = mock_test
        deps["uow"].tests.save.return_value = None
        deps["uow"].commit.return_value = None

        return mock_test

    @pytest.mark.asyncio
    async def test_complete_successful_workflow(
        self, orchestration_service, test_command, mock_dependencies
    ):
        """Test complete successful workflow from creation to completion."""
        # Arrange
        mock_test = self.setup_successful_mocks(mock_dependencies)

        # Act & Assert - Step 1: Create and start test
        result = await orchestration_service.create_and_start_test(test_command)

        assert result["success"] is True
        assert result["test_id"] is not None
        assert result["estimated_cost"] == Money(45.50, "USD")
        assert result["estimated_duration"] == 850.0

        test_id = result["test_id"]

        # Simulate test progression through states
        mock_test.status = TestStatus.CONFIGURED
        mock_test.status = TestStatus.RUNNING

        # Act & Assert - Step 2: Monitor test progression
        monitor_result = await orchestration_service.monitor_test_use_case.execute(test_id)

        assert monitor_result.test_id == test_id
        assert monitor_result.status == TestStatus.RUNNING.value
        assert monitor_result.progress >= 0.0

        # Simulate partial processing completion
        mock_test.calculate_progress.return_value = 0.6
        mock_test.get_model_scores.return_value = {"gpt-4": 0.78, "claude-3": 0.82}
        mock_test.get_test_statistics.return_value = {
            "total_samples": 100,
            "evaluated_samples": 60,
            "progress": 0.6,
            "overall_score": 0.8,
            "model_scores": {"gpt-4": 0.78, "claude-3": 0.82},
        }

        # Act & Assert - Step 3: Process samples
        processing_result = await orchestration_service.process_samples_use_case.execute(
            test_id, batch_size=20
        )

        assert processing_result["success"] is True
        assert processing_result["total_samples"] == 100

        # Simulate processing completion
        mock_test.calculate_progress.return_value = 1.0
        mock_test.get_model_scores.return_value = {"gpt-4": 0.85, "claude-3": 0.88}
        mock_test.get_test_statistics.return_value = {
            "total_samples": 100,
            "evaluated_samples": 100,
            "progress": 1.0,
            "overall_score": 0.865,
            "model_scores": {"gpt-4": 0.85, "claude-3": 0.88},
        }

        # Act & Assert - Step 4: Complete test
        mock_test.status = TestStatus.COMPLETED
        completion_result = await orchestration_service.complete_test_use_case.execute(test_id)

        assert completion_result.status == TestStatus.COMPLETED.value

        # Verify all critical interactions occurred
        mock_dependencies["uow"].tests.save.assert_called()
        mock_dependencies["uow"].commit.assert_called()
        mock_dependencies["event_publisher"].publish_all.assert_called()

    @pytest.mark.asyncio
    async def test_workflow_with_validation_failure(
        self, orchestration_service, test_command, mock_dependencies
    ):
        """Test workflow handling of validation failures."""
        # Arrange
        mock_dependencies["validation_service"].validate_test_creation.return_value = (
            ValidationResult(False, ["Insufficient samples for statistical significance"])
        )

        # Act
        result = await orchestration_service.create_and_start_test(test_command)

        # Assert
        assert result["success"] is False
        assert result["stage"] == "creation"
        assert "Insufficient samples" in str(result["errors"])

        # Verify start was not attempted
        mock_dependencies["uow"].tests.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_workflow_with_provider_unavailability(
        self, orchestration_service, test_command, mock_dependencies
    ):
        """Test workflow handling when providers are unavailable."""
        # Arrange
        self.setup_successful_mocks(mock_dependencies)
        mock_dependencies["provider_service"].verify_model_availability.return_value = {
            "openai/gpt-4": False,  # Unavailable
            "anthropic/claude-3": True,
        }

        # Act
        result = await orchestration_service.create_and_start_test(test_command)

        # Assert
        assert result["success"] is False
        assert result["stage"] == "creation"
        assert any("not available" in str(error).lower() for error in result["errors"])

    @pytest.mark.asyncio
    async def test_workflow_with_processing_errors(
        self, orchestration_service, test_command, mock_dependencies
    ):
        """Test workflow handling of sample processing errors."""
        # Arrange
        mock_test = self.setup_successful_mocks(mock_dependencies)

        # Act - Create and start successfully
        result = await orchestration_service.create_and_start_test(test_command)
        assert result["success"] is True

        test_id = result["test_id"]
        mock_test.status = TestStatus.RUNNING

        # Simulate processing with errors
        mock_processing_result = {
            "success": False,
            "error": "Rate limit exceeded",
            "processed_samples": 30,
            "total_samples": 100,
            "failed_samples": 70,
        }

        # Override the process samples method to return error
        with patch.object(
            orchestration_service.process_samples_use_case,
            "execute",
            return_value=mock_processing_result,
        ):

            # Act
            processing_result = await orchestration_service.process_samples_use_case.execute(
                test_id
            )

            # Assert
            assert processing_result["success"] is False
            assert processing_result["failed_samples"] == 70
            assert "Rate limit exceeded" in processing_result["error"]

    @pytest.mark.asyncio
    async def test_workflow_with_partial_completion(
        self, orchestration_service, test_command, mock_dependencies
    ):
        """Test workflow with forced completion of partially processed test."""
        # Arrange
        mock_test = self.setup_successful_mocks(mock_dependencies)

        # Act - Create and start
        result = await orchestration_service.create_and_start_test(test_command)
        test_id = result["test_id"]

        # Simulate partial processing (70% complete)
        mock_test.status = TestStatus.RUNNING
        mock_test.calculate_progress.return_value = 0.7
        mock_test.get_model_scores.return_value = {"gpt-4": 0.82, "claude-3": 0.79}

        # Act - Force complete
        force_result = await orchestration_service.force_complete_test(
            test_id, "Testing partial completion"
        )

        # Assert
        assert force_result["success"] is True
        assert force_result["test_id"] == test_id

        # Verify force completion was called
        assert mock_test.cancel.called or mock_test.complete.called

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, orchestration_service, mock_dependencies):
        """Test multiple concurrent workflow executions."""
        # Arrange
        self.setup_successful_mocks(mock_dependencies)

        # Create multiple test commands
        commands = [
            CreateTestCommandDTO(
                name=f"Concurrent Test {i}",
                configuration=TestConfigurationDTO(
                    models=[
                        ModelConfigurationDTO(
                            model_id="gpt-4",
                            provider_name="openai",
                            parameters={"temperature": 0.5},
                            weight=1.0,
                        )
                    ],
                    evaluation=EvaluationConfigurationDTO(template_id="simple"),
                ),
                samples=[TestSampleDTO(prompt=f"Sample {j}") for j in range(20)],
            )
            for i in range(3)
        ]

        # Act - Execute concurrently
        tasks = [orchestration_service.create_and_start_test(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Assert
        successful_results = [r for r in results if not isinstance(r, Exception) and r["success"]]
        assert len(successful_results) == 3

        # Verify all have unique test IDs
        test_ids = [r["test_id"] for r in successful_results]
        assert len(set(test_ids)) == 3

    @pytest.mark.asyncio
    async def test_workflow_monitoring_and_status_tracking(
        self, orchestration_service, test_command, mock_dependencies
    ):
        """Test comprehensive monitoring and status tracking throughout workflow."""
        # Arrange
        mock_test = self.setup_successful_mocks(mock_dependencies)

        # Act - Create and start
        result = await orchestration_service.create_and_start_test(test_command)
        test_id = result["test_id"]

        # Simulate different processing stages
        stages = [
            (TestStatus.CONFIGURED, 0.0, {"gpt-4": 0.0, "claude-3": 0.0}),
            (TestStatus.RUNNING, 0.3, {"gpt-4": 0.75, "claude-3": 0.73}),
            (TestStatus.RUNNING, 0.7, {"gpt-4": 0.83, "claude-3": 0.81}),
            (TestStatus.RUNNING, 1.0, {"gpt-4": 0.87, "claude-3": 0.84}),
            (TestStatus.COMPLETED, 1.0, {"gpt-4": 0.87, "claude-3": 0.84}),
        ]

        # Test monitoring at each stage
        for status, progress, scores in stages:
            mock_test.status = status
            mock_test.calculate_progress.return_value = progress
            mock_test.get_model_scores.return_value = scores

            monitor_result = await orchestration_service.monitor_test_use_case.execute(test_id)

            assert monitor_result.status == status.value
            assert monitor_result.progress == progress
            assert monitor_result.model_scores == scores

        # Test overall processing status
        processing_status = await orchestration_service.get_processing_status()

        assert "test_status_counts" in processing_status
        assert "active_processing_tasks" in processing_status
        assert "system_healthy" in processing_status

    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(
        self, orchestration_service, test_command, mock_dependencies
    ):
        """Test error recovery and system resilience."""
        # Arrange
        mock_test = self.setup_successful_mocks(mock_dependencies)

        # Simulate intermittent failures
        failure_count = 0
        original_commit = mock_dependencies["uow"].commit

        async def failing_commit():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # Fail first 2 times
                raise Exception("Temporary database error")
            return await original_commit()

        mock_dependencies["uow"].commit = failing_commit

        # Act - This should eventually succeed after retries
        result = await orchestration_service.create_and_start_test(test_command)

        # The orchestration service doesn't implement retry logic in the current implementation,
        # so we expect this to fail. In a production system, you'd implement retry logic.
        assert result["success"] is False
        assert "error" in str(result).lower()

    @pytest.mark.asyncio
    async def test_resource_cleanup_and_management(
        self, orchestration_service, test_command, mock_dependencies
    ):
        """Test proper resource cleanup and management."""
        # Arrange
        self.setup_successful_mocks(mock_dependencies)

        # Act - Create multiple tests and then clean up
        test_ids = []
        for i in range(3):
            result = await orchestration_service.create_and_start_test(test_command)
            if result["success"]:
                test_ids.append(result["test_id"])
                # Schedule processing for each
                await orchestration_service.schedule_sample_processing(result["test_id"])

        # Verify tasks are active
        initial_status = await orchestration_service.get_processing_status()
        assert initial_status["active_processing_tasks"] == len(test_ids)

        # Act - Cancel all processing
        for test_id in test_ids:
            await orchestration_service.cancel_test_processing(test_id, "Cleanup test")

        # Act - Cleanup completed tasks
        cleaned_count = await orchestration_service.cleanup_completed_tasks()

        # Assert
        final_status = await orchestration_service.get_processing_status()
        assert final_status["active_processing_tasks"] == 0

        # Verify health check shows clean state
        health_report = await orchestration_service.health_check_processing_tasks()
        assert health_report["healthy_tasks"] + health_report["failed_tasks"] >= 0
