"""Tests for Test Management domain services."""

from typing import List
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.domain.test_management.entities.test import Test
from src.domain.test_management.entities.test_configuration import TestConfiguration
from src.domain.test_management.entities.test_sample import TestSample
from src.domain.test_management.events.test_events import TestCompleted, TestFailed, TestStarted
from src.domain.test_management.exceptions import BusinessRuleViolation, InvalidStateTransition
from src.domain.test_management.repositories.test_repository import TestRepository
from src.domain.test_management.services.test_orchestrator import TestOrchestrator
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.test_status import TestStatus


class MockEventPublisher:
    """Mock event publisher for testing."""

    def __init__(self):
        self.published_events = []

    async def publish(self, event) -> None:
        """Mock publish method."""
        self.published_events.append(event)

    def get_published_events(self) -> List:
        """Get all published events."""
        return self.published_events.copy()

    def get_events_of_type(self, event_type) -> List:
        """Get events of specific type."""
        return [event for event in self.published_events if isinstance(event, event_type)]


class TestTestOrchestrator:
    """Tests for TestOrchestrator domain service."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = Mock(spec=TestRepository)
        repo.save = AsyncMock()
        repo.find_by_id = AsyncMock()
        repo.find_by_status = AsyncMock()
        repo.find_active_tests = AsyncMock()
        return repo

    @pytest.fixture
    def mock_event_publisher(self):
        """Create mock event publisher."""
        return MockEventPublisher()

    @pytest.fixture
    def orchestrator(self, mock_repository, mock_event_publisher):
        """Create test orchestrator."""
        return TestOrchestrator(mock_repository, mock_event_publisher)

    @pytest.fixture
    def test_configuration(self):
        """Create test configuration."""
        return TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)

    @pytest.fixture
    def sample_test(self, test_configuration):
        """Create sample test with samples."""
        test = Test.create("Sample Test", test_configuration)
        for i in range(10):
            sample = TestSample(prompt=f"Test prompt {i}", difficulty=DifficultyLevel.MEDIUM)
            test.add_sample(sample)
        return test

    @pytest.mark.asyncio
    async def test_orchestrator_creation(self, mock_repository, mock_event_publisher):
        """Test creating test orchestrator."""
        orchestrator = TestOrchestrator(mock_repository, mock_event_publisher)
        assert orchestrator._repository == mock_repository
        assert orchestrator._event_publisher == mock_event_publisher

    @pytest.mark.asyncio
    async def test_create_test(self, orchestrator, mock_repository, mock_event_publisher):
        """Test creating new test through orchestrator."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)

        test = await orchestrator.create_test("New Test", config)

        assert test.name == "New Test"
        assert test.configuration == config
        assert test.status == TestStatus.DRAFT

        # Verify repository save was called
        mock_repository.save.assert_called_once_with(test)

        # Verify events were published
        published_events = mock_event_publisher.get_published_events()
        assert len(published_events) > 0

    @pytest.mark.asyncio
    async def test_create_test_with_invalid_configuration(self, orchestrator):
        """Test creating test with invalid configuration."""
        invalid_config = TestConfiguration(
            models=["gpt-4"], max_tokens=1000, temperature=0.7  # Only one model
        )

        with pytest.raises(BusinessRuleViolation):
            await orchestrator.create_test("Invalid Test", invalid_config)

    @pytest.mark.asyncio
    async def test_add_samples_to_test(self, orchestrator, mock_repository, sample_test):
        """Test adding samples to test through orchestrator."""
        mock_repository.find_by_id.return_value = sample_test

        new_samples = [
            TestSample(prompt="New prompt 1", difficulty=DifficultyLevel.EASY),
            TestSample(prompt="New prompt 2", difficulty=DifficultyLevel.HARD),
        ]

        await orchestrator.add_samples_to_test(sample_test.id, new_samples)

        # Verify samples were added
        assert len(sample_test.samples) == 12  # 10 original + 2 new

        # Verify repository save was called
        mock_repository.save.assert_called_once_with(sample_test)

    @pytest.mark.asyncio
    async def test_add_samples_to_non_existing_test(self, orchestrator, mock_repository):
        """Test adding samples to non-existing test."""
        mock_repository.find_by_id.return_value = None

        samples = [TestSample(prompt="Test", difficulty=DifficultyLevel.EASY)]

        with pytest.raises(ValueError, match="Test not found"):
            await orchestrator.add_samples_to_test(uuid4(), samples)

    @pytest.mark.asyncio
    async def test_configure_test(
        self, orchestrator, mock_repository, mock_event_publisher, sample_test
    ):
        """Test configuring test through orchestrator."""
        mock_repository.find_by_id.return_value = sample_test

        await orchestrator.configure_test(sample_test.id)

        assert sample_test.status == TestStatus.CONFIGURED
        mock_repository.save.assert_called_once_with(sample_test)

    @pytest.mark.asyncio
    async def test_configure_test_without_minimum_samples(
        self, orchestrator, mock_repository, test_configuration
    ):
        """Test configuring test without minimum samples."""
        test = Test.create("Test", test_configuration)
        # Add only 5 samples (less than minimum 10)
        for i in range(5):
            sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(sample)

        mock_repository.find_by_id.return_value = test

        with pytest.raises(BusinessRuleViolation):
            await orchestrator.configure_test(test.id)

    @pytest.mark.asyncio
    async def test_start_test(
        self, orchestrator, mock_repository, mock_event_publisher, sample_test
    ):
        """Test starting test through orchestrator."""
        sample_test.configure()
        mock_repository.find_by_id.return_value = sample_test

        await orchestrator.start_test(sample_test.id)

        assert sample_test.status == TestStatus.RUNNING
        mock_repository.save.assert_called_once_with(sample_test)

        # Verify test started event was published
        started_events = mock_event_publisher.get_events_of_type(TestStarted)
        assert len(started_events) > 0
        assert started_events[-1].test_id == sample_test.id

    @pytest.mark.asyncio
    async def test_start_test_from_wrong_state(self, orchestrator, mock_repository, sample_test):
        """Test starting test from wrong state."""
        # Test is in DRAFT state, should be CONFIGURED first
        mock_repository.find_by_id.return_value = sample_test

        with pytest.raises(InvalidStateTransition):
            await orchestrator.start_test(sample_test.id)

    @pytest.mark.asyncio
    async def test_complete_test(
        self, orchestrator, mock_repository, mock_event_publisher, sample_test
    ):
        """Test completing test through orchestrator."""
        sample_test.configure()
        sample_test.start()
        mock_repository.find_by_id.return_value = sample_test

        await orchestrator.complete_test(sample_test.id)

        assert sample_test.status == TestStatus.COMPLETED
        assert sample_test.completed_at is not None
        mock_repository.save.assert_called_once_with(sample_test)

        # Verify test completed event was published
        completed_events = mock_event_publisher.get_events_of_type(TestCompleted)
        assert len(completed_events) > 0
        assert completed_events[-1].test_id == sample_test.id

    @pytest.mark.asyncio
    async def test_fail_test(
        self, orchestrator, mock_repository, mock_event_publisher, sample_test
    ):
        """Test failing test through orchestrator."""
        sample_test.configure()
        sample_test.start()
        mock_repository.find_by_id.return_value = sample_test

        error_message = "Test execution failed due to timeout"
        await orchestrator.fail_test(sample_test.id, error_message)

        assert sample_test.status == TestStatus.FAILED
        assert sample_test.failure_reason == error_message
        mock_repository.save.assert_called_once_with(sample_test)

        # Verify test failed event was published
        failed_events = mock_event_publisher.get_events_of_type(TestFailed)
        assert len(failed_events) > 0
        assert failed_events[-1].test_id == sample_test.id
        assert failed_events[-1].reason == error_message

    @pytest.mark.asyncio
    async def test_cancel_test(self, orchestrator, mock_repository, sample_test):
        """Test canceling test through orchestrator."""
        mock_repository.find_by_id.return_value = sample_test

        await orchestrator.cancel_test(sample_test.id)

        assert sample_test.status == TestStatus.CANCELLED
        mock_repository.save.assert_called_once_with(sample_test)

    @pytest.mark.asyncio
    async def test_get_test_progress(self, orchestrator, mock_repository, sample_test):
        """Test getting test progress through orchestrator."""
        # Add some evaluations to samples
        sample_test.samples[0].add_evaluation_result("gpt-4", {"score": 0.8})
        sample_test.samples[1].add_evaluation_result("gpt-4", {"score": 0.9})

        mock_repository.find_by_id.return_value = sample_test

        progress = await orchestrator.get_test_progress(sample_test.id)

        assert progress == 0.2  # 2 out of 10 samples evaluated

    @pytest.mark.asyncio
    async def test_get_test_results_summary(self, orchestrator, mock_repository, sample_test):
        """Test getting test results summary."""
        # Add evaluations to all samples
        for i, sample in enumerate(sample_test.samples):
            sample.add_evaluation_result("gpt-4", {"score": 0.8 + (i * 0.01)})
            sample.add_evaluation_result("claude-3", {"score": 0.75 + (i * 0.01)})

        mock_repository.find_by_id.return_value = sample_test

        summary = await orchestrator.get_test_results_summary(sample_test.id)

        assert "overall_score" in summary
        assert "model_scores" in summary
        assert "gpt-4" in summary["model_scores"]
        assert "claude-3" in summary["model_scores"]
        assert "total_samples" in summary
        assert summary["total_samples"] == 10
        assert "evaluated_samples" in summary
        assert summary["evaluated_samples"] == 10

    @pytest.mark.asyncio
    async def test_find_active_tests(self, orchestrator, mock_repository):
        """Test finding active tests through orchestrator."""
        active_tests = [
            Mock(spec=Test, id=uuid4(), status=TestStatus.DRAFT),
            Mock(spec=Test, id=uuid4(), status=TestStatus.RUNNING),
        ]
        mock_repository.find_active_tests.return_value = active_tests

        result = await orchestrator.find_active_tests()

        assert len(result) == 2
        mock_repository.find_active_tests.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_completed_tests(self, orchestrator, mock_repository):
        """Test cleaning up old completed tests."""
        completed_tests = [
            Mock(spec=Test, id=uuid4(), status=TestStatus.COMPLETED),
            Mock(spec=Test, id=uuid4(), status=TestStatus.FAILED),
        ]
        mock_repository.find_by_status.return_value = completed_tests

        # Mock the repository delete method
        mock_repository.delete = AsyncMock()

        cleaned_count = await orchestrator.cleanup_old_tests(max_age_days=30)

        # Should find and delete old tests
        assert mock_repository.find_by_status.call_count >= 1

    @pytest.mark.asyncio
    async def test_validate_test_configuration(self, orchestrator):
        """Test validating test configuration through orchestrator."""
        valid_config = TestConfiguration(
            models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7
        )

        result = await orchestrator.validate_test_configuration(valid_config)
        assert result.is_valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_estimate_test_duration(self, orchestrator, sample_test):
        """Test estimating test duration."""
        duration_estimate = await orchestrator.estimate_test_duration(
            num_samples=len(sample_test.samples),
            num_models=len(sample_test.configuration.models),
            avg_response_time_seconds=2.0,
        )

        # Should return reasonable duration estimate
        assert duration_estimate > 0
        expected_base_time = len(sample_test.samples) * len(sample_test.configuration.models) * 2.0
        assert duration_estimate >= expected_base_time

    @pytest.mark.asyncio
    async def test_batch_create_samples(self, orchestrator):
        """Test batch creating samples with different difficulties."""
        prompts = ["Easy prompt", "Medium prompt", "Hard prompt"]
        difficulties = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]

        samples = await orchestrator.batch_create_samples(prompts, difficulties)

        assert len(samples) == 3
        for i, sample in enumerate(samples):
            assert sample.prompt == prompts[i]
            assert sample.difficulty == difficulties[i]

    @pytest.mark.asyncio
    async def test_event_publishing_on_state_changes(
        self, orchestrator, mock_repository, mock_event_publisher, sample_test
    ):
        """Test that domain events are published on state changes."""
        mock_repository.find_by_id.return_value = sample_test

        # Configure test
        await orchestrator.configure_test(sample_test.id)

        # Start test
        await orchestrator.start_test(sample_test.id)

        # Complete test
        await orchestrator.complete_test(sample_test.id)

        # Verify all events were published
        published_events = mock_event_publisher.get_published_events()

        # Should have at least TestStarted and TestCompleted events
        started_events = [e for e in published_events if isinstance(e, TestStarted)]
        completed_events = [e for e in published_events if isinstance(e, TestCompleted)]

        assert len(started_events) >= 1
        assert len(completed_events) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_test_operations(
        self, orchestrator, mock_repository, test_configuration
    ):
        """Test handling concurrent operations on different tests."""
        test1 = Test.create("Test 1", test_configuration)
        test2 = Test.create("Test 2", test_configuration)

        # Add samples to both tests
        for test in [test1, test2]:
            for i in range(10):
                sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
                test.add_sample(sample)

        # Mock repository to return appropriate test
        def mock_find_by_id(test_id):
            if test_id == test1.id:
                return test1
            elif test_id == test2.id:
                return test2
            return None

        mock_repository.find_by_id.side_effect = mock_find_by_id

        # Configure both tests concurrently
        await orchestrator.configure_test(test1.id)
        await orchestrator.configure_test(test2.id)

        # Both tests should be configured
        assert test1.status == TestStatus.CONFIGURED
        assert test2.status == TestStatus.CONFIGURED

        # Repository save should have been called for both
        assert mock_repository.save.call_count == 2
