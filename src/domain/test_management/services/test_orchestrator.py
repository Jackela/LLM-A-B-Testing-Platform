"""Test orchestrator domain service for Test Management."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..entities.test import Test
from ..entities.test_configuration import TestConfiguration
from ..entities.test_sample import TestSample
from ..events.test_events import DomainEvent
from ..exceptions import TestNotFound
from ..repositories.test_repository import TestRepository
from ..value_objects.difficulty_level import DifficultyLevel
from ..value_objects.test_status import TestStatus
from ..value_objects.validation_result import ValidationResult


class TestOrchestrator:
    """Domain service for orchestrating test operations."""

    def __init__(self, repository: TestRepository, event_publisher):
        """Initialize orchestrator with dependencies."""
        self._repository = repository
        self._event_publisher = event_publisher

    async def create_test(self, name: str, configuration: TestConfiguration) -> Test:
        """Create a new test."""
        test = Test.create(name, configuration)
        await self._repository.save(test)
        await self._publish_domain_events(test)
        return test

    async def add_samples_to_test(self, test_id: UUID, samples: List[TestSample]) -> None:
        """Add samples to a test."""
        test = await self._repository.find_by_id(test_id)
        if not test:
            raise ValueError("Test not found")

        for sample in samples:
            test.add_sample(sample)

        await self._repository.save(test)

    async def configure_test(self, test_id: UUID) -> None:
        """Configure a test."""
        test = await self._repository.find_by_id(test_id)
        if not test:
            raise TestNotFound(f"Test with ID {test_id} not found")

        test.configure()
        await self._repository.save(test)
        await self._publish_domain_events(test)

    async def start_test(self, test_id: UUID) -> None:
        """Start test execution."""
        test = await self._repository.find_by_id(test_id)
        if not test:
            raise TestNotFound(f"Test with ID {test_id} not found")

        test.start()
        await self._repository.save(test)
        await self._publish_domain_events(test)

    async def complete_test(self, test_id: UUID) -> None:
        """Complete test execution."""
        test = await self._repository.find_by_id(test_id)
        if not test:
            raise TestNotFound(f"Test with ID {test_id} not found")

        test.complete()
        await self._repository.save(test)
        await self._publish_domain_events(test)

    async def fail_test(
        self, test_id: UUID, reason: str, error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Fail test execution."""
        test = await self._repository.find_by_id(test_id)
        if not test:
            raise TestNotFound(f"Test with ID {test_id} not found")

        test.fail(reason, error_details)
        await self._repository.save(test)
        await self._publish_domain_events(test)

    async def cancel_test(self, test_id: UUID, reason: Optional[str] = None) -> None:
        """Cancel test execution."""
        test = await self._repository.find_by_id(test_id)
        if not test:
            raise TestNotFound(f"Test with ID {test_id} not found")

        test.cancel(reason)
        await self._repository.save(test)
        await self._publish_domain_events(test)

    async def get_test_progress(self, test_id: UUID) -> float:
        """Get test progress as percentage."""
        test = await self._repository.find_by_id(test_id)
        if not test:
            raise TestNotFound(f"Test with ID {test_id} not found")

        return test.calculate_progress()

    async def get_test_results_summary(self, test_id: UUID) -> Dict[str, Any]:
        """Get comprehensive test results summary."""
        test = await self._repository.find_by_id(test_id)
        if not test:
            raise TestNotFound(f"Test with ID {test_id} not found")

        stats = test.get_test_statistics()

        return {
            "test_id": str(test_id),
            "test_name": test.name,
            "overall_score": stats["overall_score"],
            "model_scores": stats["model_scores"],
            "total_samples": stats["total_samples"],
            "evaluated_samples": stats["evaluated_samples"],
            "progress": stats["progress"],
            "status": stats["status"],
            "created_at": stats["created_at"],
            "duration_seconds": stats.get("duration_seconds"),
            "difficulty_distribution": stats["difficulty_distribution"],
        }

    async def find_active_tests(self) -> List[Test]:
        """Find all active tests."""
        return await self._repository.find_active_tests()

    async def cleanup_old_tests(self, max_age_days: int = 30) -> int:
        """Clean up old completed tests."""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned_count = 0

        # Find completed and failed tests
        completed_tests = await self._repository.find_by_status(TestStatus.COMPLETED)
        failed_tests = await self._repository.find_by_status(TestStatus.FAILED)
        cancelled_tests = await self._repository.find_by_status(TestStatus.CANCELLED)

        all_terminal_tests = completed_tests + failed_tests + cancelled_tests

        for test in all_terminal_tests:
            if test.completed_at and test.completed_at < cutoff_date:
                await self._repository.delete(test.id)
                cleaned_count += 1

        return cleaned_count

    async def validate_test_configuration(
        self, configuration: TestConfiguration
    ) -> ValidationResult:
        """Validate test configuration."""
        return configuration.validate()

    async def estimate_test_duration(
        self, num_samples: int, num_models: int, avg_response_time_seconds: float = 2.0
    ) -> float:
        """Estimate test duration in seconds."""
        total_evaluations = num_samples * num_models
        base_time = total_evaluations * avg_response_time_seconds

        # Add overhead factor (20%) for processing, queueing, etc.
        overhead_factor = 1.2

        return base_time * overhead_factor

    async def batch_create_samples(
        self, prompts: List[str], difficulties: List[DifficultyLevel]
    ) -> List[TestSample]:
        """Create multiple samples in batch."""
        if len(prompts) != len(difficulties):
            raise ValueError("Prompts and difficulties lists must have the same length")

        samples = []
        for prompt, difficulty in zip(prompts, difficulties):
            sample = TestSample(prompt=prompt, difficulty=difficulty)
            samples.append(sample)

        return samples

    async def _publish_domain_events(self, test: Test) -> None:
        """Publish domain events from the test."""
        events = test.get_domain_events()
        for event in events:
            await self._event_publisher.publish(event)
        test.clear_domain_events()
