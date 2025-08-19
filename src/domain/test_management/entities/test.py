"""Test aggregate root for Test Management domain."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..events.test_events import (
    DomainEvent,
    TestCancelled,
    TestCompleted,
    TestConfigured,
    TestCreated,
    TestFailed,
    TestStarted,
)
from ..exceptions import BusinessRuleViolation, InvalidStateTransition
from ..value_objects.test_status import TestStatus
from .test_configuration import TestConfiguration
from .test_sample import TestSample


@dataclass
class Test:
    """Test aggregate root managing A/B test lifecycle."""

    id: UUID
    name: str
    configuration: TestConfiguration
    status: TestStatus
    samples: List[TestSample]
    created_at: datetime
    completed_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    _domain_events: List[DomainEvent] = field(default_factory=list, init=False)

    @classmethod
    def create(cls, name: str, configuration: TestConfiguration) -> "Test":
        """Factory method for creating new tests."""
        # Validate configuration
        validation_result = configuration.validate()
        if not validation_result.is_valid:
            raise BusinessRuleViolation(
                f"Cannot create test with invalid configuration: {', '.join(validation_result.errors)}"
            )

        # Create test instance
        test = cls(
            id=uuid4(),
            name=name,
            configuration=configuration,
            status=TestStatus.DRAFT,
            samples=[],
            created_at=datetime.utcnow(),
        )

        # Add creation event
        test._add_domain_event(
            TestCreated(
                event_id=uuid4(),
                occurred_at=datetime.utcnow(),
                event_type="TestCreated",
                test_id=test.id,
                test_name=test.name,
            )
        )

        return test

    def add_sample(self, sample: TestSample) -> None:
        """Add sample to test."""
        if self.status != TestStatus.DRAFT:
            raise BusinessRuleViolation(
                f"Cannot add samples to test in {self.status.value} state. "
                "Samples can only be added in DRAFT state."
            )

        # Check for duplicate samples (by ID)
        if any(s.id == sample.id for s in self.samples):
            raise BusinessRuleViolation(f"Sample with ID {sample.id} already exists in test")

        self.samples.append(sample)

    def configure(self) -> None:
        """Transition test to CONFIGURED state."""
        if self.status != TestStatus.DRAFT:
            raise InvalidStateTransition(
                f"Cannot configure test in {self.status.value} state. "
                "Test must be in DRAFT state to be configured."
            )

        # Validate business rules for configuration
        if not self.samples:
            raise BusinessRuleViolation("Cannot configure test without samples")

        # Check sample size limits (business rule)
        if len(self.samples) < 10:
            raise BusinessRuleViolation("Sample size must be between 10 and 10,000")

        if len(self.samples) > 10000:
            raise BusinessRuleViolation("Sample size must be between 10 and 10,000")

        # Check that all samples are valid
        for i, sample in enumerate(self.samples):
            if not sample.prompt.strip():
                raise BusinessRuleViolation(f"Sample {i} has empty prompt")

        # Transition to configured state
        self.status = TestStatus.CONFIGURED

        # Add configured event
        self._add_domain_event(
            TestConfigured(
                event_id=uuid4(),
                occurred_at=datetime.utcnow(),
                event_type="TestConfigured",
                test_id=self.id,
                sample_count=len(self.samples),
                model_count=len(self.configuration.models),
            )
        )

    def start(self) -> None:
        """Start test execution."""
        if self.status != TestStatus.CONFIGURED:
            raise InvalidStateTransition(
                f"Cannot start test in {self.status.value} state. "
                "Test must be in CONFIGURED state to be started."
            )

        # Additional business rule validation
        if not self.samples:
            raise BusinessRuleViolation("Cannot start test without samples")

        # Transition to running state
        self.status = TestStatus.RUNNING

        # Add started event
        self._add_domain_event(
            TestStarted(
                event_id=uuid4(),
                occurred_at=datetime.utcnow(),
                event_type="TestStarted",
                test_id=self.id,
            )
        )

    def complete(self) -> None:
        """Complete test execution."""
        if self.status != TestStatus.RUNNING:
            raise InvalidStateTransition(
                f"Cannot complete test in {self.status.value} state. "
                "Test must be in RUNNING state to be completed."
            )

        # Set completion timestamp
        self.completed_at = datetime.utcnow()
        self.status = TestStatus.COMPLETED

        # Calculate duration
        duration = (self.completed_at - self.created_at).total_seconds()

        # Add completed event
        self._add_domain_event(
            TestCompleted(
                event_id=uuid4(),
                occurred_at=datetime.utcnow(),
                event_type="TestCompleted",
                test_id=self.id,
                duration_seconds=duration,
                total_samples=len(self.samples),
            )
        )

    def fail(self, reason: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Fail test execution."""
        if self.status not in [TestStatus.CONFIGURED, TestStatus.RUNNING]:
            raise InvalidStateTransition(
                f"Cannot fail test in {self.status.value} state. "
                "Test must be in CONFIGURED or RUNNING state to be failed."
            )

        # Set failure information
        self.failure_reason = reason
        self.completed_at = datetime.utcnow()
        self.status = TestStatus.FAILED

        # Add failed event
        self._add_domain_event(
            TestFailed(
                event_id=uuid4(),
                occurred_at=datetime.utcnow(),
                event_type="TestFailed",
                test_id=self.id,
                reason=reason,
                error_details=error_details,
            )
        )

    def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel test execution."""
        if self.status.is_terminal():
            raise InvalidStateTransition(
                f"Cannot cancel test in {self.status.value} state. "
                "Test is already in a terminal state."
            )

        # Set status and completion time
        self.status = TestStatus.CANCELLED
        self.completed_at = datetime.utcnow()

        # Add cancelled event
        self._add_domain_event(
            TestCancelled(
                event_id=uuid4(),
                occurred_at=datetime.utcnow(),
                event_type="TestCancelled",
                test_id=self.id,
                reason=reason,
            )
        )

    def calculate_progress(self) -> float:
        """Calculate test completion progress as percentage."""
        if not self.samples:
            return 0.0

        evaluated_count = sum(1 for sample in self.samples if sample.is_evaluated)
        return evaluated_count / len(self.samples)

    def calculate_overall_score(self) -> float:
        """Calculate overall weighted score across all samples."""
        if not self.samples:
            return 0.0

        evaluated_samples = [s for s in self.samples if s.is_evaluated]
        if not evaluated_samples:
            return 0.0

        total_weighted_score = sum(sample.get_weighted_score() for sample in evaluated_samples)
        return total_weighted_score / len(evaluated_samples)

    def get_model_scores(self) -> Dict[str, float]:
        """Get average scores for each model."""
        model_scores = {}
        evaluated_samples = [s for s in self.samples if s.is_evaluated]

        if not evaluated_samples:
            return {model: 0.0 for model in self.configuration.models}

        for model in self.configuration.models:
            model_sample_scores = []
            for sample in evaluated_samples:
                if sample.has_evaluation_for_model(model):
                    model_sample_scores.append(sample.get_weighted_score())

            if model_sample_scores:
                model_scores[model] = sum(model_sample_scores) / len(model_sample_scores)
            else:
                model_scores[model] = 0.0

        return model_scores

    def get_test_statistics(self) -> Dict[str, Any]:
        """Get comprehensive test statistics."""
        evaluated_samples = [s for s in self.samples if s.is_evaluated]
        progress = self.calculate_progress()

        stats = {
            "total_samples": len(self.samples),
            "evaluated_samples": len(evaluated_samples),
            "progress": progress,
            "overall_score": self.calculate_overall_score(),
            "model_scores": self.get_model_scores(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "duration_seconds": None,
        }

        # Add duration if completed
        if self.completed_at:
            duration = (self.completed_at - self.created_at).total_seconds()
            stats["duration_seconds"] = duration
            stats["completed_at"] = self.completed_at.isoformat()

        # Add failure information if failed
        if self.status == TestStatus.FAILED and self.failure_reason:
            stats["failure_reason"] = self.failure_reason

        # Add difficulty distribution
        difficulty_counts = {}
        for sample in self.samples:
            difficulty = sample.difficulty.value
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        stats["difficulty_distribution"] = difficulty_counts

        return stats

    def estimate_remaining_time(self, avg_evaluation_time_seconds: float = 2.0) -> float:
        """Estimate remaining time for test completion."""
        if self.status != TestStatus.RUNNING:
            return 0.0

        remaining_evaluations = 0
        for sample in self.samples:
            for model in self.configuration.models:
                if not sample.has_evaluation_for_model(model):
                    remaining_evaluations += 1

        return remaining_evaluations * avg_evaluation_time_seconds

    def _add_domain_event(self, event: DomainEvent) -> None:
        """Add domain event to internal list."""
        self._domain_events.append(event)

    def get_domain_events(self) -> List[DomainEvent]:
        """Get all domain events."""
        return self._domain_events.copy()

    def clear_domain_events(self) -> None:
        """Clear all domain events (typically called after publishing)."""
        self._domain_events.clear()

    def can_be_modified(self) -> bool:
        """Check if test can be modified."""
        return self.status.allows_modification()

    def is_active(self) -> bool:
        """Check if test is in an active (non-terminal) state."""
        return self.status.is_active()

    def __str__(self) -> str:
        """String representation of test."""
        return (
            f"Test(id={str(self.id)[:8]}..., name='{self.name}', "
            f"status={self.status.value}, samples={len(self.samples)})"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison based on ID."""
        if not isinstance(other, Test):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)
