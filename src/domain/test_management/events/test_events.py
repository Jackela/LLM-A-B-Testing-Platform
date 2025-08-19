"""Domain events for Test Management."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID


@dataclass(frozen=True)
class DomainEvent:
    """Base domain event."""

    event_id: UUID
    occurred_at: datetime
    event_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": str(self.event_id),
            "occurred_at": self.occurred_at.isoformat(),
            "event_type": self.event_type,
        }


@dataclass(frozen=True)
class TestCreated(DomainEvent):
    """Event raised when a new test is created."""

    test_id: UUID
    test_name: str

    def __post_init__(self):
        object.__setattr__(self, "event_type", "TestCreated")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "test_id": str(self.test_id),
                "test_name": self.test_name,
            }
        )
        return base_dict


@dataclass(frozen=True)
class TestConfigured(DomainEvent):
    """Event raised when a test is configured."""

    test_id: UUID
    sample_count: int
    model_count: int

    def __post_init__(self):
        object.__setattr__(self, "event_type", "TestConfigured")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "test_id": str(self.test_id),
                "sample_count": self.sample_count,
                "model_count": self.model_count,
            }
        )
        return base_dict


@dataclass(frozen=True)
class TestStarted(DomainEvent):
    """Event raised when a test is started."""

    test_id: UUID

    def __post_init__(self):
        object.__setattr__(self, "event_type", "TestStarted")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "test_id": str(self.test_id),
            }
        )
        return base_dict


@dataclass(frozen=True)
class TestCompleted(DomainEvent):
    """Event raised when a test is completed."""

    test_id: UUID
    duration_seconds: Optional[float] = None
    total_samples: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "event_type", "TestCompleted")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "test_id": str(self.test_id),
                "duration_seconds": self.duration_seconds,
                "total_samples": self.total_samples,
            }
        )
        return base_dict


@dataclass(frozen=True)
class TestFailed(DomainEvent):
    """Event raised when a test fails."""

    test_id: UUID
    reason: str
    error_details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        object.__setattr__(self, "event_type", "TestFailed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "test_id": str(self.test_id),
                "reason": self.reason,
                "error_details": self.error_details,
            }
        )
        return base_dict


@dataclass(frozen=True)
class TestCancelled(DomainEvent):
    """Event raised when a test is cancelled."""

    test_id: UUID
    reason: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "event_type", "TestCancelled")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "test_id": str(self.test_id),
                "reason": self.reason,
            }
        )
        return base_dict


@dataclass(frozen=True)
class SampleEvaluated(DomainEvent):
    """Event raised when a sample is evaluated."""

    test_id: UUID
    sample_index: int
    model_name: str
    score: float

    def __post_init__(self):
        object.__setattr__(self, "event_type", "SampleEvaluated")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "test_id": str(self.test_id),
                "sample_index": self.sample_index,
                "model_name": self.model_name,
                "score": self.score,
            }
        )
        return base_dict
