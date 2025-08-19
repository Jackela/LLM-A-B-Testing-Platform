"""Test status value object for Test Management domain."""

from enum import Enum
from typing import Set


class TestStatus(Enum):
    """Test lifecycle status enumeration."""

    DRAFT = "DRAFT"
    CONFIGURED = "CONFIGURED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def can_transition_to(self, target_status: "TestStatus") -> bool:
        """Check if transition to target status is allowed."""
        valid_transitions = self._get_valid_transitions()
        return target_status in valid_transitions

    def _get_valid_transitions(self) -> Set["TestStatus"]:
        """Get valid transitions from current status."""
        transition_map = {
            TestStatus.DRAFT: {TestStatus.CONFIGURED, TestStatus.CANCELLED},
            TestStatus.CONFIGURED: {TestStatus.DRAFT, TestStatus.RUNNING, TestStatus.CANCELLED},
            TestStatus.RUNNING: {TestStatus.COMPLETED, TestStatus.FAILED, TestStatus.CANCELLED},
            TestStatus.COMPLETED: set(),  # Terminal state
            TestStatus.FAILED: set(),  # Terminal state
            TestStatus.CANCELLED: set(),  # Terminal state
        }
        return transition_map.get(self, set())

    def is_terminal(self) -> bool:
        """Check if this is a terminal status (no further transitions allowed)."""
        return self in {TestStatus.COMPLETED, TestStatus.FAILED, TestStatus.CANCELLED}

    def allows_modification(self) -> bool:
        """Check if test can be modified in this status."""
        return self == TestStatus.DRAFT

    def is_active(self) -> bool:
        """Check if test is in an active (non-terminal) state."""
        return not self.is_terminal()

    def __str__(self) -> str:
        """String representation of status."""
        return self.value

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"TestStatus.{self.name}"
