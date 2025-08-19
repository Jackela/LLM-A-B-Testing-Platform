"""Test repository interface for Test Management domain."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.test import Test
from ..value_objects.test_status import TestStatus


class TestRepository(ABC):
    """Repository interface for Test aggregate."""

    @abstractmethod
    async def save(self, test: Test) -> None:
        """Save test to storage."""
        pass

    @abstractmethod
    async def find_by_id(self, test_id: UUID) -> Optional[Test]:
        """Find test by ID."""
        pass

    @abstractmethod
    async def find_by_status(self, status: TestStatus) -> List[Test]:
        """Find tests by status."""
        pass

    @abstractmethod
    async def find_active_tests(self) -> List[Test]:
        """Find active (non-terminal) tests."""
        pass

    @abstractmethod
    async def delete(self, test_id: UUID) -> None:
        """Delete test by ID."""
        pass

    @abstractmethod
    async def find_by_name_pattern(self, pattern: str) -> List[Test]:
        """Find tests by name pattern."""
        pass

    @abstractmethod
    async def count_by_status(self, status: TestStatus) -> int:
        """Count tests by status."""
        pass
