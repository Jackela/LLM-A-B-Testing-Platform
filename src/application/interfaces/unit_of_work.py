"""Unit of Work pattern interface for transaction management."""

from abc import ABC, abstractmethod
from typing import Any, AsyncContextManager

from ...domain.analytics.repositories.analytics_repository import AnalyticsRepository
from ...domain.evaluation.repositories.evaluation_repository import EvaluationRepository
from ...domain.model_provider.repositories.provider_repository import ProviderRepository
from ...domain.test_management.repositories.test_repository import TestRepository


class UnitOfWork(ABC):
    """Abstract Unit of Work for managing transactions across repositories."""

    tests: TestRepository
    providers: ProviderRepository
    evaluations: EvaluationRepository
    analytics: AnalyticsRepository

    async def __aenter__(self) -> "UnitOfWork":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        if exc_type is not None:
            await self.rollback()
        else:
            await self.commit()

    @abstractmethod
    async def commit(self) -> None:
        """Commit the transaction."""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the transaction."""
        pass

    @abstractmethod
    async def begin_transaction(self) -> None:
        """Begin a new transaction."""
        pass
