"""Domain event publisher interface for publishing domain events."""

from abc import ABC, abstractmethod
from typing import List

from ...domain.test_management.events.test_events import DomainEvent


class DomainEventPublisher(ABC):
    """Interface for publishing domain events."""

    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish a single domain event."""
        pass

    @abstractmethod
    async def publish_all(self, events: List[DomainEvent]) -> None:
        """Publish multiple domain events."""
        pass
