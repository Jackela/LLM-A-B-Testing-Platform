"""Repository interface for model providers."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.model_provider import ModelProvider
from ..value_objects.provider_type import ProviderType


class ProviderRepository(ABC):
    """Abstract repository for model providers."""

    @abstractmethod
    async def save(self, provider: ModelProvider) -> None:
        """
        Save or update a model provider.

        Args:
            provider: The provider to save

        Raises:
            RepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    async def get_by_id(self, provider_id: UUID) -> Optional[ModelProvider]:
        """
        Get a provider by its ID.

        Args:
            provider_id: The unique identifier of the provider

        Returns:
            Optional[ModelProvider]: The provider if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[ModelProvider]:
        """
        Get a provider by its name.

        Args:
            name: The name of the provider

        Returns:
            Optional[ModelProvider]: The provider if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_all(self) -> List[ModelProvider]:
        """
        Get all providers.

        Returns:
            List[ModelProvider]: List of all providers
        """
        pass

    @abstractmethod
    async def delete(self, provider_id: UUID) -> bool:
        """
        Delete a provider by its ID.

        Args:
            provider_id: The unique identifier of the provider

        Returns:
            bool: True if provider was deleted, False if not found
        """
        pass

    @abstractmethod
    async def get_by_provider_type(self, provider_type: ProviderType) -> List[ModelProvider]:
        """
        Get providers by their type.

        Args:
            provider_type: The type of providers to retrieve

        Returns:
            List[ModelProvider]: List of providers of the specified type
        """
        pass

    @abstractmethod
    async def get_healthy_providers(self) -> List[ModelProvider]:
        """
        Get all providers that are currently healthy.

        Returns:
            List[ModelProvider]: List of healthy providers
        """
        pass

    @abstractmethod
    async def get_providers_supporting_model(self, model_id: str) -> List[ModelProvider]:
        """
        Get providers that support a specific model.

        Args:
            model_id: The ID of the model to search for

        Returns:
            List[ModelProvider]: List of providers supporting the model
        """
        pass

    @abstractmethod
    async def get_operational_providers(self) -> List[ModelProvider]:
        """
        Get providers that are operational (healthy or degraded).

        Returns:
            List[ModelProvider]: List of operational providers
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """
        Count total number of providers.

        Returns:
            int: Total number of providers
        """
        pass

    @abstractmethod
    async def count_by_type(self, provider_type: ProviderType) -> int:
        """
        Count providers by type.

        Args:
            provider_type: The type of providers to count

        Returns:
            int: Number of providers of the specified type
        """
        pass

    @abstractmethod
    async def exists(self, provider_id: UUID) -> bool:
        """
        Check if a provider exists.

        Args:
            provider_id: The unique identifier of the provider

        Returns:
            bool: True if provider exists, False otherwise
        """
        pass

    @abstractmethod
    async def exists_by_name(self, name: str) -> bool:
        """
        Check if a provider with the given name exists.

        Args:
            name: The name to check

        Returns:
            bool: True if provider with name exists, False otherwise
        """
        pass

    # Optional methods with default implementations that repositories can override

    async def get_by_names(self, names: List[str]) -> List[ModelProvider]:
        """
        Get providers by their names.

        Args:
            names: List of provider names

        Returns:
            List[ModelProvider]: List of found providers
        """
        providers = []
        for name in names:
            provider = await self.get_by_name(name)
            if provider:
                providers.append(provider)
        return providers

    async def get_by_ids(self, provider_ids: List[UUID]) -> List[ModelProvider]:
        """
        Get providers by their IDs.

        Args:
            provider_ids: List of provider IDs

        Returns:
            List[ModelProvider]: List of found providers
        """
        providers = []
        for provider_id in provider_ids:
            provider = await self.get_by_id(provider_id)
            if provider:
                providers.append(provider)
        return providers

    async def get_cheapest_providers(self, limit: int = 5) -> List[ModelProvider]:
        """
        Get providers with the cheapest models.

        Args:
            limit: Maximum number of providers to return

        Returns:
            List[ModelProvider]: List of providers sorted by cheapest model cost
        """
        all_providers = await self.get_all()

        # Sort by cheapest model cost
        def get_min_cost(provider: ModelProvider):
            cheapest = provider.get_cheapest_model()
            return cheapest.cost_per_input_token if cheapest else float("inf")

        sorted_providers = sorted(all_providers, key=get_min_cost)
        return sorted_providers[:limit]

    async def get_most_capable_providers(self, limit: int = 5) -> List[ModelProvider]:
        """
        Get providers with the most capable models.

        Args:
            limit: Maximum number of providers to return

        Returns:
            List[ModelProvider]: List of providers sorted by highest model capacity
        """
        all_providers = await self.get_all()

        # Sort by highest model capacity
        def get_max_capacity(provider: ModelProvider):
            most_capable = provider.get_most_capable_model()
            return most_capable.max_tokens if most_capable else 0

        sorted_providers = sorted(all_providers, key=get_max_capacity, reverse=True)
        return sorted_providers[:limit]
