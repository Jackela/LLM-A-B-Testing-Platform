"""Provider repository implementation."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ....domain.model_provider.entities.model_provider import ModelProvider
from ....domain.model_provider.repositories.provider_repository import ProviderRepository
from ....domain.model_provider.value_objects.health_status import HealthStatus
from ....domain.model_provider.value_objects.provider_type import ProviderType
from ..database import SessionFactory
from ..models.provider_models import ModelConfigModel, ModelProviderModel
from .mappers.provider_mapper import ProviderMapper


class ProviderRepositoryImpl(ProviderRepository):
    """SQLAlchemy implementation of ProviderRepository."""

    def __init__(self, session_factory: SessionFactory):
        self.session_factory = session_factory
        self.mapper = ProviderMapper()

    async def save(self, provider: ModelProvider) -> None:
        """Save or update a model provider."""
        async with self.session_factory() as session:
            try:
                # Convert domain entity to model
                provider_model = self.mapper.to_model(provider)

                # Use merge for upsert behavior
                merged_model = await session.merge(provider_model)
                await session.commit()

                # Update the domain entity ID if it was generated
                if provider.id != merged_model.id:
                    provider.id = merged_model.id

            except Exception:
                await session.rollback()
                raise

    async def get_by_id(self, provider_id: UUID) -> Optional[ModelProvider]:
        """Get a provider by its ID."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(ModelProviderModel)
                    .options(selectinload(ModelProviderModel.supported_models))
                    .where(ModelProviderModel.id == provider_id)
                )
                result = await session.execute(query)
                provider_model = result.scalar_one_or_none()

                if provider_model is None:
                    return None

                return self.mapper.to_domain(provider_model)

            except Exception:
                await session.rollback()
                raise

    async def get_by_name(self, name: str) -> Optional[ModelProvider]:
        """Get a provider by its name."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(ModelProviderModel)
                    .options(selectinload(ModelProviderModel.supported_models))
                    .where(ModelProviderModel.name == name)
                )
                result = await session.execute(query)
                provider_model = result.scalar_one_or_none()

                if provider_model is None:
                    return None

                return self.mapper.to_domain(provider_model)

            except Exception:
                await session.rollback()
                raise

    async def get_all(self) -> List[ModelProvider]:
        """Get all providers."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(ModelProviderModel)
                    .options(selectinload(ModelProviderModel.supported_models))
                    .order_by(ModelProviderModel.name)
                )
                result = await session.execute(query)
                provider_models = result.scalars().all()

                return [self.mapper.to_domain(model) for model in provider_models]

            except Exception:
                await session.rollback()
                raise

    async def delete(self, provider_id: UUID) -> bool:
        """Delete a provider by its ID."""
        async with self.session_factory() as session:
            try:
                query = delete(ModelProviderModel).where(ModelProviderModel.id == provider_id)
                result = await session.execute(query)
                await session.commit()

                return result.rowcount > 0

            except Exception:
                await session.rollback()
                raise

    async def get_by_provider_type(self, provider_type: ProviderType) -> List[ModelProvider]:
        """Get providers by their type."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(ModelProviderModel)
                    .options(selectinload(ModelProviderModel.supported_models))
                    .where(ModelProviderModel.provider_type == provider_type)
                    .order_by(ModelProviderModel.name)
                )
                result = await session.execute(query)
                provider_models = result.scalars().all()

                return [self.mapper.to_domain(model) for model in provider_models]

            except Exception:
                await session.rollback()
                raise

    async def get_healthy_providers(self) -> List[ModelProvider]:
        """Get all providers that are currently healthy."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(ModelProviderModel)
                    .options(selectinload(ModelProviderModel.supported_models))
                    .where(ModelProviderModel.health_status == HealthStatus.HEALTHY)
                    .order_by(ModelProviderModel.name)
                )
                result = await session.execute(query)
                provider_models = result.scalars().all()

                return [self.mapper.to_domain(model) for model in provider_models]

            except Exception:
                await session.rollback()
                raise

    async def get_providers_supporting_model(self, model_id: str) -> List[ModelProvider]:
        """Get providers that support a specific model."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(ModelProviderModel)
                    .options(selectinload(ModelProviderModel.supported_models))
                    .join(ModelConfigModel, ModelProviderModel.id == ModelConfigModel.provider_id)
                    .where(ModelConfigModel.model_id == model_id)
                    .order_by(ModelProviderModel.name)
                )
                result = await session.execute(query)
                provider_models = result.scalars().unique().all()

                return [self.mapper.to_domain(model) for model in provider_models]

            except Exception:
                await session.rollback()
                raise

    async def get_operational_providers(self) -> List[ModelProvider]:
        """Get providers that are operational (healthy or degraded)."""
        async with self.session_factory() as session:
            try:
                operational_statuses = [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

                query = (
                    select(ModelProviderModel)
                    .options(selectinload(ModelProviderModel.supported_models))
                    .where(ModelProviderModel.health_status.in_(operational_statuses))
                    .order_by(ModelProviderModel.name)
                )
                result = await session.execute(query)
                provider_models = result.scalars().all()

                return [self.mapper.to_domain(model) for model in provider_models]

            except Exception:
                await session.rollback()
                raise

    async def count(self) -> int:
        """Count total number of providers."""
        async with self.session_factory() as session:
            try:
                query = select(func.count(ModelProviderModel.id))
                result = await session.execute(query)
                return result.scalar() or 0

            except Exception:
                await session.rollback()
                raise

    async def count_by_type(self, provider_type: ProviderType) -> int:
        """Count providers by type."""
        async with self.session_factory() as session:
            try:
                query = select(func.count(ModelProviderModel.id)).where(
                    ModelProviderModel.provider_type == provider_type
                )
                result = await session.execute(query)
                return result.scalar() or 0

            except Exception:
                await session.rollback()
                raise

    async def exists(self, provider_id: UUID) -> bool:
        """Check if a provider exists."""
        async with self.session_factory() as session:
            try:
                query = select(func.count(ModelProviderModel.id)).where(
                    ModelProviderModel.id == provider_id
                )
                result = await session.execute(query)
                count = result.scalar() or 0
                return count > 0

            except Exception:
                await session.rollback()
                raise

    async def exists_by_name(self, name: str) -> bool:
        """Check if a provider with the given name exists."""
        async with self.session_factory() as session:
            try:
                query = select(func.count(ModelProviderModel.id)).where(
                    ModelProviderModel.name == name
                )
                result = await session.execute(query)
                count = result.scalar() or 0
                return count > 0

            except Exception:
                await session.rollback()
                raise

    # Additional optimized methods

    async def update_health_status(self, provider_id: UUID, new_status: HealthStatus) -> bool:
        """Update provider health status efficiently."""
        async with self.session_factory() as session:
            try:
                query = (
                    update(ModelProviderModel)
                    .where(ModelProviderModel.id == provider_id)
                    .values(health_status=new_status)
                )
                result = await session.execute(query)
                await session.commit()

                return result.rowcount > 0

            except Exception:
                await session.rollback()
                raise

    async def get_providers_summary(self) -> List[dict]:
        """Get summary information for all providers."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(
                        ModelProviderModel.id,
                        ModelProviderModel.name,
                        ModelProviderModel.provider_type,
                        ModelProviderModel.health_status,
                        func.count(ModelConfigModel.id).label("model_count"),
                    )
                    .outerjoin(
                        ModelConfigModel, ModelProviderModel.id == ModelConfigModel.provider_id
                    )
                    .group_by(
                        ModelProviderModel.id,
                        ModelProviderModel.name,
                        ModelProviderModel.provider_type,
                        ModelProviderModel.health_status,
                    )
                    .order_by(ModelProviderModel.name)
                )
                result = await session.execute(query)
                rows = result.all()

                return [
                    {
                        "id": str(row.id),
                        "name": row.name,
                        "provider_type": row.provider_type.value,
                        "health_status": row.health_status.name,
                        "model_count": row.model_count or 0,
                        "is_operational": row.health_status.is_operational,
                    }
                    for row in rows
                ]

            except Exception:
                await session.rollback()
                raise

    async def find_by_model_categories(self, categories: List[str]) -> List[ModelProvider]:
        """Find providers that support models in specified categories."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(ModelProviderModel)
                    .options(selectinload(ModelProviderModel.supported_models))
                    .join(ModelConfigModel, ModelProviderModel.id == ModelConfigModel.provider_id)
                    .where(ModelConfigModel.model_category.in_(categories))
                    .order_by(ModelProviderModel.name)
                )
                result = await session.execute(query)
                provider_models = result.scalars().unique().all()

                return [self.mapper.to_domain(model) for model in provider_models]

            except Exception:
                await session.rollback()
                raise
