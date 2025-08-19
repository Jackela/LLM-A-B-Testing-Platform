"""Test repository implementation."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from ....domain.test_management.entities.test import Test
from ....domain.test_management.repositories.test_repository import TestRepository
from ....domain.test_management.value_objects.test_status import TestStatus
from ..database import SessionFactory
from ..models.test_models import TestModel, TestSampleModel
from .mappers.test_mapper import TestMapper


class TestRepositoryImpl(TestRepository):
    """SQLAlchemy implementation of TestRepository."""

    def __init__(self, session_factory: SessionFactory):
        self.session_factory = session_factory
        self.mapper = TestMapper()

    async def save(self, test: Test) -> None:
        """Save test aggregate to database."""
        async with self.session_factory() as session:
            try:
                # Convert domain entity to model
                test_model = self.mapper.to_model(test)

                # Use merge for upsert behavior
                merged_model = await session.merge(test_model)
                await session.commit()

                # Update the domain entity ID if it was generated
                if test.id != merged_model.id:
                    test.id = merged_model.id

            except Exception:
                await session.rollback()
                raise

    async def find_by_id(self, test_id: UUID) -> Optional[Test]:
        """Find test by ID with optimized loading."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(TestModel)
                    .options(
                        selectinload(TestModel.samples),
                        selectinload(TestModel.model_responses),
                        selectinload(TestModel.evaluation_results),
                    )
                    .where(TestModel.id == test_id)
                )
                result = await session.execute(query)
                test_model = result.scalar_one_or_none()

                if test_model is None:
                    return None

                return self.mapper.to_domain(test_model)

            except Exception:
                await session.rollback()
                raise

    async def find_by_status(self, status: TestStatus) -> List[Test]:
        """Find tests by status with pagination support."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(TestModel)
                    .options(selectinload(TestModel.samples))
                    .where(TestModel.status == status)
                    .order_by(TestModel.created_at.desc())
                )
                result = await session.execute(query)
                test_models = result.scalars().all()

                return [self.mapper.to_domain(model) for model in test_models]

            except Exception:
                await session.rollback()
                raise

    async def find_active_tests(self) -> List[Test]:
        """Find active (non-terminal) tests."""
        async with self.session_factory() as session:
            try:
                # Active statuses are non-terminal ones
                active_statuses = [TestStatus.DRAFT, TestStatus.CONFIGURED, TestStatus.RUNNING]

                query = (
                    select(TestModel)
                    .options(selectinload(TestModel.samples))
                    .where(TestModel.status.in_(active_statuses))
                    .order_by(TestModel.created_at.desc())
                )
                result = await session.execute(query)
                test_models = result.scalars().all()

                return [self.mapper.to_domain(model) for model in test_models]

            except Exception:
                await session.rollback()
                raise

    async def delete(self, test_id: UUID) -> None:
        """Delete test by ID."""
        async with self.session_factory() as session:
            try:
                # Due to cascade delete, this will remove all related data
                query = delete(TestModel).where(TestModel.id == test_id)
                await session.execute(query)
                await session.commit()

            except Exception:
                await session.rollback()
                raise

    async def find_by_name_pattern(self, pattern: str) -> List[Test]:
        """Find tests by name pattern."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(TestModel)
                    .options(selectinload(TestModel.samples))
                    .where(TestModel.name.ilike(f"%{pattern}%"))
                    .order_by(TestModel.created_at.desc())
                )
                result = await session.execute(query)
                test_models = result.scalars().all()

                return [self.mapper.to_domain(model) for model in test_models]

            except Exception:
                await session.rollback()
                raise

    async def count_by_status(self, status: TestStatus) -> int:
        """Count tests by status."""
        async with self.session_factory() as session:
            try:
                query = select(func.count(TestModel.id)).where(TestModel.status == status)
                result = await session.execute(query)
                return result.scalar() or 0

            except Exception:
                await session.rollback()
                raise

    # Additional optimized query methods

    async def find_by_ids(self, test_ids: List[UUID]) -> List[Test]:
        """Find multiple tests by IDs efficiently."""
        if not test_ids:
            return []

        async with self.session_factory() as session:
            try:
                query = (
                    select(TestModel)
                    .options(selectinload(TestModel.samples))
                    .where(TestModel.id.in_(test_ids))
                    .order_by(TestModel.created_at.desc())
                )
                result = await session.execute(query)
                test_models = result.scalars().all()

                return [self.mapper.to_domain(model) for model in test_models]

            except Exception:
                await session.rollback()
                raise

    async def get_test_summary(self, test_id: UUID) -> Optional[dict]:
        """Get test summary without loading full entity."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(
                        TestModel.id,
                        TestModel.name,
                        TestModel.status,
                        TestModel.created_at,
                        TestModel.completed_at,
                        func.count(TestSampleModel.id).label("sample_count"),
                    )
                    .outerjoin(TestSampleModel, TestModel.id == TestSampleModel.test_id)
                    .where(TestModel.id == test_id)
                    .group_by(
                        TestModel.id,
                        TestModel.name,
                        TestModel.status,
                        TestModel.created_at,
                        TestModel.completed_at,
                    )
                )
                result = await session.execute(query)
                row = result.first()

                if row is None:
                    return None

                return {
                    "id": str(row.id),
                    "name": row.name,
                    "status": row.status.value,
                    "created_at": row.created_at.isoformat(),
                    "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                    "sample_count": row.sample_count or 0,
                }

            except Exception:
                await session.rollback()
                raise

    async def get_tests_summary(
        self, status_filter: Optional[TestStatus] = None, limit: int = 50, offset: int = 0
    ) -> List[dict]:
        """Get tests summary with pagination."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(
                        TestModel.id,
                        TestModel.name,
                        TestModel.status,
                        TestModel.created_at,
                        TestModel.completed_at,
                        func.count(TestSampleModel.id).label("sample_count"),
                    )
                    .outerjoin(TestSampleModel, TestModel.id == TestSampleModel.test_id)
                    .group_by(
                        TestModel.id,
                        TestModel.name,
                        TestModel.status,
                        TestModel.created_at,
                        TestModel.completed_at,
                    )
                    .order_by(TestModel.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )

                if status_filter:
                    query = query.where(TestModel.status == status_filter)

                result = await session.execute(query)
                rows = result.all()

                return [
                    {
                        "id": str(row.id),
                        "name": row.name,
                        "status": row.status.value,
                        "created_at": row.created_at.isoformat(),
                        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                        "sample_count": row.sample_count or 0,
                    }
                    for row in rows
                ]

            except Exception:
                await session.rollback()
                raise

    async def find_tests_with_samples_count(
        self, min_samples: int = 0, max_samples: Optional[int] = None
    ) -> List[Test]:
        """Find tests filtered by sample count."""
        async with self.session_factory() as session:
            try:
                # Subquery to count samples per test
                sample_counts = (
                    select(
                        TestSampleModel.test_id,
                        func.count(TestSampleModel.id).label("sample_count"),
                    )
                    .group_by(TestSampleModel.test_id)
                    .subquery()
                )

                query = (
                    select(TestModel)
                    .options(selectinload(TestModel.samples))
                    .join(sample_counts, TestModel.id == sample_counts.c.test_id)
                    .where(sample_counts.c.sample_count >= min_samples)
                )

                if max_samples is not None:
                    query = query.where(sample_counts.c.sample_count <= max_samples)

                query = query.order_by(TestModel.created_at.desc())

                result = await session.execute(query)
                test_models = result.scalars().all()

                return [self.mapper.to_domain(model) for model in test_models]

            except Exception:
                await session.rollback()
                raise

    async def update_test_status(self, test_id: UUID, new_status: TestStatus) -> bool:
        """Update test status efficiently without loading full entity."""
        async with self.session_factory() as session:
            try:
                query = update(TestModel).where(TestModel.id == test_id).values(status=new_status)
                result = await session.execute(query)
                await session.commit()

                return result.rowcount > 0

            except Exception:
                await session.rollback()
                raise

    async def exists(self, test_id: UUID) -> bool:
        """Check if test exists."""
        async with self.session_factory() as session:
            try:
                query = select(func.count(TestModel.id)).where(TestModel.id == test_id)
                result = await session.execute(query)
                count = result.scalar() or 0
                return count > 0

            except Exception:
                await session.rollback()
                raise
