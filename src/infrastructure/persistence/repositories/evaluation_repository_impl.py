"""Evaluation repository implementation."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ....domain.evaluation.entities.evaluation_result import EvaluationResult
from ....domain.evaluation.entities.evaluation_template import EvaluationTemplate
from ....domain.evaluation.entities.judge import Judge
from ....domain.evaluation.repositories.evaluation_repository import EvaluationRepository
from ....domain.evaluation.value_objects.calibration_data import CalibrationData
from ..database import SessionFactory
from ..models.evaluation_models import (
    CalibrationDataModel,
    DimensionModel,
    EvaluationResultModel,
    EvaluationTemplateModel,
    JudgeModel,
)
from .mappers.evaluation_mapper import EvaluationMapper


class EvaluationRepositoryImpl(EvaluationRepository):
    """SQLAlchemy implementation of EvaluationRepository."""

    def __init__(self, session_factory: SessionFactory):
        self.session_factory = session_factory
        self.mapper = EvaluationMapper()

    # Judge repository methods
    async def save_judge(self, judge: Judge) -> None:
        """Save or update judge."""
        async with self.session_factory() as session:
            try:
                judge_model = self.mapper.judge_to_model(judge)
                merged_model = await session.merge(judge_model)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_judge(self, judge_id: str) -> Optional[Judge]:
        """Get judge by ID."""
        async with self.session_factory() as session:
            try:
                query = select(JudgeModel).where(JudgeModel.id == judge_id)
                result = await session.execute(query)
                judge_model = result.scalar_one_or_none()

                return self.mapper.model_to_judge(judge_model) if judge_model else None
            except Exception:
                await session.rollback()
                raise

    async def get_judges(
        self,
        is_active: Optional[bool] = None,
        is_calibrated: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Judge]:
        """Get judges with optional filters."""
        async with self.session_factory() as session:
            try:
                query = select(JudgeModel).order_by(JudgeModel.name)

                if is_active is not None:
                    query = query.where(JudgeModel.is_active == is_active)
                if is_calibrated is not None:
                    query = query.where(JudgeModel.is_calibrated == is_calibrated)
                if limit:
                    query = query.limit(limit)
                if offset:
                    query = query.offset(offset)

                result = await session.execute(query)
                judge_models = result.scalars().all()

                return [self.mapper.model_to_judge(model) for model in judge_models]
            except Exception:
                await session.rollback()
                raise

    async def delete_judge(self, judge_id: str) -> None:
        """Delete judge."""
        async with self.session_factory() as session:
            try:
                query = delete(JudgeModel).where(JudgeModel.id == judge_id)
                await session.execute(query)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # Template repository methods
    async def save_template(self, template: EvaluationTemplate) -> None:
        """Save or update evaluation template."""
        async with self.session_factory() as session:
            try:
                template_model = self.mapper.template_to_model(template)
                merged_model = await session.merge(template_model)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_template(self, template_id: UUID) -> Optional[EvaluationTemplate]:
        """Get template by ID."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(EvaluationTemplateModel)
                    .options(selectinload(EvaluationTemplateModel.dimensions))
                    .where(EvaluationTemplateModel.id == template_id)
                )
                result = await session.execute(query)
                template_model = result.scalar_one_or_none()

                return self.mapper.model_to_template(template_model) if template_model else None
            except Exception:
                await session.rollback()
                raise

    async def get_templates(
        self,
        is_active: Optional[bool] = None,
        created_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[EvaluationTemplate]:
        """Get templates with optional filters."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(EvaluationTemplateModel)
                    .options(selectinload(EvaluationTemplateModel.dimensions))
                    .order_by(EvaluationTemplateModel.name)
                )

                if is_active is not None:
                    query = query.where(EvaluationTemplateModel.is_active == is_active)
                if created_by is not None:
                    query = query.where(EvaluationTemplateModel.created_by == created_by)
                if limit:
                    query = query.limit(limit)
                if offset:
                    query = query.offset(offset)

                result = await session.execute(query)
                template_models = result.scalars().all()

                return [self.mapper.model_to_template(model) for model in template_models]
            except Exception:
                await session.rollback()
                raise

    async def get_template_versions(self, name: str) -> List[EvaluationTemplate]:
        """Get all versions of a template by name."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(EvaluationTemplateModel)
                    .options(selectinload(EvaluationTemplateModel.dimensions))
                    .where(EvaluationTemplateModel.name == name)
                    .order_by(EvaluationTemplateModel.version.desc())
                )
                result = await session.execute(query)
                template_models = result.scalars().all()

                return [self.mapper.model_to_template(model) for model in template_models]
            except Exception:
                await session.rollback()
                raise

    async def delete_template(self, template_id: UUID) -> None:
        """Delete template."""
        async with self.session_factory() as session:
            try:
                query = delete(EvaluationTemplateModel).where(
                    EvaluationTemplateModel.id == template_id
                )
                await session.execute(query)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # Evaluation result repository methods
    async def save_evaluation_result(self, result: EvaluationResult) -> None:
        """Save or update evaluation result."""
        async with self.session_factory() as session:
            try:
                result_model = self.mapper.result_to_model(result)
                merged_model = await session.merge(result_model)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_evaluation_result(self, result_id: UUID) -> Optional[EvaluationResult]:
        """Get evaluation result by ID."""
        async with self.session_factory() as session:
            try:
                query = select(EvaluationResultModel).where(EvaluationResultModel.id == result_id)
                result = await session.execute(query)
                result_model = result.scalar_one_or_none()

                return self.mapper.model_to_result(result_model) if result_model else None
            except Exception:
                await session.rollback()
                raise

    async def get_evaluation_results(
        self,
        judge_id: Optional[str] = None,
        template_id: Optional[UUID] = None,
        is_successful: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[EvaluationResult]:
        """Get evaluation results with optional filters."""
        async with self.session_factory() as session:
            try:
                query = select(EvaluationResultModel).order_by(
                    EvaluationResultModel.created_at.desc()
                )

                if judge_id is not None:
                    query = query.where(EvaluationResultModel.judge_id == judge_id)
                if template_id is not None:
                    query = query.where(EvaluationResultModel.template_id == template_id)
                if is_successful is not None:
                    query = query.where(EvaluationResultModel.is_successful == is_successful)
                if limit:
                    query = query.limit(limit)
                if offset:
                    query = query.offset(offset)

                result = await session.execute(query)
                result_models = result.scalars().all()

                return [self.mapper.model_to_result(model) for model in result_models]
            except Exception:
                await session.rollback()
                raise

    async def get_results_for_consensus(
        self, prompt: str, response: str, template_id: UUID
    ) -> List[EvaluationResult]:
        """Get all evaluation results for specific prompt/response/template combination."""
        # Implementation would join with model_responses table to match prompt/response
        # This is a simplified version
        async with self.session_factory() as session:
            try:
                query = (
                    select(EvaluationResultModel)
                    .where(EvaluationResultModel.template_id == template_id)
                    .order_by(EvaluationResultModel.created_at.desc())
                )
                result = await session.execute(query)
                result_models = result.scalars().all()

                return [self.mapper.model_to_result(model) for model in result_models]
            except Exception:
                await session.rollback()
                raise

    async def delete_evaluation_result(self, result_id: UUID) -> None:
        """Delete evaluation result."""
        async with self.session_factory() as session:
            try:
                query = delete(EvaluationResultModel).where(EvaluationResultModel.id == result_id)
                await session.execute(query)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # Additional methods with placeholder implementations
    async def save_calibration_data(self, judge_id: str, calibration: CalibrationData) -> None:
        """Save calibration data for judge."""
        # Placeholder implementation
        pass

    async def get_calibration_data(self, judge_id: str) -> Optional[CalibrationData]:
        """Get calibration data for judge."""
        # Placeholder implementation
        return None

    async def get_calibration_history(
        self, judge_id: str, limit: Optional[int] = None
    ) -> List[CalibrationData]:
        """Get calibration history for judge."""
        # Placeholder implementation
        return []

    async def get_judge_performance_metrics(self, judge_id: str, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics for judge over specified period."""
        # Placeholder implementation
        return {}

    async def get_template_usage_stats(self, template_id: UUID, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for template over specified period."""
        # Placeholder implementation
        return {}

    async def get_consensus_statistics(
        self, template_id: Optional[UUID] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get consensus statistics over specified period."""
        # Placeholder implementation
        return {}

    async def get_quality_metrics(
        self, judge_id: Optional[str] = None, template_id: Optional[UUID] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get quality metrics over specified period."""
        # Placeholder implementation
        return {}
