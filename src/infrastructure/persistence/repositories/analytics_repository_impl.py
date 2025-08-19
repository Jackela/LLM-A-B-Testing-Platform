"""Analytics repository implementation."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ....domain.analytics.entities.analysis_result import AnalysisResult
from ....domain.analytics.entities.model_performance import ModelPerformance
from ....domain.analytics.entities.statistical_test import StatisticalTest
from ....domain.analytics.repositories.analytics_repository import AnalyticsRepository
from ....domain.analytics.value_objects.insight import Insight
from ..database import SessionFactory
from ..models.analytics_models import (
    AnalysisResultModel,
    InsightModel,
    ModelPerformanceModel,
    StatisticalTestModel,
)
from .mappers.analytics_mapper import AnalyticsMapper


class AnalyticsRepositoryImpl(AnalyticsRepository):
    """SQLAlchemy implementation of AnalyticsRepository."""

    def __init__(self, session_factory: SessionFactory):
        self.session_factory = session_factory
        self.mapper = AnalyticsMapper()

    async def save_analysis_result(self, analysis_result: AnalysisResult) -> None:
        """Save analysis result to storage."""
        async with self.session_factory() as session:
            try:
                result_model = self.mapper.analysis_result_to_model(analysis_result)
                merged_model = await session.merge(result_model)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_analysis_result(self, analysis_id: UUID) -> Optional[AnalysisResult]:
        """Retrieve analysis result by ID."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(AnalysisResultModel)
                    .options(
                        selectinload(AnalysisResultModel.model_performances),
                        selectinload(AnalysisResultModel.insights),
                    )
                    .where(AnalysisResultModel.id == analysis_id)
                )
                result = await session.execute(query)
                result_model = result.scalar_one_or_none()

                return self.mapper.model_to_analysis_result(result_model) if result_model else None
            except Exception:
                await session.rollback()
                raise

    async def get_analysis_results_by_test(self, test_id: UUID) -> List[AnalysisResult]:
        """Get all analysis results for a specific test."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(AnalysisResultModel)
                    .options(
                        selectinload(AnalysisResultModel.model_performances),
                        selectinload(AnalysisResultModel.insights),
                    )
                    .where(AnalysisResultModel.test_id == test_id)
                    .order_by(AnalysisResultModel.created_at.desc())
                )
                result = await session.execute(query)
                result_models = result.scalars().all()

                return [self.mapper.model_to_analysis_result(model) for model in result_models]
            except Exception:
                await session.rollback()
                raise

    async def save_model_performance(self, model_performance: ModelPerformance) -> None:
        """Save model performance analysis."""
        async with self.session_factory() as session:
            try:
                perf_model = self.mapper.model_performance_to_model(model_performance)
                merged_model = await session.merge(perf_model)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_model_performance(self, performance_id: UUID) -> Optional[ModelPerformance]:
        """Retrieve model performance by ID."""
        async with self.session_factory() as session:
            try:
                query = select(ModelPerformanceModel).where(
                    ModelPerformanceModel.id == performance_id
                )
                result = await session.execute(query)
                perf_model = result.scalar_one_or_none()

                return self.mapper.model_to_model_performance(perf_model) if perf_model else None
            except Exception:
                await session.rollback()
                raise

    async def get_model_performances_by_test(self, test_id: UUID) -> List[ModelPerformance]:
        """Get all model performances for a specific test."""
        async with self.session_factory() as session:
            try:
                # Need to join through analysis_results
                query = (
                    select(ModelPerformanceModel)
                    .join(
                        AnalysisResultModel,
                        ModelPerformanceModel.analysis_result_id == AnalysisResultModel.id,
                    )
                    .where(AnalysisResultModel.test_id == test_id)
                    .order_by(ModelPerformanceModel.overall_score.desc())
                )
                result = await session.execute(query)
                perf_models = result.scalars().all()

                return [self.mapper.model_to_model_performance(model) for model in perf_models]
            except Exception:
                await session.rollback()
                raise

    async def save_statistical_test(self, statistical_test: StatisticalTest) -> None:
        """Save statistical test configuration."""
        async with self.session_factory() as session:
            try:
                stat_model = self.mapper.statistical_test_to_model(statistical_test)
                merged_model = await session.merge(stat_model)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_statistical_test(self, test_id: UUID) -> Optional[StatisticalTest]:
        """Retrieve statistical test by ID."""
        async with self.session_factory() as session:
            try:
                query = select(StatisticalTestModel).where(StatisticalTestModel.id == test_id)
                result = await session.execute(query)
                stat_model = result.scalar_one_or_none()

                return self.mapper.model_to_statistical_test(stat_model) if stat_model else None
            except Exception:
                await session.rollback()
                raise

    async def save_insights(self, insights: List[Insight]) -> None:
        """Save generated insights."""
        async with self.session_factory() as session:
            try:
                insight_models = [self.mapper.insight_to_model(insight) for insight in insights]
                for model in insight_models:
                    await session.merge(model)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_insights_by_analysis(self, analysis_id: UUID) -> List[Insight]:
        """Get insights for specific analysis."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(InsightModel)
                    .where(InsightModel.analysis_result_id == analysis_id)
                    .order_by(InsightModel.priority_score.desc())
                )
                result = await session.execute(query)
                insight_models = result.scalars().all()

                return [self.mapper.model_to_insight(model) for model in insight_models]
            except Exception:
                await session.rollback()
                raise

    async def get_insights_by_severity(self, analysis_id: UUID, severity: str) -> List[Insight]:
        """Get insights filtered by severity level."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(InsightModel)
                    .where(
                        InsightModel.analysis_result_id == analysis_id,
                        InsightModel.severity == severity,
                    )
                    .order_by(InsightModel.priority_score.desc())
                )
                result = await session.execute(query)
                insight_models = result.scalars().all()

                return [self.mapper.model_to_insight(model) for model in insight_models]
            except Exception:
                await session.rollback()
                raise

    async def search_analysis_results(
        self, criteria: Dict[str, Any], limit: int = 100, offset: int = 0
    ) -> List[AnalysisResult]:
        """Search analysis results by criteria."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(AnalysisResultModel)
                    .options(
                        selectinload(AnalysisResultModel.model_performances),
                        selectinload(AnalysisResultModel.insights),
                    )
                    .order_by(AnalysisResultModel.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )

                # Apply search criteria
                if "analysis_type" in criteria:
                    query = query.where(
                        AnalysisResultModel.analysis_type == criteria["analysis_type"]
                    )
                if "status" in criteria:
                    query = query.where(AnalysisResultModel.status == criteria["status"])
                if "test_id" in criteria:
                    query = query.where(AnalysisResultModel.test_id == criteria["test_id"])

                result = await session.execute(query)
                result_models = result.scalars().all()

                return [self.mapper.model_to_analysis_result(model) for model in result_models]
            except Exception:
                await session.rollback()
                raise

    async def get_model_performance_history(
        self, model_id: str, limit: int = 50
    ) -> List[ModelPerformance]:
        """Get historical performance data for a model."""
        async with self.session_factory() as session:
            try:
                query = (
                    select(ModelPerformanceModel)
                    .where(ModelPerformanceModel.model_id == model_id)
                    .order_by(ModelPerformanceModel.calculated_at.desc())
                    .limit(limit)
                )
                result = await session.execute(query)
                perf_models = result.scalars().all()

                return [self.mapper.model_to_model_performance(model) for model in perf_models]
            except Exception:
                await session.rollback()
                raise

    async def get_test_performance_trends(
        self, test_ids: List[UUID], time_period: str = "week"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trends over time for multiple tests."""
        # This would be a complex query involving time-based aggregations
        # Placeholder implementation
        return {}

    async def delete_analysis_result(self, analysis_id: UUID) -> bool:
        """Delete analysis result and related data."""
        async with self.session_factory() as session:
            try:
                query = delete(AnalysisResultModel).where(AnalysisResultModel.id == analysis_id)
                result = await session.execute(query)
                await session.commit()

                return result.rowcount > 0
            except Exception:
                await session.rollback()
                raise

    async def update_analysis_result(
        self, analysis_id: UUID, updates: Dict[str, Any]
    ) -> Optional[AnalysisResult]:
        """Update analysis result with new data."""
        async with self.session_factory() as session:
            try:
                query = (
                    update(AnalysisResultModel)
                    .where(AnalysisResultModel.id == analysis_id)
                    .values(**updates)
                    .returning(AnalysisResultModel)
                )
                result = await session.execute(query)
                await session.commit()

                updated_model = result.scalar_one_or_none()
                return (
                    self.mapper.model_to_analysis_result(updated_model) if updated_model else None
                )
            except Exception:
                await session.rollback()
                raise
