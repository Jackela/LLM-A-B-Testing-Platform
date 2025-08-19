"""Analytics repository interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..entities.analysis_result import AnalysisResult
from ..entities.model_performance import ModelPerformance
from ..entities.statistical_test import StatisticalTest
from ..value_objects.insight import Insight


class AnalyticsRepository(ABC):
    """Repository interface for analytics domain persistence."""

    @abstractmethod
    async def save_analysis_result(self, analysis_result: AnalysisResult) -> None:
        """Save analysis result to storage."""
        pass

    @abstractmethod
    async def get_analysis_result(self, analysis_id: UUID) -> Optional[AnalysisResult]:
        """Retrieve analysis result by ID."""
        pass

    @abstractmethod
    async def get_analysis_results_by_test(self, test_id: UUID) -> List[AnalysisResult]:
        """Get all analysis results for a specific test."""
        pass

    @abstractmethod
    async def save_model_performance(self, model_performance: ModelPerformance) -> None:
        """Save model performance analysis."""
        pass

    @abstractmethod
    async def get_model_performance(self, performance_id: UUID) -> Optional[ModelPerformance]:
        """Retrieve model performance by ID."""
        pass

    @abstractmethod
    async def get_model_performances_by_test(self, test_id: UUID) -> List[ModelPerformance]:
        """Get all model performances for a specific test."""
        pass

    @abstractmethod
    async def save_statistical_test(self, statistical_test: StatisticalTest) -> None:
        """Save statistical test configuration."""
        pass

    @abstractmethod
    async def get_statistical_test(self, test_id: UUID) -> Optional[StatisticalTest]:
        """Retrieve statistical test by ID."""
        pass

    @abstractmethod
    async def save_insights(self, insights: List[Insight]) -> None:
        """Save generated insights."""
        pass

    @abstractmethod
    async def get_insights_by_analysis(self, analysis_id: UUID) -> List[Insight]:
        """Get insights for specific analysis."""
        pass

    @abstractmethod
    async def get_insights_by_severity(self, analysis_id: UUID, severity: str) -> List[Insight]:
        """Get insights filtered by severity level."""
        pass

    @abstractmethod
    async def search_analysis_results(
        self, criteria: Dict[str, Any], limit: int = 100, offset: int = 0
    ) -> List[AnalysisResult]:
        """Search analysis results by criteria."""
        pass

    @abstractmethod
    async def get_model_performance_history(
        self, model_id: str, limit: int = 50
    ) -> List[ModelPerformance]:
        """Get historical performance data for a model."""
        pass

    @abstractmethod
    async def get_test_performance_trends(
        self, test_ids: List[UUID], time_period: str = "week"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trends over time for multiple tests."""
        pass

    @abstractmethod
    async def delete_analysis_result(self, analysis_id: UUID) -> bool:
        """Delete analysis result and related data."""
        pass

    @abstractmethod
    async def update_analysis_result(
        self, analysis_id: UUID, updates: Dict[str, Any]
    ) -> Optional[AnalysisResult]:
        """Update analysis result with new data."""
        pass
