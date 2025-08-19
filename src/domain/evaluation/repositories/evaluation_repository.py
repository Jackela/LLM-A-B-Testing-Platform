"""Repository interfaces for evaluation domain."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..entities.evaluation_result import EvaluationResult
from ..entities.evaluation_template import EvaluationTemplate
from ..entities.judge import Judge
from ..value_objects.calibration_data import CalibrationData


class EvaluationRepository(ABC):
    """Repository interface for evaluation domain entities."""

    # Judge repository methods
    @abstractmethod
    async def save_judge(self, judge: Judge) -> None:
        """Save or update judge."""
        pass

    @abstractmethod
    async def get_judge(self, judge_id: str) -> Optional[Judge]:
        """Get judge by ID."""
        pass

    @abstractmethod
    async def get_judges(
        self,
        is_active: Optional[bool] = None,
        is_calibrated: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Judge]:
        """Get judges with optional filters."""
        pass

    @abstractmethod
    async def delete_judge(self, judge_id: str) -> None:
        """Delete judge."""
        pass

    # Template repository methods
    @abstractmethod
    async def save_template(self, template: EvaluationTemplate) -> None:
        """Save or update evaluation template."""
        pass

    @abstractmethod
    async def get_template(self, template_id: UUID) -> Optional[EvaluationTemplate]:
        """Get template by ID."""
        pass

    @abstractmethod
    async def get_templates(
        self,
        is_active: Optional[bool] = None,
        created_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[EvaluationTemplate]:
        """Get templates with optional filters."""
        pass

    @abstractmethod
    async def get_template_versions(self, name: str) -> List[EvaluationTemplate]:
        """Get all versions of a template by name."""
        pass

    @abstractmethod
    async def delete_template(self, template_id: UUID) -> None:
        """Delete template."""
        pass

    # Evaluation result repository methods
    @abstractmethod
    async def save_evaluation_result(self, result: EvaluationResult) -> None:
        """Save or update evaluation result."""
        pass

    @abstractmethod
    async def get_evaluation_result(self, result_id: UUID) -> Optional[EvaluationResult]:
        """Get evaluation result by ID."""
        pass

    @abstractmethod
    async def get_evaluation_results(
        self,
        judge_id: Optional[str] = None,
        template_id: Optional[UUID] = None,
        is_successful: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[EvaluationResult]:
        """Get evaluation results with optional filters."""
        pass

    @abstractmethod
    async def get_results_for_consensus(
        self, prompt: str, response: str, template_id: UUID
    ) -> List[EvaluationResult]:
        """Get all evaluation results for specific prompt/response/template combination."""
        pass

    @abstractmethod
    async def delete_evaluation_result(self, result_id: UUID) -> None:
        """Delete evaluation result."""
        pass

    # Calibration repository methods
    @abstractmethod
    async def save_calibration_data(self, judge_id: str, calibration: CalibrationData) -> None:
        """Save calibration data for judge."""
        pass

    @abstractmethod
    async def get_calibration_data(self, judge_id: str) -> Optional[CalibrationData]:
        """Get calibration data for judge."""
        pass

    @abstractmethod
    async def get_calibration_history(
        self, judge_id: str, limit: Optional[int] = None
    ) -> List[CalibrationData]:
        """Get calibration history for judge."""
        pass

    # Analytics and reporting methods
    @abstractmethod
    async def get_judge_performance_metrics(self, judge_id: str, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics for judge over specified period."""
        pass

    @abstractmethod
    async def get_template_usage_stats(self, template_id: UUID, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for template over specified period."""
        pass

    @abstractmethod
    async def get_consensus_statistics(
        self, template_id: Optional[UUID] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get consensus statistics over specified period."""
        pass

    @abstractmethod
    async def get_quality_metrics(
        self, judge_id: Optional[str] = None, template_id: Optional[UUID] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get quality metrics over specified period."""
        pass
