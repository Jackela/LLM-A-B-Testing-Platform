"""Domain-model mapper for Evaluation domain."""

from typing import Any, Dict, List
from uuid import UUID

from .....domain.evaluation.entities.dimension import Dimension
from .....domain.evaluation.entities.evaluation_result import EvaluationResult
from .....domain.evaluation.entities.evaluation_template import EvaluationTemplate
from .....domain.evaluation.entities.judge import Judge
from ...models.evaluation_models import (
    DimensionModel,
    EvaluationResultModel,
    EvaluationTemplateModel,
    JudgeModel,
)


class EvaluationMapper:
    """Mapper between Evaluation domain entities and database models."""

    def judge_to_model(self, judge: Judge) -> JudgeModel:
        """Convert Judge domain entity to database model."""
        return JudgeModel(
            id=judge.id,
            name=judge.name,
            description=judge.description,
            model_id=judge.model_id,
            provider_name=judge.provider_name,
            system_prompt=judge.system_prompt,
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
            is_active=judge.is_active,
            is_calibrated=judge.is_calibrated,
            calibration_score=judge.calibration_score,
            metadata=judge.metadata.copy() if judge.metadata else {},
            created_at=judge.created_at,
            updated_at=judge.updated_at,
            total_evaluations=judge.total_evaluations,
            average_latency_ms=judge.average_latency_ms,
            success_rate=judge.success_rate,
        )

    def model_to_judge(self, judge_model: JudgeModel) -> Judge:
        """Convert database model to Judge domain entity."""
        return Judge(
            id=judge_model.id,
            name=judge_model.name,
            description=judge_model.description,
            model_id=judge_model.model_id,
            provider_name=judge_model.provider_name,
            system_prompt=judge_model.system_prompt,
            temperature=judge_model.temperature,
            max_tokens=judge_model.max_tokens,
            is_active=judge_model.is_active,
            is_calibrated=judge_model.is_calibrated,
            calibration_score=judge_model.calibration_score,
            metadata=judge_model.metadata.copy() if judge_model.metadata else {},
            created_at=judge_model.created_at,
            updated_at=judge_model.updated_at,
            total_evaluations=judge_model.total_evaluations,
            average_latency_ms=judge_model.average_latency_ms,
            success_rate=judge_model.success_rate,
        )

    def template_to_model(self, template: EvaluationTemplate) -> EvaluationTemplateModel:
        """Convert EvaluationTemplate domain entity to database model."""
        template_model = EvaluationTemplateModel(
            id=template.id,
            name=template.name,
            version=template.version,
            description=template.description,
            is_active=template.is_active,
            created_by=template.created_by,
            created_at=template.created_at,
            updated_at=template.updated_at,
            prompt_template=template.prompt_template,
            response_format=template.response_format.copy(),
            scoring_rubric=template.scoring_rubric.copy(),
            metadata=template.metadata.copy() if template.metadata else {},
        )

        # Convert dimensions
        template_model.dimensions = [
            self._dimension_to_model(dim, template.id) for dim in template.dimensions
        ]

        return template_model

    def model_to_template(self, template_model: EvaluationTemplateModel) -> EvaluationTemplate:
        """Convert database model to EvaluationTemplate domain entity."""
        # Convert dimensions
        dimensions = [
            self._model_to_dimension(dim_model) for dim_model in template_model.dimensions
        ]

        return EvaluationTemplate(
            id=template_model.id,
            name=template_model.name,
            version=template_model.version,
            description=template_model.description,
            dimensions=dimensions,
            prompt_template=template_model.prompt_template,
            response_format=template_model.response_format.copy(),
            scoring_rubric=template_model.scoring_rubric.copy(),
            is_active=template_model.is_active,
            created_by=template_model.created_by,
            created_at=template_model.created_at,
            updated_at=template_model.updated_at,
            metadata=template_model.metadata.copy() if template_model.metadata else {},
        )

    def _dimension_to_model(self, dimension: Dimension, template_id: UUID) -> DimensionModel:
        """Convert Dimension to DimensionModel."""
        return DimensionModel(
            id=dimension.id,
            template_id=template_id,
            name=dimension.name,
            description=dimension.description,
            weight=dimension.weight,
            min_score=dimension.min_score,
            max_score=dimension.max_score,
            criteria=dimension.criteria.copy(),
            metadata=dimension.metadata.copy() if dimension.metadata else {},
        )

    def _model_to_dimension(self, dimension_model: DimensionModel) -> Dimension:
        """Convert DimensionModel to Dimension."""
        return Dimension(
            id=dimension_model.id,
            name=dimension_model.name,
            description=dimension_model.description,
            weight=dimension_model.weight,
            min_score=dimension_model.min_score,
            max_score=dimension_model.max_score,
            criteria=dimension_model.criteria.copy(),
            metadata=dimension_model.metadata.copy() if dimension_model.metadata else {},
        )

    def result_to_model(self, result: EvaluationResult) -> EvaluationResultModel:
        """Convert EvaluationResult domain entity to database model."""
        return EvaluationResultModel(
            id=result.id,
            test_id=result.test_id,
            model_response_id=result.model_response_id,
            judge_id=result.judge_id,
            template_id=result.template_id,
            overall_score=result.overall_score,
            dimension_scores=result.dimension_scores.copy(),
            raw_evaluation=result.raw_evaluation,
            parsed_evaluation=result.parsed_evaluation.copy() if result.parsed_evaluation else {},
            is_successful=result.is_successful,
            error_message=result.error_message,
            evaluation_time_ms=result.evaluation_time_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            evaluation_cost=result.evaluation_cost,
            created_at=result.created_at,
            completed_at=result.completed_at,
            metadata=result.metadata.copy() if result.metadata else {},
        )

    def model_to_result(self, result_model: EvaluationResultModel) -> EvaluationResult:
        """Convert database model to EvaluationResult domain entity."""
        return EvaluationResult(
            id=result_model.id,
            test_id=result_model.test_id,
            model_response_id=result_model.model_response_id,
            judge_id=result_model.judge_id,
            template_id=result_model.template_id,
            overall_score=result_model.overall_score,
            dimension_scores=result_model.dimension_scores.copy(),
            raw_evaluation=result_model.raw_evaluation,
            parsed_evaluation=(
                result_model.parsed_evaluation.copy() if result_model.parsed_evaluation else {}
            ),
            is_successful=result_model.is_successful,
            error_message=result_model.error_message,
            evaluation_time_ms=result_model.evaluation_time_ms,
            input_tokens=result_model.input_tokens,
            output_tokens=result_model.output_tokens,
            evaluation_cost=result_model.evaluation_cost,
            created_at=result_model.created_at,
            completed_at=result_model.completed_at,
            metadata=result_model.metadata.copy() if result_model.metadata else {},
        )
