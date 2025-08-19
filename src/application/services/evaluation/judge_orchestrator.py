"""Judge orchestration service for multi-judge evaluation."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from ....domain.evaluation.entities.evaluation_result import EvaluationResult
from ....domain.evaluation.entities.evaluation_template import EvaluationTemplate
from ....domain.evaluation.entities.judge import Judge
from ....domain.evaluation.exceptions import (
    EvaluationTimeoutError,
    InsufficientJudgesError,
    JudgeNotAvailableError,
    QualityControlError,
)
from ....domain.model_provider.entities.model_response import ModelResponse
from ....domain.test_management.entities.test_sample import TestSample
from ...dto.evaluation_request_dto import (
    ConsensusEvaluationResultDTO,
    ConsensusResultDTO,
    EvaluationConfigDTO,
    EvaluationRequestDTO,
    IndividualEvaluationDTO,
)
from ...interfaces.unit_of_work import UnitOfWork
from ..model_provider.model_provider_service import ModelProviderService
from .consensus_builder import ConsensusBuilder
from .parallel_evaluator import ParallelEvaluator
from .quality_assurance import QualityAssurance

logger = logging.getLogger(__name__)


class JudgeOrchestrator:
    """Orchestrates multi-judge evaluation with consensus building."""

    def __init__(
        self,
        uow: UnitOfWork,
        model_provider_service: ModelProviderService,
        consensus_builder: ConsensusBuilder,
        quality_assurance: QualityAssurance,
        parallel_evaluator: ParallelEvaluator,
    ):
        self.uow = uow
        self.model_provider_service = model_provider_service
        self.consensus_builder = consensus_builder
        self.quality_assurance = quality_assurance
        self.parallel_evaluator = parallel_evaluator
        self._active_evaluations: Dict[UUID, asyncio.Task] = {}
        self._judge_performance_cache: Dict[str, dict] = {}

    async def evaluate_response(
        self,
        model_response: ModelResponse,
        test_sample: TestSample,
        evaluation_config: EvaluationConfigDTO,
    ) -> ConsensusEvaluationResultDTO:
        """Orchestrate multi-judge evaluation with consensus building."""
        evaluation_start = datetime.utcnow()
        evaluation_id = uuid4()

        logger.info(
            f"Starting multi-judge evaluation {evaluation_id} for response {model_response.response_id}"
        )

        try:
            # Step 1: Select appropriate judges
            judges = await self._select_judges(evaluation_config, test_sample)

            if len(judges) < evaluation_config.minimum_judges:
                raise InsufficientJudgesError(
                    f"Need at least {evaluation_config.minimum_judges} judges, "
                    f"only {len(judges)} available"
                )

            # Step 2: Prepare evaluation context
            evaluation_context = self._prepare_evaluation_context(
                model_response, test_sample, evaluation_config
            )

            # Step 3: Execute parallel evaluations
            individual_results = await self._execute_parallel_evaluations(
                judges, model_response, test_sample, evaluation_context, evaluation_config
            )

            # Step 4: Filter successful results and handle failures
            valid_results, failed_results = self._filter_evaluation_results(individual_results)

            if len(valid_results) < evaluation_config.minimum_judges:
                logger.warning(
                    f"Insufficient successful evaluations: {len(valid_results)} of {len(individual_results)}"
                )
                # Try recovery strategy
                recovery_results = await self._attempt_recovery(
                    failed_results,
                    model_response,
                    test_sample,
                    evaluation_context,
                    evaluation_config,
                )
                valid_results.extend(recovery_results)

            if len(valid_results) < evaluation_config.minimum_judges:
                raise InsufficientJudgesError(
                    f"After recovery, still insufficient evaluations: {len(valid_results)}"
                )

            # Step 5: Build consensus from valid results
            consensus_result = await self.consensus_builder.build_consensus(
                valid_results, evaluation_config
            )

            # Step 6: Apply quality control
            quality_report = await self.quality_assurance.validate_consensus(
                consensus_result, valid_results, evaluation_config
            )

            # Step 7: Update judge performance tracking
            await self._update_judge_performance(judges, individual_results)

            evaluation_duration = int((datetime.utcnow() - evaluation_start).total_seconds() * 1000)

            logger.info(
                f"Completed evaluation {evaluation_id} in {evaluation_duration}ms with "
                f"{len(valid_results)} judges (consensus: {consensus_result.consensus_score})"
            )

            return ConsensusEvaluationResultDTO(
                consensus=consensus_result,
                individual_results=[self._convert_to_dto(r) for r in individual_results],
                quality_report=quality_report,
                evaluation_metadata=self._create_evaluation_metadata(
                    evaluation_config, judges, evaluation_duration
                ),
                total_judges=len(individual_results),
                successful_judges=len(valid_results),
                failed_judges=len(failed_results),
                outlier_judges=consensus_result.outlier_judges,
                consensus_reached=quality_report.quality_passed,
                evaluation_duration_ms=evaluation_duration,
                created_at=evaluation_start,
            )

        except Exception as e:
            logger.error(f"Evaluation {evaluation_id} failed: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up active evaluations tracking
            if evaluation_id in self._active_evaluations:
                del self._active_evaluations[evaluation_id]

    async def _select_judges(
        self, evaluation_config: EvaluationConfigDTO, test_sample: TestSample
    ) -> List[Judge]:
        """Select appropriate judges for the evaluation."""
        async with self.uow:
            # Get all active, calibrated judges
            all_judges = await self.uow.evaluation.find_active_judges()

            # Filter judges that are production ready
            production_judges = [judge for judge in all_judges if judge.is_production_ready()]

            if len(production_judges) < evaluation_config.minimum_judges:
                logger.warning(
                    f"Insufficient production-ready judges: {len(production_judges)} of {evaluation_config.minimum_judges} required"
                )
                # Fall back to active judges that are at least calibrated
                calibrated_judges = [judge for judge in all_judges if judge.is_calibrated()]
                production_judges = calibrated_judges

            # Apply judge selection strategy
            selected_judges = await self._apply_judge_selection_strategy(
                production_judges, evaluation_config, test_sample
            )

            logger.debug(f"Selected {len(selected_judges)} judges for evaluation")
            return selected_judges

    async def _apply_judge_selection_strategy(
        self,
        available_judges: List[Judge],
        evaluation_config: EvaluationConfigDTO,
        test_sample: TestSample,
    ) -> List[Judge]:
        """Apply intelligent judge selection strategy."""
        # For now, implement a simple strategy
        # In production, this could consider:
        # - Judge specialization for test domain
        # - Load balancing across judges
        # - Performance history
        # - Diversity of perspectives

        # Ensure we have enough judges
        target_count = min(
            len(available_judges), evaluation_config.minimum_judges + 2
        )  # Add 2 for redundancy

        # Sort by performance metrics (mock implementation)
        sorted_judges = sorted(
            available_judges, key=lambda j: self._calculate_judge_priority_score(j), reverse=True
        )

        return sorted_judges[:target_count]

    def _calculate_judge_priority_score(self, judge: Judge) -> float:
        """Calculate priority score for judge selection."""
        # Mock implementation - in production would use real performance metrics
        base_score = 1.0

        # Boost for recent calibration
        if judge.calibration_data:
            accuracy = float(judge.calibration_data.accuracy)
            consistency = float(judge.calibration_data.consistency)
            bias_penalty = float(judge.calibration_data.bias_score) * 0.1
            base_score = accuracy * consistency - bias_penalty

        # Consider recent performance
        recent_stats = judge.get_performance_stats(days=7)
        success_rate = recent_stats.get("success_rate", 0.8)
        avg_confidence = recent_stats.get("average_confidence", 0.5)

        priority_score = base_score * success_rate * (0.5 + avg_confidence * 0.5)

        return max(0.1, min(2.0, priority_score))  # Clamp between 0.1 and 2.0

    def _prepare_evaluation_context(
        self,
        model_response: ModelResponse,
        test_sample: TestSample,
        evaluation_config: EvaluationConfigDTO,
    ) -> dict:
        """Prepare evaluation context for judges."""
        return {
            "response_id": str(model_response.response_id),
            "sample_id": str(test_sample.sample_id),
            "prompt": test_sample.prompt,
            "response": model_response.content,
            "expected_response": test_sample.expected_response,
            "difficulty_level": (
                test_sample.difficulty_level.name if test_sample.difficulty_level else None
            ),
            "domain": test_sample.metadata.get("domain", "general"),
            "evaluation_criteria": evaluation_config.batch_size,  # This would contain actual criteria
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _execute_parallel_evaluations(
        self,
        judges: List[Judge],
        model_response: ModelResponse,
        test_sample: TestSample,
        evaluation_context: dict,
        evaluation_config: EvaluationConfigDTO,
    ) -> List[EvaluationResult]:
        """Execute evaluations in parallel using dedicated service."""
        evaluation_tasks = []

        for judge in judges:
            # Get appropriate template for judge
            template = await self._get_evaluation_template(judge, test_sample)

            if template:
                task = self.parallel_evaluator.evaluate_with_judge(
                    judge=judge,
                    template=template,
                    model_response=model_response,
                    test_sample=test_sample,
                    context=evaluation_context,
                    timeout_seconds=evaluation_config.max_evaluation_time_seconds,
                )
                evaluation_tasks.append(task)

        # Execute with concurrency control
        semaphore = asyncio.Semaphore(evaluation_config.max_concurrent_evaluations)

        async def bounded_evaluation(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_evaluation(task) for task in evaluation_tasks]

        # Wait for all evaluations to complete
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        # Convert exceptions to failed evaluation results
        evaluation_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create failed evaluation result
                judge = judges[i] if i < len(judges) else judges[0]
                failed_result = EvaluationResult.create_failed(
                    judge_id=judge.judge_id,
                    template_id=uuid4(),  # Placeholder
                    error_message=str(result),
                )
                evaluation_results.append(failed_result)
            else:
                evaluation_results.append(result)

        return evaluation_results

    async def _get_evaluation_template(
        self, judge: Judge, test_sample: TestSample
    ) -> Optional[EvaluationTemplate]:
        """Get appropriate evaluation template for judge and sample."""
        # Select template based on judge's templates and sample requirements
        if not judge.templates:
            logger.warning(f"Judge {judge.judge_id} has no templates")
            return None

        # For now, use the first template
        # In production, this would select based on:
        # - Test domain
        # - Difficulty level
        # - Evaluation criteria
        return judge.templates[0]

    def _filter_evaluation_results(
        self, results: List[EvaluationResult]
    ) -> Tuple[List[EvaluationResult], List[EvaluationResult]]:
        """Filter evaluation results into successful and failed."""
        valid_results = [r for r in results if r.is_successful()]
        failed_results = [r for r in results if not r.is_successful()]

        logger.debug(
            f"Evaluation results: {len(valid_results)} successful, {len(failed_results)} failed"
        )

        return valid_results, failed_results

    async def _attempt_recovery(
        self,
        failed_results: List[EvaluationResult],
        model_response: ModelResponse,
        test_sample: TestSample,
        evaluation_context: dict,
        evaluation_config: EvaluationConfigDTO,
    ) -> List[EvaluationResult]:
        """Attempt to recover from failed evaluations."""
        if not evaluation_config.retry_failed_evaluations:
            return []

        logger.info(f"Attempting recovery for {len(failed_results)} failed evaluations")

        recovery_results = []

        # Try to get backup judges for failed evaluations
        async with self.uow:
            backup_judges = await self._get_backup_judges(failed_results)

            for judge in backup_judges[: min(len(failed_results), 3)]:  # Limit retries
                template = await self._get_evaluation_template(judge, test_sample)

                if template:
                    try:
                        result = await self.parallel_evaluator.evaluate_with_judge(
                            judge=judge,
                            template=template,
                            model_response=model_response,
                            test_sample=test_sample,
                            context=evaluation_context,
                            timeout_seconds=evaluation_config.max_evaluation_time_seconds
                            // 2,  # Shorter timeout for retries
                        )

                        if result.is_successful():
                            recovery_results.append(result)

                    except Exception as e:
                        logger.warning(
                            f"Recovery evaluation failed for judge {judge.judge_id}: {e}"
                        )
                        continue

        logger.info(f"Recovery yielded {len(recovery_results)} additional successful evaluations")
        return recovery_results

    async def _get_backup_judges(self, failed_results: List[EvaluationResult]) -> List[Judge]:
        """Get backup judges for failed evaluations."""
        # Get failed judge IDs
        failed_judge_ids = {r.judge_id for r in failed_results}

        # Get all available judges excluding failed ones
        all_judges = await self.uow.evaluation.find_active_judges()
        backup_judges = [
            judge
            for judge in all_judges
            if judge.judge_id not in failed_judge_ids and judge.is_calibrated()
        ]

        return backup_judges

    async def _update_judge_performance(
        self, judges: List[Judge], results: List[EvaluationResult]
    ) -> None:
        """Update judge performance tracking."""
        # Group results by judge
        results_by_judge = {}
        for result in results:
            if result.judge_id not in results_by_judge:
                results_by_judge[result.judge_id] = []
            results_by_judge[result.judge_id].append(result)

        # Update performance metrics for each judge
        for judge in judges:
            judge_results = results_by_judge.get(judge.judge_id, [])

            if judge_results:
                await self._update_individual_judge_performance(judge, judge_results)

    async def _update_individual_judge_performance(
        self, judge: Judge, results: List[EvaluationResult]
    ) -> None:
        """Update performance metrics for individual judge."""
        # This would update judge calibration and performance history
        # For now, just log the update
        successful_count = len([r for r in results if r.is_successful()])
        total_count = len(results)

        logger.debug(
            f"Updated performance for judge {judge.judge_id}: "
            f"{successful_count}/{total_count} successful"
        )

        # In production, this would:
        # 1. Update calibration data
        # 2. Recalculate performance metrics
        # 3. Check if recalibration is needed
        # 4. Store updated judge state

    def _convert_to_dto(self, evaluation_result: EvaluationResult) -> IndividualEvaluationDTO:
        """Convert domain evaluation result to DTO."""
        return IndividualEvaluationDTO(
            evaluation_id=evaluation_result.result_id,
            judge_id=evaluation_result.judge_id,
            overall_score=evaluation_result.overall_score,
            dimension_scores=evaluation_result.dimension_scores.copy(),
            confidence_score=evaluation_result.confidence_score,
            reasoning=evaluation_result.reasoning,
            evaluation_time_ms=evaluation_result.evaluation_time_ms,
            template_id=evaluation_result.template_id,
            is_successful=evaluation_result.is_successful(),
            error_message=evaluation_result.error_message,
            metadata=evaluation_result.metadata.copy() if evaluation_result.metadata else None,
            created_at=evaluation_result.created_at,
        )

    def _create_evaluation_metadata(
        self,
        evaluation_config: EvaluationConfigDTO,
        judges: List[Judge],
        evaluation_duration_ms: int,
    ) -> dict:
        """Create metadata for evaluation."""
        return {
            "evaluation_config": {
                "minimum_judges": evaluation_config.minimum_judges,
                "consensus_method": evaluation_config.consensus_method,
                "confidence_weighting": evaluation_config.confidence_weighting,
                "exclude_outliers": evaluation_config.exclude_outliers,
            },
            "judges_used": [
                {
                    "judge_id": judge.judge_id,
                    "judge_name": judge.name,
                    "is_production_ready": judge.is_production_ready(),
                }
                for judge in judges
            ],
            "evaluation_duration_ms": evaluation_duration_ms,
            "orchestrator_version": "1.0.0",
        }
