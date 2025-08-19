"""Parallel evaluation service for efficient judge evaluation."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ....domain.evaluation.entities.evaluation_result import EvaluationResult
from ....domain.evaluation.entities.evaluation_template import EvaluationTemplate
from ....domain.evaluation.entities.judge import Judge
from ....domain.evaluation.exceptions import (
    EvaluationTimeoutError,
    JudgeNotAvailableError,
    TemplateRenderError,
)
from ....domain.model_provider.entities.model_response import ModelResponse
from ....domain.test_management.entities.test_sample import TestSample
from ..model_provider.model_provider_service import ModelProviderService

logger = logging.getLogger(__name__)


class ParallelEvaluator:
    """Service for executing judge evaluations in parallel."""

    def __init__(self, model_provider_service: ModelProviderService):
        self.model_provider_service = model_provider_service
        self._evaluation_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._active_evaluations: Dict[str, int] = {}
        self._max_concurrent_per_judge = 3  # Prevent overwhelming individual judges

    async def evaluate_with_judge(
        self,
        judge: Judge,
        template: EvaluationTemplate,
        model_response: ModelResponse,
        test_sample: TestSample,
        context: Dict[str, Any],
        timeout_seconds: int = 300,
    ) -> EvaluationResult:
        """Execute evaluation with a specific judge."""
        evaluation_id = uuid4()

        logger.debug(f"Starting evaluation {evaluation_id} with judge {judge.judge_id}")

        # Get or create semaphore for this judge
        if judge.judge_id not in self._evaluation_semaphores:
            self._evaluation_semaphores[judge.judge_id] = asyncio.Semaphore(
                self._max_concurrent_per_judge
            )

        semaphore = self._evaluation_semaphores[judge.judge_id]

        try:
            async with semaphore:
                # Track active evaluations
                self._active_evaluations[judge.judge_id] = (
                    self._active_evaluations.get(judge.judge_id, 0) + 1
                )

                # Execute evaluation with timeout
                result = await asyncio.wait_for(
                    self._execute_single_evaluation(
                        judge, template, model_response, test_sample, context
                    ),
                    timeout=timeout_seconds,
                )

                logger.debug(
                    f"Completed evaluation {evaluation_id} with judge {judge.judge_id} "
                    f"(score: {result.overall_score})"
                )

                return result

        except asyncio.TimeoutError:
            logger.warning(
                f"Evaluation {evaluation_id} with judge {judge.judge_id} timed out after {timeout_seconds}s"
            )
            return EvaluationResult.create_failed(
                judge_id=judge.judge_id,
                template_id=template.template_id,
                error_message=f"Evaluation timed out after {timeout_seconds} seconds",
            )

        except Exception as e:
            logger.error(f"Evaluation {evaluation_id} with judge {judge.judge_id} failed: {str(e)}")
            return EvaluationResult.create_failed(
                judge_id=judge.judge_id, template_id=template.template_id, error_message=str(e)
            )

        finally:
            # Update active evaluations count
            current_count = self._active_evaluations.get(judge.judge_id, 1)
            if current_count <= 1:
                self._active_evaluations.pop(judge.judge_id, None)
            else:
                self._active_evaluations[judge.judge_id] = current_count - 1

    async def _execute_single_evaluation(
        self,
        judge: Judge,
        template: EvaluationTemplate,
        model_response: ModelResponse,
        test_sample: TestSample,
        context: Dict[str, Any],
    ) -> EvaluationResult:
        """Execute a single evaluation."""
        start_time = datetime.utcnow()

        try:
            # Validate judge readiness
            if not judge.is_active:
                raise JudgeNotAvailableError(f"Judge {judge.judge_id} is not active")

            if not judge.is_calibrated():
                raise JudgeNotAvailableError(f"Judge {judge.judge_id} is not calibrated")

            # Create pending evaluation result
            result = EvaluationResult.create_pending(
                judge_id=judge.judge_id,
                template_id=template.template_id,
                prompt=test_sample.prompt,
                response=model_response.content,
            )

            # Render evaluation prompt from template
            evaluation_prompt = await self._render_evaluation_prompt(
                template, model_response, test_sample, context
            )

            # Call model provider for evaluation
            model_response_text = await self._call_judge_model(judge, template, evaluation_prompt)

            # Parse evaluation response
            dimension_scores, confidence_score, reasoning = await self._parse_evaluation_response(
                model_response_text, template, judge
            )

            # Calculate evaluation time
            end_time = datetime.utcnow()
            evaluation_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Complete the evaluation result
            result.complete_evaluation(
                template=template,
                dimension_scores=dimension_scores,
                confidence_score=confidence_score,
                reasoning=reasoning,
                evaluation_time_ms=evaluation_time_ms,
            )

            return result

        except Exception as e:
            # Create failed evaluation result
            end_time = datetime.utcnow()
            evaluation_time_ms = int((end_time - start_time).total_seconds() * 1000)

            failed_result = EvaluationResult.create_failed(
                judge_id=judge.judge_id, template_id=template.template_id, error_message=str(e)
            )
            failed_result.evaluation_time_ms = evaluation_time_ms

            return failed_result

    async def _render_evaluation_prompt(
        self,
        template: EvaluationTemplate,
        model_response: ModelResponse,
        test_sample: TestSample,
        context: Dict[str, Any],
    ) -> str:
        """Render evaluation prompt from template."""
        try:
            evaluation_prompt = template.render(
                prompt=test_sample.prompt,
                response=model_response.content,
                expected_response=test_sample.expected_response,
                judge_name=context.get("judge_name", "Judge"),
                domain=context.get("domain", "general"),
                difficulty_level=context.get("difficulty_level", "unknown"),
                **context,
            )

            return evaluation_prompt

        except Exception as e:
            raise TemplateRenderError(f"Failed to render evaluation template: {str(e)}")

    async def _call_judge_model(
        self, judge: Judge, template: EvaluationTemplate, evaluation_prompt: str
    ) -> str:
        """Call the underlying model provider for judge evaluation."""
        try:
            # Get model provider configuration from judge
            model_provider_id = judge.model_provider_id
            model_config = template.judge_model_id  # This would be the specific model to use

            # Prepare model request
            model_request = {
                "model_id": model_config,
                "provider_id": model_provider_id,
                "prompt": evaluation_prompt,
                "parameters": template.model_parameters,
                "metadata": {
                    "evaluation_type": "judge",
                    "judge_id": judge.judge_id,
                    "template_id": str(template.template_id),
                },
            }

            # Call model provider service
            response = await self.model_provider_service.generate_response(model_request)

            if not response or not response.content:
                raise Exception("Empty response from judge model")

            return response.content

        except Exception as e:
            logger.error(f"Failed to call judge model for {judge.judge_id}: {str(e)}")
            raise

    async def _parse_evaluation_response(
        self, model_response: str, template: EvaluationTemplate, judge: Judge
    ) -> tuple[Dict[str, int], float, str]:
        """Parse model response into structured evaluation data."""
        try:
            # Use the judge's built-in parsing logic
            dimension_scores, confidence_score, reasoning = judge._parse_evaluation_response(
                model_response, template
            )

            return dimension_scores, float(confidence_score), reasoning

        except Exception as e:
            logger.error(
                f"Failed to parse evaluation response for judge {judge.judge_id}: {str(e)}"
            )

            # Return default values for failed parsing
            default_scores = {}
            for dimension in template.dimensions:
                min_score, max_score = dimension.get_score_range()
                default_scores[dimension.name] = (min_score + max_score) // 2

            return default_scores, 0.5, f"Failed to parse response: {str(e)}"

    async def evaluate_batch(
        self,
        evaluation_tasks: List[Dict[str, Any]],
        max_concurrent: int = 10,
        timeout_seconds: int = 300,
    ) -> List[EvaluationResult]:
        """Execute a batch of evaluations in parallel."""
        logger.info(f"Starting batch evaluation of {len(evaluation_tasks)} tasks")

        # Create evaluation coroutines
        evaluation_coroutines = []
        for task in evaluation_tasks:
            coroutine = self.evaluate_with_judge(
                judge=task["judge"],
                template=task["template"],
                model_response=task["model_response"],
                test_sample=task["test_sample"],
                context=task["context"],
                timeout_seconds=timeout_seconds,
            )
            evaluation_coroutines.append(coroutine)

        # Execute with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_evaluation(coro):
            async with semaphore:
                return await coro

        bounded_coroutines = [bounded_evaluation(coro) for coro in evaluation_coroutines]

        # Execute all evaluations
        start_time = datetime.utcnow()
        results = await asyncio.gather(*bounded_coroutines, return_exceptions=True)
        end_time = datetime.utcnow()

        # Convert exceptions to failed results
        evaluation_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = evaluation_tasks[i]
                failed_result = EvaluationResult.create_failed(
                    judge_id=task["judge"].judge_id,
                    template_id=task["template"].template_id,
                    error_message=str(result),
                )
                evaluation_results.append(failed_result)
            else:
                evaluation_results.append(result)

        batch_duration = (end_time - start_time).total_seconds()
        successful_count = len([r for r in evaluation_results if r.is_successful()])

        logger.info(
            f"Completed batch evaluation in {batch_duration:.2f}s: "
            f"{successful_count}/{len(evaluation_results)} successful"
        )

        return evaluation_results

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get current evaluation statistics."""
        total_active = sum(self._active_evaluations.values())

        return {
            "active_evaluations_total": total_active,
            "active_evaluations_by_judge": self._active_evaluations.copy(),
            "judges_with_semaphores": len(self._evaluation_semaphores),
            "max_concurrent_per_judge": self._max_concurrent_per_judge,
        }

    async def wait_for_completion(self, timeout_seconds: Optional[int] = None) -> None:
        """Wait for all active evaluations to complete."""
        if not self._active_evaluations:
            return

        logger.info(
            f"Waiting for {sum(self._active_evaluations.values())} active evaluations to complete"
        )

        start_time = datetime.utcnow()

        while self._active_evaluations:
            if timeout_seconds:
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    raise EvaluationTimeoutError(
                        f"Evaluations did not complete within {timeout_seconds}s"
                    )

            await asyncio.sleep(0.1)  # Short sleep to avoid busy waiting

        logger.info("All evaluations completed")

    def clear_semaphores(self) -> None:
        """Clear all judge semaphores (for testing/cleanup)."""
        self._evaluation_semaphores.clear()
        self._active_evaluations.clear()
