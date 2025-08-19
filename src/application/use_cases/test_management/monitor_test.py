"""Use case for monitoring test progress and status."""

import logging
from uuid import UUID

from ....domain.model_provider.value_objects.money import Money
from ....domain.test_management.value_objects.test_status import TestStatus
from ...dto.test_configuration_dto import TestMonitoringResultDTO
from ...interfaces.unit_of_work import UnitOfWork
from ...services.model_provider_service import ModelProviderService

logger = logging.getLogger(__name__)


class MonitorTestUseCase:
    """Use case for monitoring test execution progress."""

    def __init__(self, uow: UnitOfWork, provider_service: ModelProviderService):
        self.uow = uow
        self.provider_service = provider_service

    async def execute(self, test_id: UUID) -> TestMonitoringResultDTO:
        """Execute test monitoring with comprehensive status reporting."""
        try:
            logger.debug(f"Monitoring test: {test_id}")

            async with self.uow:
                # Step 1: Load test aggregate
                test = await self.uow.tests.find_by_id(test_id)
                if not test:
                    logger.warning(f"Test not found: {test_id}")
                    return TestMonitoringResultDTO(
                        test_id=test_id,
                        status="not_found",
                        progress=0.0,
                        total_samples=0,
                        evaluated_samples=0,
                        model_scores={},
                        estimated_remaining_time=0.0,
                        current_cost=Money(0.0, "USD"),
                        errors=[f"Test {test_id} not found"],
                    )

                # Step 2: Calculate comprehensive progress metrics
                progress_metrics = await self._calculate_progress_metrics(test)

                # Step 3: Gather execution errors and issues
                execution_errors = await self._gather_execution_errors(test)

                # Step 4: Calculate current costs
                current_cost = await self._calculate_current_cost(test)

                logger.debug(f"Test {test_id} progress: {progress_metrics['progress']:.2%}")

                return TestMonitoringResultDTO(
                    test_id=test.id,
                    status=test.status.value,
                    progress=progress_metrics["progress"],
                    total_samples=progress_metrics["total_samples"],
                    evaluated_samples=progress_metrics["evaluated_samples"],
                    model_scores=progress_metrics["model_scores"],
                    estimated_remaining_time=progress_metrics["estimated_remaining_time"],
                    current_cost=current_cost,
                    errors=execution_errors,
                )

        except Exception as e:
            logger.error(f"Error monitoring test {test_id}: {e}", exc_info=True)
            return TestMonitoringResultDTO(
                test_id=test_id,
                status="monitoring_error",
                progress=0.0,
                total_samples=0,
                evaluated_samples=0,
                model_scores={},
                estimated_remaining_time=0.0,
                current_cost=Money(0.0, "USD"),
                errors=[f"Monitoring error: {str(e)}"],
            )

    async def _calculate_progress_metrics(self, test) -> dict:
        """Calculate comprehensive progress metrics for the test."""
        total_samples = len(test.samples)
        progress = test.calculate_progress()
        evaluated_samples = int(total_samples * progress)
        model_scores = test.get_model_scores()

        # Calculate estimated remaining time
        estimated_remaining_time = 0.0
        if test.status == TestStatus.RUNNING:
            # Get average evaluation time from recent samples
            avg_evaluation_time = await self._estimate_average_evaluation_time(test)
            estimated_remaining_time = test.estimate_remaining_time(avg_evaluation_time)

        # Enhanced metrics with breakdown by model
        model_progress = {}
        for model in test.configuration.models:
            model_evaluated = sum(
                1 for sample in test.samples if sample.has_evaluation_for_model(model)
            )
            model_progress[model] = model_evaluated / total_samples if total_samples > 0 else 0.0

        return {
            "progress": progress,
            "total_samples": total_samples,
            "evaluated_samples": evaluated_samples,
            "model_scores": model_scores,
            "estimated_remaining_time": estimated_remaining_time,
            "model_progress": model_progress,
        }

    async def _gather_execution_errors(self, test) -> list[str]:
        """Gather any execution errors or issues for the test."""
        errors = []

        # Check for test-level errors
        if test.status == TestStatus.FAILED and test.failure_reason:
            errors.append(f"Test failed: {test.failure_reason}")

        # Check provider health for ongoing tests
        if test.status == TestStatus.RUNNING:
            try:
                providers = await self.provider_service.get_providers_for_test(test.configuration)
                for provider in providers:
                    if not provider.health_status.is_operational:
                        errors.append(
                            f"Provider {provider.name} is not operational: {provider.health_status.name}"
                        )

                    # Check rate limits
                    if not provider.rate_limits.can_make_request():
                        errors.append(f"Provider {provider.name} has reached rate limits")
            except Exception as e:
                errors.append(f"Error checking provider status: {str(e)}")

        # Check for samples with evaluation errors
        failed_evaluations = 0
        for sample in test.samples:
            # This would check for evaluation errors in a real implementation
            # For now, we'll simulate based on sample metadata
            if sample.metadata and sample.metadata.get("evaluation_error"):
                failed_evaluations += 1

        if failed_evaluations > 0:
            failure_rate = failed_evaluations / len(test.samples)
            if failure_rate > 0.1:  # More than 10% failures
                errors.append(
                    f"High evaluation failure rate: {failed_evaluations}/{len(test.samples)} "
                    f"samples ({failure_rate:.1%})"
                )

        return errors

    async def _calculate_current_cost(self, test) -> Money:
        """Calculate current cost based on evaluations completed so far."""
        # This is a simplified calculation - in practice would track actual API costs
        evaluated_samples = int(len(test.samples) * test.calculate_progress())

        if evaluated_samples == 0:
            return Money(0.0, "USD")

        # Estimate costs per model based on average token usage
        avg_input_tokens = 150
        avg_output_tokens = 100
        total_cost = 0.0

        try:
            # Get cost estimates for completed evaluations
            model_configs = []
            for model in test.configuration.models:
                provider_name = self._extract_provider_name(model, test.configuration)
                model_configs.append({"model_id": model, "provider_name": provider_name})

            cost_estimates = await self.provider_service.get_model_cost_estimates(
                model_configs, evaluated_samples
            )

            total_cost = sum(cost_estimates.values())

        except Exception as e:
            logger.warning(f"Error calculating current cost: {e}")
            # Fallback to simple estimation
            total_cost = (
                evaluated_samples * len(test.configuration.models) * 0.001
            )  # $0.001 per evaluation

        return Money(total_cost, "USD")

    def _extract_provider_name(self, model: str, configuration) -> str:
        """Extract provider name for a model from configuration."""
        model_params = getattr(configuration, "model_parameters", {})
        for key in model_params:
            if model in key:
                return key.split("/")[0]
        return "default"

    async def _estimate_average_evaluation_time(self, test) -> float:
        """Estimate average evaluation time based on completed samples."""
        # This would analyze timing data from completed evaluations
        # For now, return a reasonable default

        # Consider test complexity factors
        base_time = 2.0  # seconds per evaluation

        # Adjust based on number of models (more models = more time)
        model_count = len(test.configuration.models)
        time_per_model = base_time * (1 + (model_count - 1) * 0.3)

        # Adjust based on judge count
        judge_count = getattr(test.configuration, "evaluation_template", {}).get("judge_count", 3)
        time_with_judges = time_per_model * (1 + (judge_count - 1) * 0.2)

        # Add overhead for coordination
        total_time = time_with_judges * 1.2

        return min(total_time, 10.0)  # Cap at 10 seconds per evaluation
