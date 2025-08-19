"""Model inference service for integrating with Agent 2.1 test execution workflow."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ....domain.model_provider.entities.model_provider import ModelProvider
from ....domain.test_management.entities.test_configuration import TestConfiguration
from ....domain.test_management.entities.test_sample import TestSample
from ...dto.model_request_dto import ModelRequestDTO
from ...dto.model_response_dto import ModelResponseDTO
from .model_provider_service import ModelProviderService
from .provider_selector import SelectionCriteria

logger = logging.getLogger(__name__)


class TestContext:
    """Context information for test execution."""

    def __init__(
        self,
        test_id: str,
        test_name: str,
        sample_id: str,
        model_configurations: Dict[str, Any],
        evaluation_config: Optional[Dict[str, Any]] = None,
    ):
        self.test_id = test_id
        self.test_name = test_name
        self.sample_id = sample_id
        self.model_configurations = model_configurations
        self.evaluation_config = evaluation_config or {}
        self.created_at = datetime.utcnow()


class ModelInferenceService:
    """Service for processing test samples through model providers."""

    def __init__(self, model_provider_service: ModelProviderService):
        self.model_provider_service = model_provider_service
        self.logger = logger

    async def process_test_sample(
        self, sample: TestSample, model_config: Dict[str, Any], test_context: TestContext
    ) -> ModelResponseDTO:
        """
        Process a single test sample through a model provider.

        This method is called by Agent 2.1's ProcessSamplesUseCase.

        Args:
            sample: Test sample to process
            model_config: Model configuration containing provider and parameters
            test_context: Test execution context

        Returns:
            Standardized model response
        """
        try:
            # Extract provider and model information from config
            provider_id = model_config.get("provider_id")
            model_id = model_config.get("model_id")
            parameters = model_config.get("parameters", {})

            if not provider_id or not model_id:
                raise ValueError(
                    f"Model configuration missing provider_id or model_id: {model_config}"
                )

            self.logger.debug(
                f"Processing test sample {sample.id} with {provider_id}/{model_id}",
                extra={
                    "test_id": test_context.test_id,
                    "sample_id": str(sample.id),
                    "provider": provider_id,
                    "model": model_id,
                },
            )

            # Create test-specific context
            provider_test_context = {
                "test_id": test_context.test_id,
                "test_name": test_context.test_name,
                "sample_id": str(sample.id),
                "sample_metadata": sample.metadata or {},
                "evaluation_config": test_context.evaluation_config,
                "processing_timestamp": datetime.utcnow().isoformat(),
            }

            # Use the model provider service to make the call
            response = await self.model_provider_service.process_test_sample(
                sample=sample,
                provider_id=provider_id,
                model_id=model_id,
                parameters=parameters,
                test_context=provider_test_context,
            )

            self.logger.info(
                f"Processed test sample {sample.id}: status={response.status.value}, "
                f"tokens={response.total_tokens}, cost=${response.total_cost}",
                extra={
                    "test_id": test_context.test_id,
                    "sample_id": str(sample.id),
                    "response_status": response.status.value,
                    "latency_ms": response.latency_ms,
                },
            )

            return response

        except Exception as e:
            self.logger.error(
                f"Error processing test sample {sample.id}: {e}",
                extra={
                    "test_id": test_context.test_id,
                    "sample_id": str(sample.id),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            # Create error response for integration with Agent 2.1
            return ModelResponseDTO.create_error_response(
                provider_id=model_config.get("provider_id", "unknown"),
                model_id=model_config.get("model_id", "unknown"),
                error_message=f"Test sample processing failed: {str(e)}",
            )

    async def process_test_samples_batch(
        self,
        samples: List[TestSample],
        model_configs: Dict[str, Dict[str, Any]],
        test_context: TestContext,
        batch_size: int = 5,
    ) -> Dict[str, List[ModelResponseDTO]]:
        """
        Process multiple test samples in batches for multiple models.

        Args:
            samples: List of test samples to process
            model_configs: Dictionary mapping model names to configurations
            test_context: Test execution context
            batch_size: Maximum concurrent requests per model

        Returns:
            Dictionary mapping model names to lists of responses
        """
        results = {}

        for model_name, model_config in model_configs.items():
            self.logger.info(
                f"Processing {len(samples)} samples for model {model_name}",
                extra={
                    "test_id": test_context.test_id,
                    "model": model_name,
                    "sample_count": len(samples),
                },
            )

            # Create semaphore for batch size control
            semaphore = asyncio.Semaphore(batch_size)

            async def process_sample_with_semaphore(sample: TestSample) -> ModelResponseDTO:
                async with semaphore:
                    return await self.process_test_sample(sample, model_config, test_context)

            # Process all samples for this model
            tasks = [process_sample_with_semaphore(sample) for sample in samples]

            try:
                model_responses = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle any exceptions
                processed_responses = []
                for i, response in enumerate(model_responses):
                    if isinstance(response, Exception):
                        self.logger.error(
                            f"Error processing sample {samples[i].id} for model {model_name}: {response}",
                            extra={"test_id": test_context.test_id},
                        )
                        error_response = ModelResponseDTO.create_error_response(
                            provider_id=model_config.get("provider_id", "unknown"),
                            model_id=model_config.get("model_id", "unknown"),
                            error_message=f"Batch processing error: {str(response)}",
                        )
                        processed_responses.append(error_response)
                    else:
                        processed_responses.append(response)

                results[model_name] = processed_responses

                # Log batch completion
                successful = sum(1 for r in processed_responses if r.is_successful())
                self.logger.info(
                    f"Completed batch processing for {model_name}: "
                    f"{successful}/{len(processed_responses)} successful",
                    extra={
                        "test_id": test_context.test_id,
                        "model": model_name,
                        "success_rate": (
                            successful / len(processed_responses) if processed_responses else 0
                        ),
                    },
                )

            except Exception as e:
                self.logger.error(
                    f"Batch processing failed for model {model_name}: {e}",
                    extra={"test_id": test_context.test_id},
                    exc_info=True,
                )
                # Create error responses for all samples
                error_responses = [
                    ModelResponseDTO.create_error_response(
                        provider_id=model_config.get("provider_id", "unknown"),
                        model_id=model_config.get("model_id", "unknown"),
                        error_message=f"Batch processing failed: {str(e)}",
                    )
                    for _ in samples
                ]
                results[model_name] = error_responses

        return results

    async def process_test_with_intelligent_routing(
        self,
        samples: List[TestSample],
        model_requirements: List[Dict[str, Any]],
        test_context: TestContext,
        selection_criteria: SelectionCriteria = SelectionCriteria.BALANCED,
    ) -> Dict[str, List[ModelResponseDTO]]:
        """
        Process test samples with intelligent provider routing.

        Args:
            samples: Test samples to process
            model_requirements: List of model requirements (model_id, parameters, etc.)
            test_context: Test execution context
            selection_criteria: Criteria for provider selection

        Returns:
            Dictionary mapping model IDs to response lists
        """
        results = {}

        for model_req in model_requirements:
            model_id = model_req["model_id"]
            parameters = model_req.get("parameters", {})

            self.logger.info(
                f"Processing {len(samples)} samples for model {model_id} with intelligent routing",
                extra={
                    "test_id": test_context.test_id,
                    "model": model_id,
                    "selection_criteria": selection_criteria.value,
                },
            )

            # Process each sample with intelligent routing
            model_responses = []
            for sample in samples:
                try:
                    response = (
                        await self.model_provider_service.call_model_with_intelligent_routing(
                            model_id=model_id,
                            prompt=sample.prompt,
                            parameters=parameters,
                            selection_criteria=selection_criteria,
                            test_context={
                                "test_id": test_context.test_id,
                                "sample_id": str(sample.id),
                                "sample_metadata": sample.metadata or {},
                            },
                        )
                    )
                    model_responses.append(response)

                    # Small delay to avoid overwhelming providers
                    await asyncio.sleep(0.1)

                except Exception as e:
                    self.logger.error(
                        f"Intelligent routing failed for sample {sample.id}: {e}",
                        extra={"test_id": test_context.test_id},
                        exc_info=True,
                    )
                    error_response = ModelResponseDTO.create_error_response(
                        provider_id="intelligent_routing",
                        model_id=model_id,
                        error_message=f"Intelligent routing failed: {str(e)}",
                    )
                    model_responses.append(error_response)

            results[model_id] = model_responses

            # Log completion
            successful = sum(1 for r in model_responses if r.is_successful())
            self.logger.info(
                f"Completed intelligent routing for {model_id}: "
                f"{successful}/{len(model_responses)} successful",
                extra={
                    "test_id": test_context.test_id,
                    "model": model_id,
                    "success_rate": successful / len(model_responses) if model_responses else 0,
                },
            )

        return results

    def calculate_test_metrics(
        self, responses: Dict[str, List[ModelResponseDTO]]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for test execution.

        Args:
            responses: Dictionary mapping model names to response lists

        Returns:
            Dictionary containing test metrics
        """
        total_requests = 0
        successful_requests = 0
        total_latency = 0.0
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        model_metrics = {}

        for model_name, model_responses in responses.items():
            model_successful = 0
            model_latency = 0.0
            model_cost = 0.0
            model_input_tokens = 0
            model_output_tokens = 0

            for response in model_responses:
                total_requests += 1

                if response.is_successful():
                    successful_requests += 1
                    model_successful += 1

                if response.latency_ms:
                    total_latency += response.latency_ms
                    model_latency += response.latency_ms

                if response.total_cost:
                    cost_value = float(response.total_cost)
                    total_cost += cost_value
                    model_cost += cost_value

                if response.input_tokens:
                    total_input_tokens += response.input_tokens
                    model_input_tokens += response.input_tokens

                if response.output_tokens:
                    total_output_tokens += response.output_tokens
                    model_output_tokens += response.output_tokens

            # Calculate model-specific metrics
            model_count = len(model_responses)
            model_metrics[model_name] = {
                "request_count": model_count,
                "successful_requests": model_successful,
                "success_rate": model_successful / model_count if model_count > 0 else 0,
                "average_latency_ms": model_latency / model_count if model_count > 0 else 0,
                "total_cost": model_cost,
                "average_cost_per_request": model_cost / model_count if model_count > 0 else 0,
                "total_input_tokens": model_input_tokens,
                "total_output_tokens": model_output_tokens,
                "total_tokens": model_input_tokens + model_output_tokens,
                "average_tokens_per_request": (
                    (model_input_tokens + model_output_tokens) / model_count
                    if model_count > 0
                    else 0
                ),
            }

        # Overall test metrics
        overall_metrics = {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "overall_success_rate": (
                successful_requests / total_requests if total_requests > 0 else 0
            ),
            "average_latency_ms": total_latency / total_requests if total_requests > 0 else 0,
            "total_cost": total_cost,
            "average_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "models_tested": len(responses),
            "model_metrics": model_metrics,
        }

        return overall_metrics

    def create_test_context_from_configuration(
        self, test_id: str, test_name: str, sample_id: str, test_configuration: TestConfiguration
    ) -> TestContext:
        """
        Create a TestContext from a TestConfiguration entity.

        Args:
            test_id: Test identifier
            test_name: Test name
            sample_id: Sample identifier
            test_configuration: Test configuration entity

        Returns:
            TestContext for model inference
        """
        # Extract model configurations from the test configuration
        model_configurations = {}

        # This assumes the test configuration has model parameters
        if hasattr(test_configuration, "model_parameters"):
            model_configurations = test_configuration.model_parameters

        # Extract evaluation configuration
        evaluation_config = {}
        if hasattr(test_configuration, "evaluation_template"):
            evaluation_config = test_configuration.evaluation_template

        return TestContext(
            test_id=test_id,
            test_name=test_name,
            sample_id=sample_id,
            model_configurations=model_configurations,
            evaluation_config=evaluation_config,
        )
