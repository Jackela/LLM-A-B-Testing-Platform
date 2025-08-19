"""Use case for processing samples during test execution."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from ....domain.test_management.entities.test import Test
from ....domain.test_management.entities.test_sample import TestSample
from ....domain.test_management.exceptions import BusinessRuleViolation
from ....domain.test_management.value_objects.test_status import TestStatus
from ...interfaces.domain_event_publisher import DomainEventPublisher
from ...interfaces.unit_of_work import UnitOfWork
from ...services.model_provider_service import ModelProviderService

logger = logging.getLogger(__name__)


class ProcessSamplesUseCase:
    """Use case for processing samples through model inference and evaluation."""

    def __init__(
        self,
        uow: UnitOfWork,
        event_publisher: DomainEventPublisher,
        provider_service: ModelProviderService,
    ):
        self.uow = uow
        self.event_publisher = event_publisher
        self.provider_service = provider_service

    async def execute(self, test_id: UUID, batch_size: int = 10) -> Dict:
        """Execute sample processing with parallel execution and error handling."""
        try:
            logger.info(f"Starting sample processing for test: {test_id}")

            async with self.uow:
                # Step 1: Load test aggregate
                test = await self.uow.tests.find_by_id(test_id)
                if not test:
                    logger.error(f"Test not found: {test_id}")
                    return {
                        "success": False,
                        "error": f"Test {test_id} not found",
                        "processed_samples": 0,
                        "total_samples": 0,
                        "failed_samples": 0,
                    }

                # Step 2: Validate test is in correct state for processing
                if test.status != TestStatus.RUNNING:
                    logger.error(f"Test {test_id} is not running (status: {test.status.value})")
                    return {
                        "success": False,
                        "error": f"Test must be in RUNNING state, currently {test.status.value}",
                        "processed_samples": 0,
                        "total_samples": len(test.samples),
                        "failed_samples": 0,
                    }

                # Step 3: Get unprocessed samples
                unprocessed_samples = await self._get_unprocessed_samples(test)
                if not unprocessed_samples:
                    logger.info(f"No unprocessed samples found for test {test_id}")
                    return {
                        "success": True,
                        "message": "All samples already processed",
                        "processed_samples": 0,
                        "total_samples": len(test.samples),
                        "failed_samples": 0,
                    }

                logger.info(f"Processing {len(unprocessed_samples)} unprocessed samples")

                # Step 4: Get providers for the test
                providers = await self.provider_service.get_providers_for_test(test.configuration)

                # Step 5: Process samples in batches
                processing_results = await self._process_samples_in_batches(
                    test, unprocessed_samples, providers, batch_size
                )

                # Step 6: Update test progress and save
                await self.uow.tests.save(test)
                await self.uow.commit()

            # Step 7: Publish progress events
            domain_events = test.get_domain_events()
            await self.event_publisher.publish_all(domain_events)
            test.clear_domain_events()

            # Step 8: Check if test is complete
            await self._check_test_completion(test_id)

            logger.info(
                f"Sample processing completed for test {test_id}: "
                f"{processing_results['processed']} processed, "
                f"{processing_results['failed']} failed"
            )

            return {
                "success": True,
                "processed_samples": processing_results["processed"],
                "total_samples": len(test.samples),
                "failed_samples": processing_results["failed"],
                "errors": processing_results["errors"],
            }

        except Exception as e:
            logger.error(f"Error processing samples for test {test_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "processed_samples": 0,
                "total_samples": 0,
                "failed_samples": 0,
            }

    async def _get_unprocessed_samples(self, test: Test) -> List[TestSample]:
        """Get samples that haven't been processed yet."""
        unprocessed = []

        for sample in test.samples:
            # Check if sample has been evaluated for all models
            missing_evaluations = False
            for model in test.configuration.models:
                if not sample.has_evaluation_for_model(model):
                    missing_evaluations = True
                    break

            if missing_evaluations:
                unprocessed.append(sample)

        return unprocessed

    async def _process_samples_in_batches(
        self, test: Test, samples: List[TestSample], providers: List, batch_size: int
    ) -> Dict:
        """Process samples in parallel batches."""
        processed = 0
        failed = 0
        errors = []

        # Process samples in batches
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch)} samples")

            try:
                batch_results = await self._process_sample_batch(test, batch, providers)
                processed += batch_results["processed"]
                failed += batch_results["failed"]
                errors.extend(batch_results["errors"])

                # Brief pause between batches to avoid overwhelming providers
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error processing batch: {e}", exc_info=True)
                failed += len(batch)
                errors.append(f"Batch processing error: {str(e)}")

        return {"processed": processed, "failed": failed, "errors": errors}

    async def _process_sample_batch(
        self, test: Test, batch: List[TestSample], providers: List
    ) -> Dict:
        """Process a single batch of samples."""
        processed = 0
        failed = 0
        errors = []

        # Create tasks for parallel processing
        tasks = []
        for sample in batch:
            task = self._process_single_sample(test, sample, providers)
            tasks.append(task)

        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing sample {i}: {result}")
                failed += 1
                errors.append(f"Sample {i}: {str(result)}")
            elif result.get("success", False):
                processed += 1
            else:
                failed += 1
                errors.append(f"Sample {i}: {result.get('error', 'Unknown error')}")

        return {"processed": processed, "failed": failed, "errors": errors}

    async def _process_single_sample(self, test: Test, sample: TestSample, providers: List) -> Dict:
        """Process a single sample through all models and evaluation."""
        try:
            # Step 1: Generate responses from all models
            model_responses = {}
            for model in test.configuration.models:
                provider = self._find_provider_for_model(providers, model, test.configuration)
                if not provider:
                    raise ValueError(f"Provider not found for model {model}")

                # Generate model response
                response = await self._generate_model_response(provider, model, sample)
                model_responses[model] = response

            # Step 2: Evaluate responses
            evaluation_results = await self._evaluate_sample_responses(
                test, sample, model_responses
            )

            # Step 3: Update sample with results
            await self._update_sample_with_results(sample, model_responses, evaluation_results)

            return {
                "success": True,
                "sample_id": sample.id,
                "models_processed": len(model_responses),
                "evaluations_completed": len(evaluation_results),
            }

        except Exception as e:
            logger.error(f"Error processing sample {sample.id}: {e}")

            # Mark sample as failed in metadata
            if not sample.metadata:
                sample.metadata = {}
            sample.metadata["processing_error"] = str(e)
            sample.metadata["processing_failed"] = True

            return {"success": False, "error": str(e), "sample_id": sample.id}

    def _find_provider_for_model(self, providers: List, model: str, configuration) -> Optional:
        """Find the provider that supports the given model."""
        # Extract provider name from configuration
        model_params = getattr(configuration, "model_parameters", {})
        for key in model_params:
            if model in key:
                provider_name = key.split("/")[0]
                return next((p for p in providers if p.name == provider_name), None)
        return None

    async def _generate_model_response(self, provider, model: str, sample: TestSample) -> Dict:
        """Generate response from a model for a sample."""
        try:
            # Get model parameters from test configuration
            # This is simplified - in practice would extract from configuration
            parameters = {"max_tokens": 150, "temperature": 0.7}

            # Call the model (this would be handled by infrastructure layer)
            # For now, we simulate the response
            response = provider.call_model(model, sample.prompt, **parameters)

            # In practice, this would wait for the actual API response
            # For simulation, create a mock response
            mock_response = {
                "text": f"Mock response from {model} for prompt: {sample.prompt[:50]}...",
                "token_count": len(sample.prompt.split()) + 20,
                "latency_ms": 150,
                "provider": provider.name,
                "model": model,
            }

            return mock_response

        except Exception as e:
            logger.error(f"Error generating response from {model}: {e}")
            raise

    async def _evaluate_sample_responses(
        self, test: Test, sample: TestSample, model_responses: Dict
    ) -> Dict:
        """Evaluate model responses for a sample."""
        evaluation_results = {}

        # Get evaluation configuration
        eval_config = getattr(test.configuration, "evaluation_template", {})
        judge_count = eval_config.get("judge_count", 3)

        try:
            # In practice, this would interface with the evaluation domain
            # For now, we'll simulate evaluation results

            for model, response in model_responses.items():
                # Simulate judge evaluations
                judge_scores = []
                for judge_id in range(judge_count):
                    # Mock evaluation based on response length and content
                    base_score = 0.7  # Base score
                    length_bonus = min(0.2, len(response.get("text", "")) / 500)
                    random_variation = 0.1 * ((judge_id + 1) % 3 - 1)  # -0.1 to 0.1

                    score = max(0.0, min(1.0, base_score + length_bonus + random_variation))
                    judge_scores.append(score)

                # Calculate consensus
                avg_score = sum(judge_scores) / len(judge_scores)
                consensus = self._calculate_consensus(judge_scores)

                evaluation_results[model] = {
                    "judge_scores": judge_scores,
                    "average_score": avg_score,
                    "consensus": consensus,
                    "quality_threshold_met": consensus
                    >= eval_config.get("consensus_threshold", 0.7),
                }

        except Exception as e:
            logger.error(f"Error evaluating sample {sample.id}: {e}")
            raise

        return evaluation_results

    def _calculate_consensus(self, scores: List[float]) -> float:
        """Calculate consensus among judge scores."""
        if len(scores) <= 1:
            return 1.0

        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance**0.5

        # Consensus is inversely related to standard deviation
        # High consensus = low variance
        consensus = max(0.0, 1.0 - (std_dev * 2))  # Normalize to 0-1 range
        return consensus

    async def _update_sample_with_results(
        self, sample: TestSample, model_responses: Dict, evaluation_results: Dict
    ) -> None:
        """Update sample with processing results."""
        # Update sample metadata with results
        if not sample.metadata:
            sample.metadata = {}

        sample.metadata["model_responses"] = {
            model: {
                "text": response.get("text", ""),
                "token_count": response.get("token_count", 0),
                "latency_ms": response.get("latency_ms", 0),
            }
            for model, response in model_responses.items()
        }

        sample.metadata["evaluation_results"] = evaluation_results
        sample.metadata["processed_at"] = asyncio.get_event_loop().time()
        sample.metadata["is_evaluated"] = True

        # Mark sample as evaluated for all models
        for model in model_responses.keys():
            # This would update the sample's evaluation status
            # For now, we'll set it in metadata
            sample.metadata[f"{model}_evaluated"] = True

    async def _check_test_completion(self, test_id: UUID) -> None:
        """Check if test processing is complete and handle completion."""
        try:
            async with self.uow:
                test = await self.uow.tests.find_by_id(test_id)
                if not test:
                    return

                # Check if all samples are processed
                progress = test.calculate_progress()
                if progress >= 1.0:  # 100% complete
                    logger.info(f"Test {test_id} processing complete, triggering completion")

                    # This would trigger the CompleteTestUseCase
                    # For now, we'll just log
                    logger.info(f"Test {test_id} ready for completion")
                else:
                    logger.debug(f"Test {test_id} progress: {progress:.2%}")

        except Exception as e:
            logger.error(f"Error checking test completion: {e}")

    async def get_processing_status(self, test_id: UUID) -> Dict:
        """Get current processing status for a test."""
        try:
            async with self.uow:
                test = await self.uow.tests.find_by_id(test_id)
                if not test:
                    return {"error": f"Test {test_id} not found"}

                total_samples = len(test.samples)
                progress = test.calculate_progress()
                evaluated_samples = int(total_samples * progress)

                # Calculate processing statistics
                processing_stats = {
                    "test_id": str(test_id),
                    "status": test.status.value,
                    "total_samples": total_samples,
                    "evaluated_samples": evaluated_samples,
                    "remaining_samples": total_samples - evaluated_samples,
                    "progress_percentage": progress * 100,
                    "estimated_completion_time": test.estimate_remaining_time(),
                }

                # Add model-specific progress
                model_progress = {}
                for model in test.configuration.models:
                    model_evaluated = sum(
                        1
                        for sample in test.samples
                        if sample.metadata and sample.metadata.get(f"{model}_evaluated", False)
                    )
                    model_progress[model] = {
                        "evaluated": model_evaluated,
                        "progress": model_evaluated / total_samples if total_samples > 0 else 0,
                    }

                processing_stats["model_progress"] = model_progress

                return processing_stats

        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return {"error": str(e)}
