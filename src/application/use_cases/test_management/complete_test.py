"""Use case for completing test execution and generating final results."""

import logging
from uuid import UUID

from ....domain.test_management.exceptions import BusinessRuleViolation, InvalidStateTransition
from ....domain.test_management.value_objects.test_status import TestStatus
from ...dto.test_configuration_dto import TestResultDTO
from ...interfaces.domain_event_publisher import DomainEventPublisher
from ...interfaces.unit_of_work import UnitOfWork

logger = logging.getLogger(__name__)


class CompleteTestUseCase:
    """Use case for completing test execution with result finalization."""

    def __init__(self, uow: UnitOfWork, event_publisher: DomainEventPublisher):
        self.uow = uow
        self.event_publisher = event_publisher

    async def execute(self, test_id: UUID, force_completion: bool = False) -> TestResultDTO:
        """Execute test completion workflow with result validation."""
        try:
            logger.info(f"Completing test: {test_id} (force={force_completion})")

            async with self.uow:
                # Step 1: Load test aggregate
                test = await self.uow.tests.find_by_id(test_id)
                if not test:
                    logger.warning(f"Test not found: {test_id}")
                    return TestResultDTO(
                        test_id=test_id,
                        status="not_found",
                        created_test=False,
                        errors=[f"Test {test_id} not found"],
                    )

                # Step 2: Validate test can be completed
                validation_errors = await self._validate_completion_conditions(
                    test, force_completion
                )
                if validation_errors:
                    logger.warning(f"Test completion validation failed: {validation_errors}")
                    return TestResultDTO(
                        test_id=test_id,
                        status="validation_failed",
                        created_test=False,
                        errors=validation_errors,
                    )

                # Step 3: Generate and validate final results
                logger.debug("Generating final test results")
                results_summary = await self._generate_results_summary(test)

                # Step 4: Complete the test
                logger.debug("Marking test as completed")
                test.complete()

                # Step 5: Persist final state
                await self.uow.tests.save(test)

                # Step 6: Store final results in analytics domain
                await self._store_final_results(test, results_summary)

                await self.uow.commit()

            # Step 7: Publish completion events
            logger.debug("Publishing completion events")
            domain_events = test.get_domain_events()
            await self.event_publisher.publish_all(domain_events)
            test.clear_domain_events()

            # Step 8: Trigger post-completion workflows
            await self._trigger_post_completion_workflows(test, results_summary)

            logger.info(f"Test completed successfully: {test_id}")
            return TestResultDTO(test_id=test.id, status=test.status.value, created_test=False)

        except InvalidStateTransition as e:
            logger.warning(f"Invalid state transition during test completion: {e}")
            return TestResultDTO(
                test_id=test_id,
                status="invalid_state_transition",
                created_test=False,
                errors=[str(e)],
            )
        except BusinessRuleViolation as e:
            logger.warning(f"Business rule violation during test completion: {e}")
            return TestResultDTO(
                test_id=test_id,
                status="business_rule_violation",
                created_test=False,
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Unexpected error during test completion: {e}", exc_info=True)
            return TestResultDTO(
                test_id=test_id,
                status="system_error",
                created_test=False,
                errors=[f"System error: {str(e)}"],
            )

    async def _validate_completion_conditions(self, test, force_completion: bool) -> list[str]:
        """Validate that test can be completed."""
        errors = []

        # Check test is in correct state
        if test.status != TestStatus.RUNNING:
            errors.append(
                f"Test must be in RUNNING state to complete, currently {test.status.value}"
            )
            return errors  # Can't continue validation if not running

        # If not forcing completion, check that all samples are evaluated
        if not force_completion:
            progress = test.calculate_progress()
            if progress < 1.0:
                errors.append(
                    f"Test is only {progress:.1%} complete. "
                    "Use force_completion=True to complete with partial results."
                )

        # Check minimum completion threshold for statistical validity
        progress = test.calculate_progress()
        min_completion_threshold = 0.8  # 80% minimum for valid results

        if progress < min_completion_threshold and not force_completion:
            errors.append(
                f"Test completion is {progress:.1%}, below minimum threshold of "
                f"{min_completion_threshold:.1%} for valid statistical results"
            )

        # Validate that we have meaningful data for comparison
        if progress > 0:
            model_scores = test.get_model_scores()
            models_with_data = sum(1 for score in model_scores.values() if score > 0)

            if models_with_data < 2:
                errors.append(
                    "Insufficient data for comparison - at least 2 models need evaluation results"
                )

        return errors

    async def _generate_results_summary(self, test) -> dict:
        """Generate comprehensive results summary for the test."""
        stats = test.get_test_statistics()
        model_scores = test.get_model_scores()

        # Calculate statistical significance (simplified)
        significance_results = await self._calculate_statistical_significance(test)

        # Generate winner determination
        winner_analysis = self._determine_winner(model_scores, significance_results)

        # Calculate confidence intervals for each model
        confidence_intervals = await self._calculate_confidence_intervals(test)

        results_summary = {
            "test_id": str(test.id),
            "test_name": test.name,
            "completion_timestamp": test.completed_at.isoformat() if test.completed_at else None,
            "total_samples": stats["total_samples"],
            "evaluated_samples": stats["evaluated_samples"],
            "completion_rate": stats["progress"],
            "model_scores": model_scores,
            "winner_analysis": winner_analysis,
            "statistical_significance": significance_results,
            "confidence_intervals": confidence_intervals,
            "duration_seconds": stats.get("duration_seconds"),
            "difficulty_distribution": stats.get("difficulty_distribution", {}),
            "overall_score": stats.get("overall_score", 0.0),
        }

        return results_summary

    async def _calculate_statistical_significance(self, test) -> dict:
        """Calculate statistical significance of differences between models."""
        # This is a simplified implementation - in practice would use proper statistical tests
        model_scores = test.get_model_scores()
        models = list(model_scores.keys())

        if len(models) < 2:
            return {"significant_differences": [], "p_values": {}, "method": "insufficient_data"}

        # Simple heuristic for demonstration - in practice would use t-tests, etc.
        significant_differences = []
        p_values = {}

        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                score_diff = abs(model_scores[model_a] - model_scores[model_b])

                # Simple significance test based on score difference and sample size
                sample_count = len([s for s in test.samples if s.is_evaluated])

                # Simplified p-value calculation (not statistically rigorous)
                if sample_count > 50 and score_diff > 0.05:  # 5% difference with good sample size
                    p_value = max(0.001, 0.05 - (score_diff * 10))  # Simplified calculation
                    significant_differences.append(
                        {
                            "model_a": model_a,
                            "model_b": model_b,
                            "score_difference": score_diff,
                            "significant": p_value < 0.05,
                        }
                    )
                    p_values[f"{model_a}_vs_{model_b}"] = p_value

        return {
            "significant_differences": significant_differences,
            "p_values": p_values,
            "method": "simplified_score_comparison",
            "alpha": 0.05,
        }

    def _determine_winner(self, model_scores: dict, significance_results: dict) -> dict:
        """Determine winning model based on scores and significance."""
        if not model_scores:
            return {"winner": None, "confidence": "no_data", "margin": 0.0}

        # Find highest scoring model
        best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
        best_score = model_scores[best_model]

        # Find second best
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_models) < 2:
            return {"winner": best_model, "confidence": "only_one_model", "margin": 0.0}

        second_best_score = sorted_models[1][1]
        margin = best_score - second_best_score

        # Determine confidence based on statistical significance
        confidence = "low"
        if significance_results.get("significant_differences"):
            for diff in significance_results["significant_differences"]:
                if (diff["model_a"] == best_model or diff["model_b"] == best_model) and diff[
                    "significant"
                ]:
                    confidence = "high"
                    break

        # Adjust confidence based on margin
        if margin > 0.1:  # 10% margin
            confidence = "high" if confidence != "low" else "medium"
        elif margin > 0.05:  # 5% margin
            confidence = "medium" if confidence != "high" else "medium"

        return {
            "winner": best_model,
            "confidence": confidence,
            "margin": margin,
            "runner_up": sorted_models[1][0] if len(sorted_models) > 1 else None,
        }

    async def _calculate_confidence_intervals(self, test) -> dict:
        """Calculate confidence intervals for model scores."""
        # Simplified confidence interval calculation
        model_scores = test.get_model_scores()
        confidence_intervals = {}

        sample_count = len([s for s in test.samples if s.is_evaluated])

        for model, score in model_scores.items():
            if sample_count > 10:
                # Simplified standard error calculation
                # In practice, would calculate actual standard deviation from samples
                standard_error = 0.1 / (sample_count**0.5)  # Simplified
                margin_of_error = 1.96 * standard_error  # 95% confidence

                confidence_intervals[model] = {
                    "lower": max(0.0, score - margin_of_error),
                    "upper": min(1.0, score + margin_of_error),
                    "confidence_level": 0.95,
                }
            else:
                # Wide confidence interval for small samples
                confidence_intervals[model] = {
                    "lower": max(0.0, score - 0.2),
                    "upper": min(1.0, score + 0.2),
                    "confidence_level": 0.95,
                }

        return confidence_intervals

    async def _store_final_results(self, test, results_summary: dict) -> None:
        """Store final test results in analytics domain."""
        # This would create analytics entities for the completed test
        # For now, we'll log the results
        logger.info(f"Storing final results for test {test.id}")
        logger.debug(f"Results summary: {results_summary}")

        # In a full implementation, this would:
        # 1. Create AnalysisResult aggregate
        # 2. Store model performance data
        # 3. Create statistical test results
        # 4. Generate insights and recommendations

    async def _trigger_post_completion_workflows(self, test, results_summary: dict) -> None:
        """Trigger workflows that should run after test completion."""
        # This could trigger various post-completion activities
        logger.info(f"Triggering post-completion workflows for test {test.id}")

        # Examples of what could be triggered:
        # - Report generation
        # - Notification sending
        # - Model performance tracking updates
        # - Cost accounting
        # - Recommendation generation for future tests

        # Log the completion for now
        winner = results_summary.get("winner_analysis", {}).get("winner")
        if winner:
            logger.info(f"Test {test.id} completed. Winner: {winner}")
        else:
            logger.info(f"Test {test.id} completed with no clear winner")
