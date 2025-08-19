"""Use case for adding samples to tests."""

import logging
from typing import List

from ....domain.test_management.entities.test_sample import TestSample
from ....domain.test_management.exceptions import BusinessRuleViolation, InvalidStateTransition
from ...dto.test_configuration_dto import AddSamplesCommandDTO, TestResultDTO
from ...interfaces.domain_event_publisher import DomainEventPublisher
from ...interfaces.unit_of_work import UnitOfWork

logger = logging.getLogger(__name__)


class AddSamplesUseCase:
    """Use case for adding samples to tests with validation."""

    def __init__(self, uow: UnitOfWork, event_publisher: DomainEventPublisher):
        self.uow = uow
        self.event_publisher = event_publisher

    async def execute(self, command: AddSamplesCommandDTO) -> TestResultDTO:
        """Execute sample addition with comprehensive validation."""
        try:
            logger.info(f"Adding {len(command.samples)} samples to test: {command.test_id}")

            async with self.uow:
                # Step 1: Load test aggregate
                test = await self.uow.tests.find_by_id(command.test_id)
                if not test:
                    logger.warning(f"Test not found: {command.test_id}")
                    return TestResultDTO(
                        test_id=command.test_id,
                        status="not_found",
                        created_test=False,
                        errors=[f"Test {command.test_id} not found"],
                    )

                # Step 2: Validate test can accept new samples
                logger.debug("Validating sample addition permissions")
                validation_errors = await self._validate_sample_addition(test, command.samples)
                if validation_errors:
                    logger.warning(f"Sample addition validation failed: {validation_errors}")
                    return TestResultDTO(
                        test_id=command.test_id,
                        status="validation_failed",
                        created_test=False,
                        errors=validation_errors,
                    )

                # Step 3: Create domain samples with validation
                logger.debug("Creating domain sample entities")
                new_samples = await self._create_domain_samples(command.samples, test)

                # Step 4: Add samples to test with duplicate checking
                logger.debug("Adding samples to test aggregate")
                added_count = 0
                skipped_duplicates = 0

                for sample in new_samples:
                    try:
                        test.add_sample(sample)
                        added_count += 1
                    except BusinessRuleViolation as e:
                        if "already exists" in str(e):
                            skipped_duplicates += 1
                            logger.debug(f"Skipped duplicate sample: {sample.id}")
                        else:
                            raise  # Re-raise other business rule violations

                # Step 5: Validate final test state
                logger.debug("Validating final test state")
                final_validation = await self._validate_final_test_state(test)
                if final_validation:
                    logger.warning(f"Final test state validation failed: {final_validation}")
                    return TestResultDTO(
                        test_id=command.test_id,
                        status="final_validation_failed",
                        created_test=False,
                        errors=final_validation,
                    )

                # Step 6: Persist changes
                await self.uow.tests.save(test)
                await self.uow.commit()

            # Step 7: Publish events
            logger.debug("Publishing domain events")
            domain_events = test.get_domain_events()
            await self.event_publisher.publish_all(domain_events)
            test.clear_domain_events()

            # Step 8: Log results
            logger.info(
                f"Sample addition completed for test {command.test_id}: "
                f"{added_count} added, {skipped_duplicates} duplicates skipped"
            )

            result_errors = []
            if skipped_duplicates > 0:
                result_errors.append(f"{skipped_duplicates} duplicate samples were skipped")

            return TestResultDTO(
                test_id=command.test_id,
                status=test.status.value,
                created_test=False,
                errors=result_errors,
            )

        except InvalidStateTransition as e:
            logger.warning(f"Invalid state transition during sample addition: {e}")
            return TestResultDTO(
                test_id=command.test_id,
                status="invalid_state_transition",
                created_test=False,
                errors=[str(e)],
            )
        except BusinessRuleViolation as e:
            logger.warning(f"Business rule violation during sample addition: {e}")
            return TestResultDTO(
                test_id=command.test_id,
                status="business_rule_violation",
                created_test=False,
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Unexpected error during sample addition: {e}", exc_info=True)
            return TestResultDTO(
                test_id=command.test_id,
                status="system_error",
                created_test=False,
                errors=[f"System error: {str(e)}"],
            )

    async def _validate_sample_addition(self, test, sample_dtos) -> List[str]:
        """Validate that samples can be added to the test."""
        errors = []

        # Check test state allows sample addition
        if not test.can_be_modified():
            errors.append(
                f"Test in {test.status.value} state cannot accept new samples. "
                "Samples can only be added to DRAFT tests."
            )
            return errors  # Can't continue validation if state is wrong

        # Check total sample count limits
        current_count = len(test.samples)
        new_count = len(sample_dtos)
        total_count = current_count + new_count

        if total_count > 10000:
            errors.append(
                f"Total sample count would exceed limit: {total_count} > 10,000 "
                f"(current: {current_count}, adding: {new_count})"
            )

        # Validate individual samples
        for i, sample_dto in enumerate(sample_dtos):
            sample_errors = self._validate_individual_sample(sample_dto, i)
            errors.extend(sample_errors)

        # Check for duplicates within the new samples
        prompts = [sample.prompt for sample in sample_dtos]
        unique_prompts = set(prompts)
        if len(prompts) != len(unique_prompts):
            errors.append(f"Duplicate prompts found in new samples")

        # Check for duplicates with existing samples
        existing_prompts = {sample.prompt for sample in test.samples}
        duplicate_prompts = [prompt for prompt in prompts if prompt in existing_prompts]
        if duplicate_prompts:
            errors.append(f"Found {len(duplicate_prompts)} prompts that already exist in test")

        return errors

    def _validate_individual_sample(self, sample_dto, index: int) -> List[str]:
        """Validate an individual sample DTO."""
        errors = []

        # Prompt validation
        if not sample_dto.prompt.strip():
            errors.append(f"Sample {index + 1}: Prompt cannot be empty")
        elif len(sample_dto.prompt) > 10000:
            errors.append(f"Sample {index + 1}: Prompt exceeds 10,000 character limit")
        elif len(sample_dto.prompt) < 10:
            errors.append(f"Sample {index + 1}: Prompt is too short (minimum 10 characters)")

        # Expected output validation (if provided)
        if sample_dto.expected_output is not None:
            if len(sample_dto.expected_output) > 5000:
                errors.append(f"Sample {index + 1}: Expected output exceeds 5,000 character limit")

        # Metadata validation
        if sample_dto.metadata:
            if not isinstance(sample_dto.metadata, dict):
                errors.append(f"Sample {index + 1}: Metadata must be a dictionary")
            elif len(str(sample_dto.metadata)) > 2000:
                errors.append(f"Sample {index + 1}: Metadata is too large")

        return errors

    async def _create_domain_samples(self, sample_dtos, test) -> List[TestSample]:
        """Create domain sample entities from DTOs."""
        domain_samples = []

        for sample_dto in sample_dtos:
            # Create domain sample
            sample = TestSample.create(
                prompt=sample_dto.prompt.strip(),
                expected_output=(
                    sample_dto.expected_output.strip() if sample_dto.expected_output else None
                ),
                difficulty=sample_dto.difficulty,
                metadata=sample_dto.metadata or {},
            )

            domain_samples.append(sample)

        return domain_samples

    async def _validate_final_test_state(self, test) -> List[str]:
        """Validate the final state of test after adding samples."""
        errors = []

        # Check minimum sample requirements
        total_samples = len(test.samples)
        if total_samples < 10:
            errors.append(f"Test has insufficient samples: {total_samples} < 10 minimum required")

        # Validate sample distribution by difficulty
        difficulty_counts = {}
        for sample in test.samples:
            difficulty = sample.difficulty.value
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

        # Check for reasonable difficulty distribution
        if len(difficulty_counts) == 1:
            single_difficulty = list(difficulty_counts.keys())[0]
            logger.warning(f"All samples have same difficulty: {single_difficulty}")
            # This is a warning, not an error

        # Validate against test configuration if available
        if hasattr(test, "configuration") and test.configuration:
            model_count = len(test.configuration.models)
            samples_per_model = total_samples // model_count

            if samples_per_model < 5:
                errors.append(
                    f"Insufficient samples per model: {samples_per_model} "
                    f"(total: {total_samples}, models: {model_count})"
                )

        return errors

    async def batch_add_samples(
        self, command: AddSamplesCommandDTO, batch_size: int = 100
    ) -> List[TestResultDTO]:
        """Add samples in batches for large datasets."""
        logger.info(f"Batch adding {len(command.samples)} samples in batches of {batch_size}")

        results = []
        total_samples = command.samples

        for i in range(0, len(total_samples), batch_size):
            batch_samples = total_samples[i : i + batch_size]
            batch_command = AddSamplesCommandDTO(
                test_id=command.test_id, samples=batch_samples, creator_id=command.creator_id
            )

            logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch_samples)} samples")
            result = await self.execute(batch_command)
            results.append(result)

            # Stop if we encounter errors
            if not result.created_test and result.errors:
                logger.error(f"Batch processing stopped due to errors: {result.errors}")
                break

        return results

    async def validate_samples_only(self, sample_dtos, test_id=None) -> dict:
        """Validate samples without adding them to a test."""
        logger.debug(f"Validating {len(sample_dtos)} samples")

        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "sample_count": len(sample_dtos),
            "duplicate_count": 0,
        }

        # Validate individual samples
        for i, sample_dto in enumerate(sample_dtos):
            sample_errors = self._validate_individual_sample(sample_dto, i)
            validation_result["errors"].extend(sample_errors)

        # Check for duplicates within samples
        prompts = [sample.prompt for sample in sample_dtos]
        unique_prompts = set(prompts)
        duplicate_count = len(prompts) - len(unique_prompts)
        validation_result["duplicate_count"] = duplicate_count

        if duplicate_count > 0:
            validation_result["warnings"].append(
                f"{duplicate_count} duplicate prompts found in sample set"
            )

        # If test_id provided, check against existing samples
        if test_id:
            try:
                async with self.uow:
                    test = await self.uow.tests.find_by_id(test_id)
                    if test:
                        existing_prompts = {sample.prompt for sample in test.samples}
                        existing_duplicates = sum(
                            1 for prompt in prompts if prompt in existing_prompts
                        )
                        if existing_duplicates > 0:
                            validation_result["warnings"].append(
                                f"{existing_duplicates} prompts already exist in test"
                            )
            except Exception as e:
                validation_result["warnings"].append(f"Could not check existing samples: {e}")

        # Overall validation
        validation_result["is_valid"] = len(validation_result["errors"]) == 0

        return validation_result
