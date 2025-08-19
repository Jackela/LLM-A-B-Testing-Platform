"""Use case for updating test configuration."""

import logging
from uuid import UUID

from ....domain.test_management.entities.test_configuration import TestConfiguration
from ....domain.test_management.exceptions import BusinessRuleViolation, InvalidStateTransition
from ...dto.test_configuration_dto import TestResultDTO, UpdateTestConfigurationCommandDTO
from ...interfaces.domain_event_publisher import DomainEventPublisher
from ...interfaces.unit_of_work import UnitOfWork
from ...services.test_validation_service import TestValidationService

logger = logging.getLogger(__name__)


class UpdateConfigurationUseCase:
    """Use case for updating test configuration with validation."""

    def __init__(
        self,
        uow: UnitOfWork,
        event_publisher: DomainEventPublisher,
        validation_service: TestValidationService,
    ):
        self.uow = uow
        self.event_publisher = event_publisher
        self.validation_service = validation_service

    async def execute(self, command: UpdateTestConfigurationCommandDTO) -> TestResultDTO:
        """Execute test configuration update with comprehensive validation."""
        try:
            logger.info(f"Updating configuration for test: {command.test_id}")

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

                # Step 2: Validate test can be modified
                logger.debug("Validating test modification permissions")
                modification_errors = await self._validate_modification_permissions(test)
                if modification_errors:
                    logger.warning(f"Test modification validation failed: {modification_errors}")
                    return TestResultDTO(
                        test_id=command.test_id,
                        status="modification_not_allowed",
                        created_test=False,
                        errors=modification_errors,
                    )

                # Step 3: Validate new configuration
                logger.debug("Validating new configuration")
                validation_result = await self.validation_service.validate_configuration_update(
                    command.test_id, command.configuration
                )
                if not validation_result.is_valid:
                    logger.warning(f"Configuration validation failed: {validation_result.errors}")
                    return TestResultDTO(
                        test_id=command.test_id,
                        status="validation_failed",
                        created_test=False,
                        errors=validation_result.errors,
                    )

                # Step 4: Check for breaking changes
                logger.debug("Checking for breaking changes")
                breaking_changes = await self._analyze_breaking_changes(test, command.configuration)
                if breaking_changes["has_breaking_changes"]:
                    logger.warning(f"Breaking changes detected: {breaking_changes['changes']}")
                    # For now, we'll reject breaking changes - could add force_update option
                    return TestResultDTO(
                        test_id=command.test_id,
                        status="breaking_changes_detected",
                        created_test=False,
                        errors=[
                            f"Breaking changes detected: {', '.join(breaking_changes['changes'])}. "
                            "Consider creating a new test instead."
                        ],
                    )

                # Step 5: Create new configuration
                logger.debug("Creating new test configuration")
                old_configuration = test.configuration
                new_configuration = await self._create_updated_configuration(
                    command.configuration, old_configuration
                )

                # Step 6: Update test configuration
                test.configuration = new_configuration

                # Step 7: Reconfigure test if needed
                if test.status.allows_modification():
                    logger.debug("Reconfiguring test with new configuration")
                    # Clear samples if configuration changes require it
                    if breaking_changes.get("requires_sample_reset", False):
                        test.samples = []

                    # If test was configured, we need to validate it can still be configured
                    from ....domain.test_management.value_objects.test_status import TestStatus

                    if test.status == TestStatus.CONFIGURED:
                        # Reset to draft so we can reconfigure
                        test.status = TestStatus.DRAFT

                # Step 8: Persist changes
                await self.uow.tests.save(test)
                await self.uow.commit()

            # Step 9: Publish events
            logger.debug("Publishing configuration update events")
            domain_events = test.get_domain_events()
            await self.event_publisher.publish_all(domain_events)
            test.clear_domain_events()

            # Step 10: Log configuration changes
            await self._log_configuration_changes(test.id, old_configuration, new_configuration)

            logger.info(f"Configuration updated successfully for test: {command.test_id}")
            return TestResultDTO(
                test_id=command.test_id, status=test.status.value, created_test=False
            )

        except InvalidStateTransition as e:
            logger.warning(f"Invalid state transition during configuration update: {e}")
            return TestResultDTO(
                test_id=command.test_id,
                status="invalid_state_transition",
                created_test=False,
                errors=[str(e)],
            )
        except BusinessRuleViolation as e:
            logger.warning(f"Business rule violation during configuration update: {e}")
            return TestResultDTO(
                test_id=command.test_id,
                status="business_rule_violation",
                created_test=False,
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Unexpected error during configuration update: {e}", exc_info=True)
            return TestResultDTO(
                test_id=command.test_id,
                status="system_error",
                created_test=False,
                errors=[f"System error: {str(e)}"],
            )

    async def _validate_modification_permissions(self, test) -> list[str]:
        """Validate that test configuration can be modified."""
        errors = []

        # Check test state allows modification
        if not test.can_be_modified():
            errors.append(
                f"Test in {test.status.value} state cannot be modified. "
                "Only DRAFT and CONFIGURED tests can be modified."
            )

        # Check if test has active processes
        if test.status.is_active() and test.calculate_progress() > 0:
            errors.append(
                "Test has active evaluation processes. "
                "Wait for completion or cancel test before modifying configuration."
            )

        return errors

    async def _analyze_breaking_changes(self, test, new_configuration) -> dict:
        """Analyze if configuration changes are breaking changes."""
        old_config = test.configuration
        breaking_changes = []
        requires_sample_reset = False

        # Check for model changes
        old_models = set(old_config.models)
        new_models = set(model.model_id for model in new_configuration.models)

        if old_models != new_models:
            added_models = new_models - old_models
            removed_models = old_models - new_models

            if removed_models:
                breaking_changes.append(f"Removed models: {', '.join(removed_models)}")
                requires_sample_reset = True

            if added_models:
                # Adding models is less breaking but still significant
                breaking_changes.append(f"Added models: {', '.join(added_models)}")

        # Check for evaluation template changes
        old_template_id = getattr(old_config, "evaluation_template", {}).get("template_id")
        new_template_id = new_configuration.evaluation.template_id

        if old_template_id != new_template_id:
            breaking_changes.append(
                f"Changed evaluation template from {old_template_id} to {new_template_id}"
            )
            requires_sample_reset = True

        # Check for significant evaluation parameter changes
        old_eval = getattr(old_config, "evaluation_template", {})
        new_eval_dict = {
            "judge_count": new_configuration.evaluation.judge_count,
            "consensus_threshold": new_configuration.evaluation.consensus_threshold,
            "quality_threshold": new_configuration.evaluation.quality_threshold,
        }

        if old_eval.get("judge_count") != new_eval_dict["judge_count"]:
            if abs(old_eval.get("judge_count", 3) - new_eval_dict["judge_count"]) > 1:
                breaking_changes.append(
                    f"Significant judge count change: {old_eval.get('judge_count', 3)} → {new_eval_dict['judge_count']}"
                )

        # Check for max cost changes that are restrictive
        old_max_cost = getattr(old_config, "max_cost", None)
        new_max_cost = new_configuration.max_cost

        if old_max_cost and new_max_cost:
            if new_max_cost.amount < old_max_cost.amount * 0.8:  # 20% reduction
                breaking_changes.append(
                    f"Significant budget reduction: {old_max_cost} → {new_max_cost}"
                )

        return {
            "has_breaking_changes": len(breaking_changes) > 0,
            "changes": breaking_changes,
            "requires_sample_reset": requires_sample_reset,
        }

    async def _create_updated_configuration(
        self, new_config_dto, old_configuration
    ) -> TestConfiguration:
        """Create updated domain configuration from DTO."""
        # Convert models list to the format expected by TestConfiguration
        models = [model.model_id for model in new_config_dto.models]

        # Create evaluation template mapping
        evaluation_template = {
            "template_id": new_config_dto.evaluation.template_id,
            "judge_count": new_config_dto.evaluation.judge_count,
            "consensus_threshold": new_config_dto.evaluation.consensus_threshold,
            "quality_threshold": new_config_dto.evaluation.quality_threshold,
            "dimensions": new_config_dto.evaluation.dimensions,
        }

        # Create model parameters mapping
        model_parameters = {}
        for model_config in new_config_dto.models:
            key = f"{model_config.provider_name}/{model_config.model_id}"
            model_parameters[key] = {
                "parameters": model_config.parameters,
                "weight": model_config.weight,
            }

        # Preserve any metadata from old configuration
        old_metadata = getattr(old_configuration, "metadata", {})

        return TestConfiguration(
            models=models,
            evaluation_template=evaluation_template,
            max_cost=new_config_dto.max_cost,
            description=new_config_dto.description,
            model_parameters=model_parameters,
            metadata=old_metadata,
        )

    async def _log_configuration_changes(
        self, test_id: UUID, old_configuration, new_configuration
    ) -> None:
        """Log detailed configuration changes for audit trail."""
        logger.info(f"Configuration updated for test {test_id}")

        # Log model changes
        old_models = set(getattr(old_configuration, "models", []))
        new_models = set(getattr(new_configuration, "models", []))

        if old_models != new_models:
            added = new_models - old_models
            removed = old_models - new_models

            if added:
                logger.info(f"Test {test_id}: Added models: {', '.join(added)}")
            if removed:
                logger.info(f"Test {test_id}: Removed models: {', '.join(removed)}")

        # Log evaluation changes
        old_eval = getattr(old_configuration, "evaluation_template", {})
        new_eval = getattr(new_configuration, "evaluation_template", {})

        if old_eval != new_eval:
            logger.info(f"Test {test_id}: Evaluation configuration changed")

            for key in ["judge_count", "consensus_threshold", "quality_threshold"]:
                old_val = old_eval.get(key)
                new_val = new_eval.get(key)
                if old_val != new_val:
                    logger.debug(f"Test {test_id}: {key} changed from {old_val} to {new_val}")

        # Log cost changes
        old_cost = getattr(old_configuration, "max_cost", None)
        new_cost = getattr(new_configuration, "max_cost", None)

        if old_cost != new_cost:
            logger.info(f"Test {test_id}: Max cost changed from {old_cost} to {new_cost}")

        # Log description changes
        old_desc = getattr(old_configuration, "description", "")
        new_desc = getattr(new_configuration, "description", "")

        if old_desc != new_desc:
            logger.debug(f"Test {test_id}: Description updated")
