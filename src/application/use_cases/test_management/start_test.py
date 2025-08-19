"""Use case for starting test execution."""

import logging
from uuid import UUID, uuid4

from ....domain.test_management.exceptions import BusinessRuleViolation, InvalidStateTransition
from ...dto.test_configuration_dto import StartTestCommandDTO, TestResultDTO
from ...interfaces.domain_event_publisher import DomainEventPublisher
from ...interfaces.unit_of_work import UnitOfWork
from ...services.model_provider_service import ModelProviderService

logger = logging.getLogger(__name__)


class StartTestUseCase:
    """Use case for starting test execution with pre-flight validation."""

    def __init__(
        self,
        uow: UnitOfWork,
        event_publisher: DomainEventPublisher,
        provider_service: ModelProviderService,
    ):
        self.uow = uow
        self.event_publisher = event_publisher
        self.provider_service = provider_service

    async def execute(self, command: StartTestCommandDTO) -> TestResultDTO:
        """Execute test start workflow with comprehensive validation."""
        try:
            logger.info(f"Starting test: {command.test_id}")

            # Step 1: Load test aggregate
            async with self.uow:
                test = await self.uow.tests.find_by_id(command.test_id)
                if not test:
                    logger.warning(f"Test not found: {command.test_id}")
                    return TestResultDTO(
                        test_id=command.test_id,
                        status="not_found",
                        created_test=False,
                        errors=[f"Test {command.test_id} not found"],
                    )

                # Step 2: Validate test can be started
                logger.debug("Validating test can be started")
                validation_errors = await self._validate_test_start_conditions(test)
                if validation_errors:
                    logger.warning(f"Test start validation failed: {validation_errors}")
                    return TestResultDTO(
                        test_id=command.test_id,
                        status="validation_failed",
                        created_test=False,
                        errors=validation_errors,
                    )

                # Step 3: Pre-flight checks for providers and models
                logger.debug("Performing pre-flight checks")
                await self._perform_preflight_checks(test)

                # Step 4: Start test execution
                logger.debug("Starting test execution")
                test.start()

                # Step 5: Persist state change
                await self.uow.tests.save(test)
                await self.uow.commit()

            # Step 6: Publish domain events
            logger.debug("Publishing domain events")
            domain_events = test.get_domain_events()
            await self.event_publisher.publish_all(domain_events)
            test.clear_domain_events()

            # Step 7: Schedule sample processing (would be handled by infrastructure)
            logger.debug("Test execution workflow initiated")
            await self._schedule_sample_processing(test)

            logger.info(f"Test started successfully: {command.test_id}")
            return TestResultDTO(
                test_id=command.test_id,
                status=test.status.value,
                created_test=False,  # Test already existed
            )

        except InvalidStateTransition as e:
            logger.warning(f"Invalid state transition during test start: {e}")
            return TestResultDTO(
                test_id=command.test_id,
                status="invalid_state_transition",
                created_test=False,
                errors=[str(e)],
            )
        except BusinessRuleViolation as e:
            logger.warning(f"Business rule violation during test start: {e}")
            return TestResultDTO(
                test_id=command.test_id,
                status="business_rule_violation",
                created_test=False,
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Unexpected error during test start: {e}", exc_info=True)
            return TestResultDTO(
                test_id=command.test_id,
                status="system_error",
                created_test=False,
                errors=[f"System error: {str(e)}"],
            )

    async def _validate_test_start_conditions(self, test) -> list[str]:
        """Validate that test meets all conditions to be started."""
        errors = []

        # Check test has samples
        if not test.samples:
            errors.append("Test has no samples - cannot start")

        # Check test is in correct state
        from ....domain.test_management.value_objects.test_status import TestStatus

        if test.status != TestStatus.CONFIGURED:
            errors.append(
                f"Test must be in CONFIGURED state to start, currently {test.status.value}"
            )

        # Check minimum sample count for statistical validity
        if len(test.samples) < 10:
            errors.append(f"Test has only {len(test.samples)} samples - minimum 10 required")

        # Validate configuration is complete
        config_validation = test.configuration.validate()
        if not config_validation.is_valid:
            errors.extend([f"Configuration error: {err}" for err in config_validation.errors])

        return errors

    async def _perform_preflight_checks(self, test) -> None:
        """Perform comprehensive pre-flight checks before starting test."""
        # Check all providers are healthy
        providers = await self.provider_service.get_providers_for_test(test.configuration)

        unhealthy_providers = []
        for provider in providers:
            if not provider.health_status.is_operational:
                unhealthy_providers.append(f"{provider.name} ({provider.health_status.name})")

        if unhealthy_providers:
            raise BusinessRuleViolation(
                f"Cannot start test - providers not operational: {', '.join(unhealthy_providers)}"
            )

        # Verify specific models are available
        model_configs = []
        for model in test.configuration.models:
            # Extract provider name from model configuration
            # This assumes models are stored with provider info in configuration
            model_configs.append(
                {
                    "model_id": model,
                    "provider_name": self._extract_provider_name(model, test.configuration),
                }
            )

        availability = await self.provider_service.verify_model_availability(model_configs)
        unavailable_models = [
            model_key for model_key, available in availability.items() if not available
        ]

        if unavailable_models:
            raise BusinessRuleViolation(
                f"Cannot start test - models not available: {', '.join(unavailable_models)}"
            )

        # Check rate limits won't be immediately exceeded
        for provider in providers:
            if not provider.rate_limits.can_make_request():
                raise BusinessRuleViolation(
                    f"Cannot start test - provider {provider.name} rate limit reached"
                )

    def _extract_provider_name(self, model: str, configuration) -> str:
        """Extract provider name for a model from configuration."""
        # This is a simplified approach - in practice, the configuration
        # would contain more structured information about model-provider mapping
        model_params = getattr(configuration, "model_parameters", {})
        for key in model_params:
            if model in key:
                return key.split("/")[0]
        return "default"  # Fallback

    async def _schedule_sample_processing(self, test) -> None:
        """Schedule sample processing for the test (infrastructure concern)."""
        # This would typically interface with a task queue or scheduling system
        # For now, we log the intention
        logger.info(
            f"Scheduling processing for {len(test.samples)} samples "
            f"across {len(test.configuration.models)} models for test {test.id}"
        )

        # Calculate processing priority based on test characteristics
        priority = "normal"
        if len(test.samples) > 1000:
            priority = "high"
        elif len(test.samples) < 100:
            priority = "low"

        logger.debug(f"Test {test.id} scheduled with priority: {priority}")

        # In a real implementation, this would create processing jobs:
        # await self.task_scheduler.schedule_test_processing(
        #     test_id=test.id,
        #     priority=priority,
        #     estimated_duration=test.estimate_remaining_time()
        # )
