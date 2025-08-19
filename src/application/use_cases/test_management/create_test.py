"""Use case for creating new tests."""

import logging
from typing import Any, Dict
from uuid import uuid4

from ....domain.test_management.entities.test import Test
from ....domain.test_management.entities.test_configuration import TestConfiguration
from ....domain.test_management.entities.test_sample import TestSample
from ....domain.test_management.exceptions import BusinessRuleViolation
from ...dto.test_configuration_dto import CreateTestCommandDTO, TestResultDTO
from ...interfaces.domain_event_publisher import DomainEventPublisher
from ...interfaces.unit_of_work import UnitOfWork
from ...services.model_provider_service import ModelProviderService
from ...services.test_validation_service import TestValidationService

logger = logging.getLogger(__name__)


class CreateTestUseCase:
    """Use case for creating new A/B tests with full validation."""

    def __init__(
        self,
        uow: UnitOfWork,
        event_publisher: DomainEventPublisher,
        validation_service: TestValidationService,
        provider_service: ModelProviderService,
    ):
        self.uow = uow
        self.event_publisher = event_publisher
        self.validation_service = validation_service
        self.provider_service = provider_service

    async def execute(self, command: CreateTestCommandDTO) -> TestResultDTO:
        """Execute test creation use case with comprehensive validation."""
        try:
            logger.info(f"Creating test: {command.name}")

            # Step 1: Validate business rules and requirements
            logger.debug("Validating test creation requirements")
            validation_result = await self.validation_service.validate_test_creation(command)
            if not validation_result.is_valid:
                logger.warning(f"Test creation validation failed: {validation_result.errors}")
                return TestResultDTO(
                    test_id=uuid4(),  # Placeholder ID for failed creation
                    status="validation_failed",
                    created_test=False,
                    errors=validation_result.errors,
                )

            # Step 2: Verify model providers are available and healthy
            logger.debug("Verifying model provider availability")
            await self._verify_model_providers(command)

            # Step 3: Create domain entities
            logger.debug("Creating domain entities")
            test_config = await self._create_test_configuration(command)
            test = Test.create(command.name, test_config)

            # Step 4: Add samples to test
            logger.debug(f"Adding {len(command.samples)} samples to test")
            for sample_dto in command.samples:
                sample = TestSample.create(
                    prompt=sample_dto.prompt,
                    expected_output=sample_dto.expected_output,
                    difficulty=sample_dto.difficulty,
                    metadata=sample_dto.metadata,
                )
                test.add_sample(sample)

            # Step 5: Configure the test (validates business rules)
            logger.debug("Configuring test")
            test.configure()

            # Step 6: Calculate cost and duration estimates
            logger.debug("Calculating estimates")
            estimated_cost = await self.validation_service.estimate_test_cost(command)
            estimated_duration = await self.validation_service.estimate_test_duration(command)

            # Step 7: Persist test with transaction management
            logger.debug("Persisting test")
            async with self.uow:
                await self.uow.tests.save(test)
                await self.uow.commit()

            # Step 8: Publish domain events
            logger.debug("Publishing domain events")
            domain_events = test.get_domain_events()
            await self.event_publisher.publish_all(domain_events)
            test.clear_domain_events()

            logger.info(f"Test created successfully: {test.id}")
            return TestResultDTO(
                test_id=test.id,
                status=test.status.value,
                created_test=True,
                estimated_cost=estimated_cost,
                estimated_duration=estimated_duration,
            )

        except BusinessRuleViolation as e:
            logger.warning(f"Business rule violation during test creation: {e}")
            return TestResultDTO(
                test_id=uuid4(),
                status="business_rule_violation",
                created_test=False,
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Unexpected error during test creation: {e}", exc_info=True)
            return TestResultDTO(
                test_id=uuid4(),
                status="system_error",
                created_test=False,
                errors=[f"System error: {str(e)}"],
            )

    async def _verify_model_providers(self, command: CreateTestCommandDTO) -> None:
        """Verify that all required model providers are available."""
        model_configs = [
            {"model_id": model.model_id, "provider_name": model.provider_name}
            for model in command.configuration.models
        ]

        availability = await self.provider_service.verify_model_availability(model_configs)
        unavailable_models = [
            model_key for model_key, available in availability.items() if not available
        ]

        if unavailable_models:
            raise BusinessRuleViolation(
                f"The following models are not available: {', '.join(unavailable_models)}"
            )

        # Validate model parameters
        model_configs_full = [
            {
                "model_id": model.model_id,
                "provider_name": model.provider_name,
                "parameters": model.parameters,
            }
            for model in command.configuration.models
        ]

        validation_errors = await self.provider_service.validate_model_parameters(
            model_configs_full
        )
        if validation_errors:
            error_messages = []
            for model_key, errors in validation_errors.items():
                error_messages.append(f"{model_key}: {', '.join(errors)}")
            raise BusinessRuleViolation(
                f"Model parameter validation failed: {'; '.join(error_messages)}"
            )

    async def _create_test_configuration(self, command: CreateTestCommandDTO) -> TestConfiguration:
        """Create domain test configuration from command."""
        # Convert models list to the format expected by TestConfiguration
        models = [model.model_id for model in command.configuration.models]

        # Create evaluation template mapping
        evaluation_template = {
            "template_id": command.configuration.evaluation.template_id,
            "judge_count": command.configuration.evaluation.judge_count,
            "consensus_threshold": command.configuration.evaluation.consensus_threshold,
            "quality_threshold": command.configuration.evaluation.quality_threshold,
            "dimensions": command.configuration.evaluation.dimensions,
        }

        # Create model parameters mapping
        model_parameters = {}
        for model_config in command.configuration.models:
            key = f"{model_config.provider_name}/{model_config.model_id}"
            model_parameters[key] = {
                "parameters": model_config.parameters,
                "weight": model_config.weight,
            }

        return TestConfiguration(
            models=models,
            evaluation_template=evaluation_template,
            max_cost=command.configuration.max_cost,
            description=command.configuration.description,
            model_parameters=model_parameters,
        )
