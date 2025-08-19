"""Service for validating test configurations and business rules."""

import math
from typing import Dict, List
from uuid import UUID

from ...domain.model_provider.value_objects.money import Money
from ...domain.test_management.value_objects.validation_result import ValidationResult
from ..dto.test_configuration_dto import CreateTestCommandDTO, TestConfigurationDTO
from ..interfaces.unit_of_work import UnitOfWork


class TestValidationService:
    """Service for validating test creation and configuration."""

    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def validate_test_creation(self, command: CreateTestCommandDTO) -> ValidationResult:
        """Validate complete test creation command."""
        errors = []

        # Basic validation
        basic_errors = self._validate_basic_requirements(command)
        errors.extend(basic_errors)

        # Configuration validation
        config_errors = await self._validate_configuration(command.configuration)
        errors.extend(config_errors)

        # Sample validation
        sample_errors = self._validate_samples(command.samples)
        errors.extend(sample_errors)

        # Statistical validation
        stats_errors = self._validate_statistical_requirements(command)
        errors.extend(stats_errors)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def validate_configuration_update(
        self, test_id: UUID, configuration: TestConfigurationDTO
    ) -> ValidationResult:
        """Validate configuration update for existing test."""
        errors = []

        async with self.uow:
            # Check if test exists and is modifiable
            test = await self.uow.tests.find_by_id(test_id)
            if not test:
                errors.append(f"Test {test_id} not found")
                return ValidationResult(is_valid=False, errors=errors)

            if not test.can_be_modified():
                errors.append(f"Test in {test.status.value} state cannot be modified")
                return ValidationResult(is_valid=False, errors=errors)

            # Validate new configuration
            config_errors = await self._validate_configuration(configuration)
            errors.extend(config_errors)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_basic_requirements(self, command: CreateTestCommandDTO) -> List[str]:
        """Validate basic test creation requirements."""
        errors = []

        # Name validation
        if not command.name.strip():
            errors.append("Test name cannot be empty")
        elif len(command.name) > 255:
            errors.append("Test name cannot exceed 255 characters")

        # Sample count validation
        if len(command.samples) < 10:
            errors.append("Minimum 10 samples required for statistical validity")
        elif len(command.samples) > 10000:
            errors.append("Maximum 10,000 samples allowed")

        return errors

    async def _validate_configuration(self, configuration: TestConfigurationDTO) -> List[str]:
        """Validate test configuration."""
        errors = []

        # Model validation
        if len(configuration.models) < 2:
            errors.append("At least two models required for A/B testing")
        elif len(configuration.models) > 10:
            errors.append("Maximum 10 models allowed per test")

        # Check for duplicate models
        model_keys = set()
        for model_config in configuration.models:
            key = (model_config.model_id, model_config.provider_name)
            if key in model_keys:
                errors.append(
                    f"Duplicate model: {model_config.model_id} from {model_config.provider_name}"
                )
            model_keys.add(key)

        # Validate model weights
        total_weight = sum(model.weight for model in configuration.models)
        if not (0.8 <= total_weight <= 1.2):  # Allow small variance due to floating point
            errors.append(f"Total model weights ({total_weight:.2f}) should sum to 1.0")

        # Evaluation configuration validation
        eval_config = configuration.evaluation
        if eval_config.judge_count < 1 or eval_config.judge_count > 10:
            errors.append("Judge count must be between 1 and 10")

        if not (0.5 <= eval_config.consensus_threshold <= 1.0):
            errors.append("Consensus threshold must be between 0.5 and 1.0")

        if not (0.0 < eval_config.quality_threshold <= 1.0):
            errors.append("Quality threshold must be between 0.0 and 1.0")

        # Validate providers and models exist
        async with self.uow:
            for model_config in configuration.models:
                provider = await self.uow.providers.find_by_name(model_config.provider_name)
                if not provider:
                    errors.append(f"Provider '{model_config.provider_name}' not found")
                    continue

                if not provider.find_model_config(model_config.model_id):
                    errors.append(
                        f"Model '{model_config.model_id}' not found in provider '{model_config.provider_name}'"
                    )

        return errors

    def _validate_samples(self, samples: List) -> List[str]:
        """Validate test samples."""
        errors = []

        for i, sample in enumerate(samples):
            if not sample.prompt.strip():
                errors.append(f"Sample {i+1}: Prompt cannot be empty")

            if len(sample.prompt) > 10000:
                errors.append(f"Sample {i+1}: Prompt cannot exceed 10,000 characters")

        # Check for duplicate prompts
        prompts = [sample.prompt for sample in samples]
        if len(prompts) != len(set(prompts)):
            errors.append("Duplicate prompts found - all prompts must be unique")

        return errors

    def _validate_statistical_requirements(self, command: CreateTestCommandDTO) -> List[str]:
        """Validate statistical requirements for valid A/B testing."""
        errors = []

        sample_count = len(command.samples)
        model_count = len(command.configuration.models)

        # Calculate minimum sample size for statistical power
        # Using simplified power analysis for two-proportion test
        # Assumptions: α = 0.05, β = 0.20 (80% power), effect size = 0.1
        min_samples_per_group = 200
        min_total_samples = min_samples_per_group * model_count

        if sample_count < min_total_samples:
            errors.append(
                f"Sample size ({sample_count}) may be insufficient for {model_count} models. "
                f"Consider at least {min_total_samples} samples for adequate statistical power."
            )

        # Calculate samples per model
        samples_per_model = sample_count // model_count
        if samples_per_model < 50:
            errors.append(
                f"Each model will receive only ~{samples_per_model} samples. "
                "Consider more samples for reliable comparisons."
            )

        return errors

    async def estimate_test_cost(self, command: CreateTestCommandDTO) -> Money:
        """Estimate total cost for running the test."""
        from .model_provider_service import ModelProviderService

        provider_service = ModelProviderService(self.uow)

        # Convert configuration to format expected by provider service
        model_configs = [
            {
                "model_id": model.model_id,
                "provider_name": model.provider_name,
                "parameters": model.parameters,
            }
            for model in command.configuration.models
        ]

        cost_estimates = await provider_service.get_model_cost_estimates(
            model_configs, len(command.samples)
        )

        total_cost = sum(cost_estimates.values())
        return Money(amount=total_cost, currency="USD")

    async def estimate_test_duration(self, command: CreateTestCommandDTO) -> float:
        """Estimate test duration in seconds."""
        sample_count = len(command.samples)
        model_count = len(command.configuration.models)
        judge_count = command.configuration.evaluation.judge_count

        # Estimation factors (in seconds)
        avg_model_inference_time = 2.0  # seconds per sample per model
        avg_evaluation_time = 1.5  # seconds per sample per judge
        coordination_overhead = 0.5  # seconds per sample for coordination

        # Calculate parallel processing efficiency
        # Assume some level of parallel processing for models and judges
        parallelism_factor = 0.7  # 70% efficiency due to coordination overhead

        model_time = (sample_count * model_count * avg_model_inference_time) * parallelism_factor
        evaluation_time = (sample_count * judge_count * avg_evaluation_time) * parallelism_factor
        coordination_time = sample_count * coordination_overhead

        total_duration = model_time + evaluation_time + coordination_time

        # Add buffer for startup and cleanup
        buffer_time = max(30.0, total_duration * 0.1)  # 10% buffer, minimum 30 seconds

        return total_duration + buffer_time
