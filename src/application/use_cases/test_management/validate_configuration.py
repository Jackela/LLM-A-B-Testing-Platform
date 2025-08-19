"""Use case for validating test configuration."""

import logging
from typing import Dict, List
from uuid import UUID

from ....domain.model_provider.value_objects.money import Money
from ....domain.test_management.value_objects.validation_result import ValidationResult
from ...dto.test_configuration_dto import TestConfigurationDTO
from ...interfaces.unit_of_work import UnitOfWork
from ...services.model_provider_service import ModelProviderService
from ...services.test_validation_service import TestValidationService

logger = logging.getLogger(__name__)


class ValidateConfigurationUseCase:
    """Use case for comprehensive test configuration validation."""

    def __init__(
        self,
        uow: UnitOfWork,
        validation_service: TestValidationService,
        provider_service: ModelProviderService,
    ):
        self.uow = uow
        self.validation_service = validation_service
        self.provider_service = provider_service

    async def execute(self, configuration: TestConfigurationDTO) -> Dict:
        """Execute comprehensive configuration validation."""
        try:
            logger.debug("Validating test configuration")

            # Initialize validation results
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "recommendations": [],
                "cost_estimate": None,
                "model_availability": {},
                "statistical_power_analysis": {},
                "resource_requirements": {},
            }

            # Step 1: Basic configuration validation
            logger.debug("Performing basic configuration validation")
            basic_validation = await self._validate_basic_configuration(configuration)
            validation_result["errors"].extend(basic_validation["errors"])
            validation_result["warnings"].extend(basic_validation["warnings"])

            # Step 2: Model provider validation
            logger.debug("Validating model providers and availability")
            provider_validation = await self._validate_model_providers(configuration)
            validation_result["errors"].extend(provider_validation["errors"])
            validation_result["warnings"].extend(provider_validation["warnings"])
            validation_result["model_availability"] = provider_validation["availability"]

            # Step 3: Cost estimation and budget validation
            logger.debug("Estimating costs and validating budget")
            cost_validation = await self._validate_cost_requirements(configuration)
            validation_result["errors"].extend(cost_validation["errors"])
            validation_result["warnings"].extend(cost_validation["warnings"])
            validation_result["cost_estimate"] = cost_validation["cost_estimate"]

            # Step 4: Statistical power analysis
            logger.debug("Performing statistical power analysis")
            power_analysis = await self._analyze_statistical_power(configuration)
            validation_result["warnings"].extend(power_analysis["warnings"])
            validation_result["recommendations"].extend(power_analysis["recommendations"])
            validation_result["statistical_power_analysis"] = power_analysis["analysis"]

            # Step 5: Resource capacity validation
            logger.debug("Validating resource capacity")
            resource_validation = await self._validate_resource_capacity(configuration)
            validation_result["errors"].extend(resource_validation["errors"])
            validation_result["warnings"].extend(resource_validation["warnings"])
            validation_result["resource_requirements"] = resource_validation["requirements"]

            # Step 6: Evaluation template compatibility
            logger.debug("Validating evaluation template compatibility")
            template_validation = await self._validate_evaluation_template(configuration)
            validation_result["errors"].extend(template_validation["errors"])
            validation_result["warnings"].extend(template_validation["warnings"])

            # Set overall validity
            validation_result["is_valid"] = len(validation_result["errors"]) == 0

            logger.debug(
                f"Configuration validation complete. Valid: {validation_result['is_valid']}, "
                f"Errors: {len(validation_result['errors'])}, "
                f"Warnings: {len(validation_result['warnings'])}"
            )

            return validation_result

        except Exception as e:
            logger.error(f"Error during configuration validation: {e}", exc_info=True)
            return {
                "is_valid": False,
                "errors": [f"Validation system error: {str(e)}"],
                "warnings": [],
                "recommendations": [],
                "cost_estimate": None,
                "model_availability": {},
                "statistical_power_analysis": {},
                "resource_requirements": {},
            }

    async def _validate_basic_configuration(self, configuration: TestConfigurationDTO) -> Dict:
        """Validate basic configuration structure and business rules."""
        errors = []
        warnings = []

        # Model configuration validation
        if not configuration.models:
            errors.append("At least one model configuration is required")
        elif len(configuration.models) < 2:
            errors.append("At least two models are required for A/B testing")
        elif len(configuration.models) > 10:
            warnings.append("Large number of models may impact test performance and cost")

        # Validate model weights
        total_weight = sum(model.weight for model in configuration.models)
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating-point variance
            errors.append(f"Model weights sum to {total_weight:.3f}, should sum to 1.0")

        # Check for invalid weights
        for i, model in enumerate(configuration.models):
            if model.weight <= 0:
                errors.append(f"Model {i+1} has invalid weight: {model.weight}")
            elif model.weight > 0.8:
                warnings.append(
                    f"Model {i+1} has high weight ({model.weight:.2f}) - may skew results"
                )

        # Evaluation configuration validation
        eval_config = configuration.evaluation
        if eval_config.judge_count < 1:
            errors.append("At least one judge is required for evaluation")
        elif eval_config.judge_count > 10:
            warnings.append("High judge count will increase evaluation time and cost")

        if not (0.5 <= eval_config.consensus_threshold <= 1.0):
            errors.append("Consensus threshold must be between 0.5 and 1.0")
        elif eval_config.consensus_threshold < 0.7:
            warnings.append("Low consensus threshold may produce unreliable results")

        if not (0.0 < eval_config.quality_threshold <= 1.0):
            errors.append("Quality threshold must be between 0.0 and 1.0")
        elif eval_config.quality_threshold < 0.6:
            warnings.append("Low quality threshold may include poor evaluations")

        # Description validation
        if configuration.description and len(configuration.description) > 1000:
            warnings.append("Test description is very long and may be truncated in reports")

        return {"errors": errors, "warnings": warnings}

    async def _validate_model_providers(self, configuration: TestConfigurationDTO) -> Dict:
        """Validate model providers and their availability."""
        errors = []
        warnings = []
        availability = {}

        async with self.uow:
            for model_config in configuration.models:
                model_key = f"{model_config.provider_name}/{model_config.model_id}"

                # Check if provider exists
                provider = await self.uow.providers.find_by_name(model_config.provider_name)
                if not provider:
                    errors.append(f"Provider '{model_config.provider_name}' not found")
                    availability[model_key] = False
                    continue

                # Check if model exists in provider
                model_cfg = provider.find_model_config(model_config.model_id)
                if not model_cfg:
                    errors.append(
                        f"Model '{model_config.model_id}' not found in provider '{model_config.provider_name}'"
                    )
                    availability[model_key] = False
                    continue

                # Check provider health
                if not provider.health_status.is_operational:
                    warnings.append(
                        f"Provider '{model_config.provider_name}' is not operational: {provider.health_status.name}"
                    )

                # Check rate limits
                if not provider.rate_limits.can_make_request():
                    warnings.append(
                        f"Provider '{model_config.provider_name}' has reached rate limits"
                    )
                    availability[model_key] = False
                else:
                    availability[model_key] = True

                # Validate model parameters
                for param_name, param_value in model_config.parameters.items():
                    if not model_cfg.supports_parameter(param_name):
                        errors.append(
                            f"Parameter '{param_name}' not supported by model '{model_config.model_id}'"
                        )

                # Validate parameter values
                if "max_tokens" in model_config.parameters:
                    max_tokens = model_config.parameters["max_tokens"]
                    if max_tokens > model_cfg.max_tokens:
                        errors.append(
                            f"max_tokens ({max_tokens}) exceeds model limit ({model_cfg.max_tokens}) "
                            f"for '{model_config.model_id}'"
                        )

                # Validate temperature and other common parameters
                if "temperature" in model_config.parameters:
                    temp = model_config.parameters["temperature"]
                    if not (0.0 <= temp <= 2.0):
                        warnings.append(
                            f"Temperature ({temp}) for '{model_config.model_id}' is outside typical range (0.0-2.0)"
                        )

        return {"errors": errors, "warnings": warnings, "availability": availability}

    async def _validate_cost_requirements(self, configuration: TestConfigurationDTO) -> Dict:
        """Validate cost requirements and budget constraints."""
        errors = []
        warnings = []
        cost_estimate = None

        try:
            # Calculate cost estimate for typical test size
            typical_sample_count = 1000  # Use typical sample count for estimation

            model_configs = [
                {
                    "model_id": model.model_id,
                    "provider_name": model.provider_name,
                    "parameters": model.parameters,
                }
                for model in configuration.models
            ]

            cost_estimates = await self.provider_service.get_model_cost_estimates(
                model_configs, typical_sample_count
            )

            total_estimated_cost = sum(cost_estimates.values())
            cost_estimate = Money(total_estimated_cost, "USD")

            # Validate against max cost if specified
            if configuration.max_cost:
                if total_estimated_cost > configuration.max_cost.amount:
                    errors.append(
                        f"Estimated cost ({cost_estimate}) exceeds maximum budget ({configuration.max_cost})"
                    )
                elif total_estimated_cost > configuration.max_cost.amount * 0.8:
                    warnings.append(
                        f"Estimated cost ({cost_estimate}) is close to budget limit ({configuration.max_cost})"
                    )
            else:
                # Warn about high costs
                if total_estimated_cost > 100:  # $100 threshold
                    warnings.append(
                        f"Estimated cost is high ({cost_estimate}). Consider setting a max_cost budget."
                    )

            # Analyze cost distribution across models
            if cost_estimates:
                max_cost_model = max(cost_estimates, key=cost_estimates.get)
                min_cost_model = min(cost_estimates, key=cost_estimates.get)
                cost_ratio = cost_estimates[max_cost_model] / cost_estimates[min_cost_model]

                if cost_ratio > 10:  # 10x difference
                    warnings.append(
                        f"Large cost difference between models. "
                        f"Most expensive: {max_cost_model} (${cost_estimates[max_cost_model]:.3f}), "
                        f"Cheapest: {min_cost_model} (${cost_estimates[min_cost_model]:.3f})"
                    )

        except Exception as e:
            warnings.append(f"Unable to calculate cost estimate: {str(e)}")

        return {"errors": errors, "warnings": warnings, "cost_estimate": cost_estimate}

    async def _analyze_statistical_power(self, configuration: TestConfigurationDTO) -> Dict:
        """Analyze statistical power for the test configuration."""
        warnings = []
        recommendations = []
        analysis = {}

        # Basic power analysis assuming typical parameters
        model_count = len(configuration.models)
        effect_size = 0.1  # Assume 10% effect size we want to detect
        alpha = 0.05
        power = 0.8

        # Simplified power calculation for two-proportion test
        import math

        z_alpha = 1.96  # For alpha = 0.05
        z_beta = 0.84  # For power = 0.8

        # Estimate required sample size per group
        p1 = 0.5  # Assume base success rate of 50%
        p2 = p1 + effect_size
        p_pooled = (p1 + p2) / 2

        sample_per_group = ((z_alpha + z_beta) ** 2 * p_pooled * (1 - p_pooled)) / (effect_size**2)
        min_total_samples = int(sample_per_group * model_count)

        analysis = {
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "estimated_min_samples_per_group": int(sample_per_group),
            "estimated_min_total_samples": min_total_samples,
            "model_count": model_count,
        }

        # Generate recommendations based on analysis
        if min_total_samples > 5000:
            warnings.append(
                f"Large sample size recommended ({min_total_samples}) for detecting {effect_size:.1%} "
                f"effect with {model_count} models"
            )
            recommendations.append(
                "Consider reducing number of models or accepting larger effect size for detection"
            )
        elif min_total_samples < 200:
            recommendations.append(
                f"Minimum {min_total_samples} samples recommended for adequate statistical power"
            )
        else:
            recommendations.append(
                f"Recommended minimum sample size: {min_total_samples} samples total"
            )

        # Analyze judge configuration impact
        judge_count = configuration.evaluation.judge_count
        if judge_count == 1:
            warnings.append("Single judge evaluation may introduce bias")
            recommendations.append("Consider using multiple judges for more reliable evaluation")
        elif judge_count > 5:
            warnings.append("High judge count will significantly increase evaluation time and cost")

        return {"warnings": warnings, "recommendations": recommendations, "analysis": analysis}

    async def _validate_resource_capacity(self, configuration: TestConfigurationDTO) -> Dict:
        """Validate system resource capacity for the test configuration."""
        errors = []
        warnings = []
        requirements = {}

        # Calculate resource requirements
        model_count = len(configuration.models)
        judge_count = configuration.evaluation.judge_count

        # Estimate concurrent request capacity needed
        typical_sample_count = 1000
        concurrent_requests = min(model_count * 2, 10)  # Reasonable concurrency limit

        # Estimate memory requirements (simplified)
        memory_per_model = 100  # MB per model
        total_memory = model_count * memory_per_model

        # Estimate processing time
        avg_inference_time = 2.0  # seconds per sample per model
        avg_evaluation_time = 1.5  # seconds per sample per judge

        total_processing_time = (
            (typical_sample_count * model_count * avg_inference_time)
            + (typical_sample_count * judge_count * avg_evaluation_time)
        ) / concurrent_requests  # Account for parallelization

        requirements = {
            "estimated_concurrent_requests": concurrent_requests,
            "estimated_memory_mb": total_memory,
            "estimated_processing_time_seconds": total_processing_time,
            "model_count": model_count,
            "judge_count": judge_count,
        }

        # Validate against system limits (simplified - would check actual system capacity)
        if concurrent_requests > 20:
            warnings.append(
                f"High concurrent request load ({concurrent_requests}) may impact performance"
            )

        if total_memory > 2000:  # 2GB
            warnings.append(f"High memory requirement ({total_memory} MB) may impact performance")

        if total_processing_time > 3600:  # 1 hour
            warnings.append(
                f"Long processing time estimated ({total_processing_time/60:.1f} minutes)"
            )

        return {"errors": errors, "warnings": warnings, "requirements": requirements}

    async def _validate_evaluation_template(self, configuration: TestConfigurationDTO) -> Dict:
        """Validate evaluation template exists and is compatible."""
        errors = []
        warnings = []

        async with self.uow:
            # Check if evaluation template exists
            template_id = configuration.evaluation.template_id
            template = await self.uow.evaluations.find_template_by_id(template_id)

            if not template:
                errors.append(f"Evaluation template '{template_id}' not found")
                return {"errors": errors, "warnings": warnings}

            # Validate template compatibility with models
            # This would check if the template is suitable for the types of models being tested
            model_categories = set()
            async with self.uow:
                for model_config in configuration.models:
                    provider = await self.uow.providers.find_by_name(model_config.provider_name)
                    if provider:
                        model_cfg = provider.find_model_config(model_config.model_id)
                        if model_cfg:
                            model_categories.add(model_cfg.model_category)

            # Check if template supports all model categories
            # This is simplified - in practice would check template specifications
            if len(model_categories) > 1:
                warnings.append("Mixed model categories may require different evaluation criteria")

            # Validate dimension requirements
            if configuration.evaluation.dimensions:
                # Check if template supports requested dimensions
                # This would validate against actual template definition
                unsupported_dimensions = []
                for dimension in configuration.evaluation.dimensions:
                    # Simplified check - would validate against actual template
                    if len(dimension.strip()) == 0:
                        unsupported_dimensions.append("empty dimension")

                if unsupported_dimensions:
                    errors.append(
                        f"Template does not support dimensions: {', '.join(unsupported_dimensions)}"
                    )

        return {"errors": errors, "warnings": warnings}
