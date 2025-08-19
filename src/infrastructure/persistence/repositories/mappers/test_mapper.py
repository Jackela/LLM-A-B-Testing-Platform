"""Domain-model mapper for Test Management domain."""

from typing import List, Optional
from uuid import UUID

from .....domain.test_management.entities.test import Test
from .....domain.test_management.entities.test_configuration import TestConfiguration
from .....domain.test_management.entities.test_sample import TestSample
from .....domain.test_management.value_objects.difficulty_level import DifficultyLevel
from .....domain.test_management.value_objects.test_status import TestStatus
from ...models.test_models import TestModel, TestSampleModel


class TestMapper:
    """Mapper between Test domain entities and database models."""

    def to_model(self, test: Test) -> TestModel:
        """Convert Test domain entity to database model."""
        test_model = TestModel(
            id=test.id,
            name=test.name,
            status=test.status,
            configuration=self._configuration_to_dict(test.configuration),
            created_at=test.created_at,
            completed_at=test.completed_at,
            failure_reason=test.failure_reason,
        )

        # Convert samples
        test_model.samples = [self._sample_to_model(sample, test.id) for sample in test.samples]

        return test_model

    def to_domain(self, test_model: TestModel) -> Test:
        """Convert database model to Test domain entity."""
        # Convert samples
        samples = [self._model_to_sample(sample_model) for sample_model in test_model.samples]

        # Create test configuration from JSON
        configuration = self._dict_to_configuration(test_model.configuration)

        # Create test entity
        test = Test(
            id=test_model.id,
            name=test_model.name,
            configuration=configuration,
            status=test_model.status,
            samples=samples,
            created_at=test_model.created_at,
            completed_at=test_model.completed_at,
            failure_reason=test_model.failure_reason,
        )

        # Clear domain events after loading from database
        test.clear_domain_events()

        return test

    def _sample_to_model(self, sample: TestSample, test_id: UUID) -> TestSampleModel:
        """Convert TestSample to TestSampleModel."""
        return TestSampleModel(
            id=sample.id,
            test_id=test_id,
            prompt=sample.prompt,
            difficulty=sample.difficulty,
            expected_output=sample.expected_output,
            tags=sample.tags.copy() if sample.tags else [],
            metadata=sample.metadata.copy() if sample.metadata else {},
            evaluation_results=(
                sample.evaluation_results.copy() if sample.evaluation_results else {}
            ),
            is_frozen=sample._is_frozen,
        )

    def _model_to_sample(self, sample_model: TestSampleModel) -> TestSample:
        """Convert TestSampleModel to TestSample."""
        sample = TestSample(
            prompt=sample_model.prompt,
            difficulty=sample_model.difficulty,
            expected_output=sample_model.expected_output,
            tags=sample_model.tags.copy() if sample_model.tags else [],
            metadata=sample_model.metadata.copy() if sample_model.metadata else {},
            id=sample_model.id,
        )

        # Restore evaluation results
        if sample_model.evaluation_results:
            for model_name, result in sample_model.evaluation_results.items():
                sample.add_evaluation_result(model_name, result)

        # Restore frozen state
        sample._is_frozen = sample_model.is_frozen

        return sample

    def _configuration_to_dict(self, configuration: TestConfiguration) -> dict:
        """Convert TestConfiguration to dictionary for JSON storage."""
        return {
            "models": configuration.models.copy(),
            "evaluation_templates": [str(t) for t in configuration.evaluation_templates],
            "randomization_seed": configuration.randomization_seed,
            "parallel_executions": configuration.parallel_executions.copy(),
            "timeout_seconds": configuration.timeout_seconds.copy(),
            "retry_config": configuration.retry_config.copy(),
        }

    def _dict_to_configuration(self, config_dict: dict) -> TestConfiguration:
        """Convert dictionary to TestConfiguration."""
        return TestConfiguration(
            models=config_dict.get("models", []),
            evaluation_templates=[UUID(t) for t in config_dict.get("evaluation_templates", [])],
            randomization_seed=config_dict.get("randomization_seed"),
            parallel_executions=config_dict.get("parallel_executions", {}),
            timeout_seconds=config_dict.get("timeout_seconds", {}),
            retry_config=config_dict.get("retry_config", {}),
        )
