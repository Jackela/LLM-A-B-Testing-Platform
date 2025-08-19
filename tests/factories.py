"""Test data factories for generating test objects."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List
from uuid import uuid4

import factory

from src.application.dto.test_configuration_dto import (
    CreateTestCommandDTO,
    EvaluationConfigurationDTO,
    ModelConfigurationDTO,
    TestConfigurationDTO,
    TestSampleDTO,
)
from src.domain.analytics.entities.analysis_result import AnalysisResult
from src.domain.analytics.entities.model_performance import ModelPerformance
from src.domain.analytics.entities.statistical_test import StatisticalTest
from src.domain.analytics.value_objects.confidence_interval import ConfidenceInterval
from src.domain.analytics.value_objects.insight import Insight
from src.domain.analytics.value_objects.performance_score import PerformanceScore
from src.domain.analytics.value_objects.test_result import TestResult as AnalyticsTestResult
from src.domain.evaluation.entities.dimension import Dimension
from src.domain.evaluation.entities.evaluation_result import EvaluationResult
from src.domain.evaluation.entities.evaluation_template import EvaluationTemplate
from src.domain.evaluation.entities.judge import Judge
from src.domain.evaluation.value_objects.consensus_result import ConsensusResult
from src.domain.evaluation.value_objects.quality_report import QualityReport
from src.domain.evaluation.value_objects.scoring_scale import ScoringScale
from src.domain.model_provider.entities.model_config import ModelConfig
from src.domain.model_provider.entities.model_provider import ModelProvider
from src.domain.model_provider.entities.model_response import ModelResponse
from src.domain.model_provider.value_objects.health_status import HealthStatus
from src.domain.model_provider.value_objects.model_category import ModelCategory
from src.domain.model_provider.value_objects.money import Money
from src.domain.model_provider.value_objects.provider_type import ProviderType
from src.domain.model_provider.value_objects.rate_limits import RateLimits
from src.domain.test_management.entities.test import Test
from src.domain.test_management.entities.test_configuration import TestConfiguration
from src.domain.test_management.entities.test_sample import TestSample
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.test_status import TestStatus


# Base Factory with common utilities
class BaseFactory(factory.Factory):
    """Base factory with common utilities."""

    @classmethod
    def _make_uuid(cls) -> str:
        """Generate a UUID string."""
        return str(uuid4())

    @classmethod
    def _make_timestamp(cls, delta_hours: int = 0) -> datetime:
        """Generate a timestamp with optional delta."""
        return datetime.utcnow() + timedelta(hours=delta_hours)


# Value Object Factories
class MoneyFactory(BaseFactory):
    """Factory for Money value objects."""

    class Meta:
        model = Money

    amount = factory.Faker("pydecimal", left_digits=3, right_digits=2, positive=True)
    currency = "USD"


class RateLimitsFactory(BaseFactory):
    """Factory for RateLimits value objects."""

    class Meta:
        model = RateLimits

    requests_per_minute = factory.Faker("random_int", min=10, max=1000)
    requests_per_hour = factory.LazyAttribute(lambda obj: obj.requests_per_minute * 60)
    requests_per_day = factory.LazyAttribute(lambda obj: obj.requests_per_hour * 24)
    tokens_per_minute = factory.Faker("random_int", min=1000, max=100000)


class HealthStatusFactory(BaseFactory):
    """Factory for HealthStatus value objects."""

    class Meta:
        model = HealthStatus

    is_healthy = True
    last_check = factory.LazyFunction(datetime.utcnow)
    response_time_ms = factory.Faker("random_int", min=50, max=500)
    error_rate = factory.Faker("pyfloat", min_value=0.0, max_value=0.05)


class ScoringScaleFactory(BaseFactory):
    """Factory for ScoringScale value objects."""

    class Meta:
        model = ScoringScale

    min_score = 1
    max_score = 10
    scale_type = "linear"
    description = factory.Faker("sentence")


class PerformanceScoreFactory(BaseFactory):
    """Factory for PerformanceScore value objects."""

    class Meta:
        model = PerformanceScore

    accuracy = factory.Faker("pyfloat", min_value=0.7, max_value=1.0)
    relevance = factory.Faker("pyfloat", min_value=0.7, max_value=1.0)
    coherence = factory.Faker("pyfloat", min_value=0.7, max_value=1.0)
    creativity = factory.Faker("pyfloat", min_value=0.5, max_value=1.0)
    overall_score = factory.LazyAttribute(
        lambda obj: (obj.accuracy + obj.relevance + obj.coherence + obj.creativity) / 4
    )


class ConfidenceIntervalFactory(BaseFactory):
    """Factory for ConfidenceInterval value objects."""

    class Meta:
        model = ConfidenceInterval

    lower_bound = factory.Faker("pyfloat", min_value=0.0, max_value=0.5)
    upper_bound = factory.LazyAttribute(lambda obj: obj.lower_bound + 0.5)
    confidence_level = 0.95


# Entity Factories
class ModelProviderFactory(BaseFactory):
    """Factory for ModelProvider entities."""

    class Meta:
        model = ModelProvider

    id = factory.LazyFunction(BaseFactory._make_uuid)
    name = factory.Faker("company")
    provider_type = factory.Faker("random_element", elements=[e.value for e in ProviderType])
    api_endpoint = factory.Faker("url")
    api_key_hash = factory.Faker("sha256")
    is_active = True
    rate_limits = factory.SubFactory(RateLimitsFactory)
    health_status = factory.SubFactory(HealthStatusFactory)
    created_at = factory.LazyFunction(datetime.utcnow)
    updated_at = factory.LazyFunction(datetime.utcnow)


class ModelConfigFactory(BaseFactory):
    """Factory for ModelConfig entities."""

    class Meta:
        model = ModelConfig

    id = factory.LazyFunction(BaseFactory._make_uuid)
    model_id = factory.Faker("slug")
    provider_id = factory.LazyFunction(BaseFactory._make_uuid)
    name = factory.Faker("name")
    category = factory.Faker("random_element", elements=[e.value for e in ModelCategory])
    parameters = factory.LazyFunction(
        lambda: {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9}
    )
    cost_per_token = factory.SubFactory(MoneyFactory)
    is_available = True
    created_at = factory.LazyFunction(datetime.utcnow)


class ModelResponseFactory(BaseFactory):
    """Factory for ModelResponse entities."""

    class Meta:
        model = ModelResponse

    id = factory.LazyFunction(BaseFactory._make_uuid)
    model_config_id = factory.LazyFunction(BaseFactory._make_uuid)
    prompt = factory.Faker("text", max_nb_chars=200)
    response_text = factory.Faker("text", max_nb_chars=500)
    tokens_used = factory.Faker("random_int", min=10, max=1000)
    cost = factory.SubFactory(MoneyFactory)
    response_time_ms = factory.Faker("random_int", min=100, max=5000)
    metadata = factory.LazyFunction(lambda: {"model_version": "1.0", "finish_reason": "stop"})
    created_at = factory.LazyFunction(datetime.utcnow)


class TestSampleFactory(BaseFactory):
    """Factory for TestSample entities."""

    class Meta:
        model = TestSample

    id = factory.LazyFunction(BaseFactory._make_uuid)
    test_id = factory.LazyFunction(BaseFactory._make_uuid)
    prompt = factory.Faker("text", max_nb_chars=200)
    expected_response = factory.Faker("text", max_nb_chars=300)
    difficulty = factory.Faker("random_element", elements=[e.value for e in DifficultyLevel])
    metadata = factory.LazyFunction(lambda: {"source": "synthetic"})
    created_at = factory.LazyFunction(datetime.utcnow)


class TestConfigurationFactory(BaseFactory):
    """Factory for TestConfiguration entities."""

    class Meta:
        model = TestConfiguration

    id = factory.LazyFunction(BaseFactory._make_uuid)
    name = factory.Faker("sentence", nb_words=4)
    description = factory.Faker("text", max_nb_chars=200)
    model_configs = factory.LazyFunction(lambda: [])
    evaluation_template_id = factory.LazyFunction(BaseFactory._make_uuid)
    max_cost = factory.SubFactory(MoneyFactory)
    timeout_seconds = factory.Faker("random_int", min=30, max=600)
    created_at = factory.LazyFunction(datetime.utcnow)
    updated_at = factory.LazyFunction(datetime.utcnow)


class TestFactory(BaseFactory):
    """Factory for Test entities."""

    class Meta:
        model = Test

    id = factory.LazyFunction(BaseFactory._make_uuid)
    name = factory.Faker("sentence", nb_words=4)
    configuration = factory.SubFactory(TestConfigurationFactory)
    status = TestStatus.CONFIGURED
    samples = factory.LazyFunction(lambda: [])
    created_at = factory.LazyFunction(datetime.utcnow)
    updated_at = factory.LazyFunction(datetime.utcnow)
    started_at = None
    completed_at = None


class DimensionFactory(BaseFactory):
    """Factory for Dimension entities."""

    class Meta:
        model = Dimension

    id = factory.LazyFunction(BaseFactory._make_uuid)
    name = factory.Faker("word")
    description = factory.Faker("sentence")
    weight = factory.Faker("pyfloat", min_value=0.1, max_value=1.0)
    scoring_scale = factory.SubFactory(ScoringScaleFactory)
    created_at = factory.LazyFunction(datetime.utcnow)


class EvaluationTemplateFactory(BaseFactory):
    """Factory for EvaluationTemplate entities."""

    class Meta:
        model = EvaluationTemplate

    id = factory.LazyFunction(BaseFactory._make_uuid)
    name = factory.Faker("sentence", nb_words=3)
    description = factory.Faker("text", max_nb_chars=200)
    dimensions = factory.LazyFunction(lambda: [])
    instructions = factory.Faker("text", max_nb_chars=500)
    example_evaluations = factory.LazyFunction(lambda: [])
    created_at = factory.LazyFunction(datetime.utcnow)


class JudgeFactory(BaseFactory):
    """Factory for Judge entities."""

    class Meta:
        model = Judge

    id = factory.LazyFunction(BaseFactory._make_uuid)
    name = factory.Faker("name")
    model_config_id = factory.LazyFunction(BaseFactory._make_uuid)
    expertise_domains = factory.LazyFunction(lambda: ["general", "technical"])
    calibration_score = factory.Faker("pyfloat", min_value=0.8, max_value=1.0)
    is_active = True
    created_at = factory.LazyFunction(datetime.utcnow)


class EvaluationResultFactory(BaseFactory):
    """Factory for EvaluationResult entities."""

    class Meta:
        model = EvaluationResult

    id = factory.LazyFunction(BaseFactory._make_uuid)
    sample_id = factory.LazyFunction(BaseFactory._make_uuid)
    judge_id = factory.LazyFunction(BaseFactory._make_uuid)
    model_response_id = factory.LazyFunction(BaseFactory._make_uuid)
    scores = factory.LazyFunction(lambda: {"accuracy": 8.5, "relevance": 9.0})
    reasoning = factory.Faker("text", max_nb_chars=300)
    confidence = factory.Faker("pyfloat", min_value=0.7, max_value=1.0)
    created_at = factory.LazyFunction(datetime.utcnow)


class ModelPerformanceFactory(BaseFactory):
    """Factory for ModelPerformance entities."""

    class Meta:
        model = ModelPerformance

    id = factory.LazyFunction(BaseFactory._make_uuid)
    test_id = factory.LazyFunction(BaseFactory._make_uuid)
    model_config_id = factory.LazyFunction(BaseFactory._make_uuid)
    performance_score = factory.SubFactory(PerformanceScoreFactory)
    total_samples = factory.Faker("random_int", min=10, max=1000)
    successful_samples = factory.LazyAttribute(lambda obj: int(obj.total_samples * 0.95))
    total_cost = factory.SubFactory(MoneyFactory)
    average_response_time = factory.Faker("pyfloat", min_value=100, max_value=2000)
    created_at = factory.LazyFunction(datetime.utcnow)


class StatisticalTestFactory(BaseFactory):
    """Factory for StatisticalTest entities."""

    class Meta:
        model = StatisticalTest

    id = factory.LazyFunction(BaseFactory._make_uuid)
    analysis_id = factory.LazyFunction(BaseFactory._make_uuid)
    test_type = factory.Faker("random_element", elements=["t_test", "mann_whitney", "chi_square"])
    statistic_value = factory.Faker("pyfloat", min_value=0.1, max_value=10.0)
    p_value = factory.Faker("pyfloat", min_value=0.001, max_value=0.999)
    degrees_of_freedom = factory.Faker("random_int", min=1, max=100)
    confidence_interval = factory.SubFactory(ConfidenceIntervalFactory)
    is_significant = factory.LazyAttribute(lambda obj: obj.p_value < 0.05)
    created_at = factory.LazyFunction(datetime.utcnow)


class AnalysisResultFactory(BaseFactory):
    """Factory for AnalysisResult entities."""

    class Meta:
        model = AnalysisResult

    id = factory.LazyFunction(BaseFactory._make_uuid)
    test_id = factory.LazyFunction(BaseFactory._make_uuid)
    analysis_type = factory.Faker("random_element", elements=["comparison", "trend", "anomaly"])
    results = factory.LazyFunction(lambda: {"winner": "model_a", "confidence": 0.85})
    insights = factory.LazyFunction(lambda: [])
    recommendations = factory.LazyFunction(
        lambda: ["Increase sample size", "Test different parameters"]
    )
    created_at = factory.LazyFunction(datetime.utcnow)


# DTO Factories
class TestSampleDTOFactory(BaseFactory):
    """Factory for TestSampleDTO."""

    class Meta:
        model = TestSampleDTO

    prompt = factory.Faker("text", max_nb_chars=200)
    expected_response = factory.Faker("text", max_nb_chars=300)
    difficulty = factory.Faker("random_element", elements=[e.value for e in DifficultyLevel])
    metadata = factory.LazyFunction(lambda: {"source": "synthetic"})


class ModelConfigurationDTOFactory(BaseFactory):
    """Factory for ModelConfigurationDTO."""

    class Meta:
        model = ModelConfigurationDTO

    model_id = factory.Faker("slug")
    provider_name = factory.Faker("random_element", elements=["openai", "anthropic", "google"])
    parameters = factory.LazyFunction(
        lambda: {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9}
    )
    weight = factory.Faker("pyfloat", min_value=0.1, max_value=1.0)


class EvaluationConfigurationDTOFactory(BaseFactory):
    """Factory for EvaluationConfigurationDTO."""

    class Meta:
        model = EvaluationConfigurationDTO

    template_id = factory.LazyFunction(BaseFactory._make_uuid)
    judge_count = factory.Faker("random_int", min=1, max=5)
    consensus_threshold = factory.Faker("pyfloat", min_value=0.6, max_value=0.9)
    quality_threshold = factory.Faker("pyfloat", min_value=0.7, max_value=0.95)


class TestConfigurationDTOFactory(BaseFactory):
    """Factory for TestConfigurationDTO."""

    class Meta:
        model = TestConfigurationDTO

    models = factory.List(
        [
            factory.SubFactory(ModelConfigurationDTOFactory),
            factory.SubFactory(ModelConfigurationDTOFactory),
        ]
    )
    evaluation = factory.SubFactory(EvaluationConfigurationDTOFactory)
    max_cost = factory.SubFactory(MoneyFactory)
    description = factory.Faker("text", max_nb_chars=200)


class CreateTestCommandDTOFactory(BaseFactory):
    """Factory for CreateTestCommandDTO."""

    class Meta:
        model = CreateTestCommandDTO

    name = factory.Faker("sentence", nb_words=4)
    configuration = factory.SubFactory(TestConfigurationDTOFactory)
    samples = factory.List([factory.SubFactory(TestSampleDTOFactory) for _ in range(20)])


# Specialized Factories for Testing Scenarios
class HighVolumeTestFactory(TestFactory):
    """Factory for high-volume test scenarios."""

    samples = factory.List([factory.SubFactory(TestSampleFactory) for _ in range(1000)])


class MultiModelTestFactory(TestFactory):
    """Factory for multi-model test scenarios."""

    @factory.lazy_attribute
    def configuration(self):
        config = TestConfigurationFactory()
        config.model_configs = [ModelConfigFactory() for _ in range(5)]
        return config


class PerformanceTestDataFactory(BaseFactory):
    """Factory for performance test data."""

    class Meta:
        model = dict

    response_times = factory.List(
        [factory.Faker("pyfloat", min_value=0.1, max_value=2.0) for _ in range(100)]
    )
    error_rates = factory.List(
        [factory.Faker("pyfloat", min_value=0.0, max_value=0.05) for _ in range(100)]
    )
    throughput = factory.List([factory.Faker("random_int", min=50, max=200) for _ in range(100)])


# Helper functions for batch creation
def create_test_with_samples(sample_count: int = 50) -> Test:
    """Create a test with specified number of samples."""
    test = TestFactory()
    test.samples = [TestSampleFactory(test_id=test.id) for _ in range(sample_count)]
    return test


def create_model_responses_for_test(test: Test, models_count: int = 2) -> List[ModelResponse]:
    """Create model responses for all samples in a test."""
    responses = []
    for sample in test.samples:
        for i in range(models_count):
            response = ModelResponseFactory()
            response.prompt = sample.prompt
            responses.append(response)
    return responses


def create_evaluation_results_for_responses(
    responses: List[ModelResponse], judges_count: int = 3
) -> List[EvaluationResult]:
    """Create evaluation results for model responses."""
    results = []
    for response in responses:
        for _ in range(judges_count):
            result = EvaluationResultFactory()
            result.model_response_id = response.id
            results.append(result)
    return results


# Test data presets
TEST_PRESETS = {
    "small_test": {"factory": TestFactory, "samples": 10, "models": 2, "judges": 3},
    "medium_test": {"factory": TestFactory, "samples": 50, "models": 3, "judges": 3},
    "large_test": {"factory": HighVolumeTestFactory, "samples": 500, "models": 5, "judges": 5},
}


def create_test_preset(preset_name: str) -> dict:
    """Create a test using a predefined preset."""
    if preset_name not in TEST_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    preset = TEST_PRESETS[preset_name]
    test = preset["factory"]()

    # Create samples
    test.samples = [TestSampleFactory(test_id=test.id) for _ in range(preset["samples"])]

    # Create model configs
    test.configuration.model_configs = [ModelConfigFactory() for _ in range(preset["models"])]

    return {
        "test": test,
        "samples": test.samples,
        "models": test.configuration.model_configs,
        "judges_count": preset["judges"],
    }
