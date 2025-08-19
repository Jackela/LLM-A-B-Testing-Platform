"""Data Transfer Objects for test configuration."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...domain.model_provider.value_objects.money import Money
from ...domain.test_management.value_objects.difficulty_level import DifficultyLevel


@dataclass(frozen=True)
class ModelConfigurationDTO:
    """DTO for model configuration in test setup."""

    model_id: str
    provider_name: str
    parameters: Dict[str, Any]
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate DTO after creation."""
        if not self.model_id.strip():
            raise ValueError("Model ID cannot be empty")
        if not self.provider_name.strip():
            raise ValueError("Provider name cannot be empty")
        if self.weight <= 0:
            raise ValueError("Weight must be positive")


@dataclass(frozen=True)
class EvaluationConfigurationDTO:
    """DTO for evaluation configuration in test setup."""

    template_id: str
    judge_count: int = 3
    consensus_threshold: float = 0.7
    quality_threshold: float = 0.8
    dimensions: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate DTO after creation."""
        if not self.template_id.strip():
            raise ValueError("Template ID cannot be empty")
        if self.judge_count < 1:
            raise ValueError("Judge count must be at least 1")
        if not 0 < self.consensus_threshold <= 1:
            raise ValueError("Consensus threshold must be between 0 and 1")
        if not 0 < self.quality_threshold <= 1:
            raise ValueError("Quality threshold must be between 0 and 1")
        if self.dimensions is None:
            object.__setattr__(self, "dimensions", [])


@dataclass(frozen=True)
class TestConfigurationDTO:
    """DTO for complete test configuration."""

    models: List[ModelConfigurationDTO]
    evaluation: EvaluationConfigurationDTO
    max_cost: Optional[Money] = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate DTO after creation."""
        if not self.models:
            raise ValueError("At least one model configuration is required")
        if len(self.models) < 2:
            raise ValueError("At least two models are required for A/B testing")

        # Validate unique model+provider combinations
        model_keys = set()
        for model_config in self.models:
            key = (model_config.model_id, model_config.provider_name)
            if key in model_keys:
                raise ValueError(
                    f"Duplicate model configuration: {model_config.model_id} from {model_config.provider_name}"
                )
            model_keys.add(key)


@dataclass(frozen=True)
class TestSampleDTO:
    """DTO for test sample data."""

    prompt: str
    expected_output: Optional[str] = None
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate DTO after creation."""
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class CreateTestCommandDTO:
    """DTO for creating a new test."""

    name: str
    configuration: TestConfigurationDTO
    samples: List[TestSampleDTO]
    creator_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate DTO after creation."""
        if not self.name.strip():
            raise ValueError("Test name cannot be empty")
        if not self.samples:
            raise ValueError("At least one sample is required")
        if len(self.samples) < 10:
            raise ValueError("At least 10 samples are required for statistical validity")
        if len(self.samples) > 10000:
            raise ValueError("Maximum 10,000 samples allowed")


@dataclass(frozen=True)
class UpdateTestConfigurationCommandDTO:
    """DTO for updating test configuration."""

    test_id: UUID
    configuration: TestConfigurationDTO
    updater_id: Optional[str] = None


@dataclass(frozen=True)
class AddSamplesCommandDTO:
    """DTO for adding samples to a test."""

    test_id: UUID
    samples: List[TestSampleDTO]
    creator_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate DTO after creation."""
        if not self.samples:
            raise ValueError("At least one sample is required")


@dataclass(frozen=True)
class StartTestCommandDTO:
    """DTO for starting a test."""

    test_id: UUID
    started_by: Optional[str] = None


@dataclass(frozen=True)
class TestMonitoringResultDTO:
    """DTO for test monitoring result."""

    test_id: UUID
    status: str
    progress: float
    total_samples: int
    evaluated_samples: int
    model_scores: Dict[str, float]
    estimated_remaining_time: float
    current_cost: Money
    errors: List[str]


@dataclass(frozen=True)
class TestResultDTO:
    """DTO for test execution result."""

    test_id: UUID
    status: str
    created_test: bool
    estimated_cost: Optional[Money] = None
    estimated_duration: Optional[float] = None
    errors: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Initialize defaults."""
        if self.errors is None:
            object.__setattr__(self, "errors", [])
