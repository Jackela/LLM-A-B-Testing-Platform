"""Test configuration entity for Test Management domain."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..exceptions import BusinessRuleViolation, ValidationError
from ..value_objects.validation_result import ValidationResult


@dataclass
class TestConfiguration:
    """Test configuration entity defining parameters for A/B test execution."""

    models: List[str]
    max_tokens: int
    temperature: float
    timeout_seconds: int = 30
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize configuration after creation."""
        # Validate critical business rules and basic validation constraints
        if len(self.models) < 2:
            raise BusinessRuleViolation("At least 2 models are required for A/B testing")

        # Validate temperature range
        if not (0.0 <= self.temperature <= 2.0):
            raise ValidationError("temperature must be between 0.0 and 2.0")

        # Validate max_tokens
        if self.max_tokens <= 0:
            raise ValidationError("max_tokens must be greater than 0")

    def validate(self) -> ValidationResult:
        """Validate the test configuration."""
        errors = []
        warnings = []

        # Validate models
        if len(self.models) < 2:
            errors.append("At least 2 models are required for A/B testing")

        if not all(isinstance(model, str) and model.strip() for model in self.models):
            errors.append("All model names must be non-empty strings")

        if len(self.models) != len(set(self.models)):
            errors.append("Model names must be unique")

        # Validate max_tokens
        if self.max_tokens <= 0:
            errors.append("max_tokens must be greater than 0")
        elif self.max_tokens > 32000:
            warnings.append("max_tokens is very high (>32000), may cause performance issues")

        # Validate temperature
        if not (0.0 <= self.temperature <= 2.0):
            errors.append("temperature must be between 0.0 and 2.0")

        # Validate timeout
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be greater than 0")
        elif self.timeout_seconds > 300:
            warnings.append("timeout_seconds is very high (>300s), tests may take a long time")

        # Validate optional parameters
        if self.top_p is not None:
            if not (0.0 <= self.top_p <= 1.0):
                errors.append("top_p must be between 0.0 and 1.0")

        if self.frequency_penalty is not None:
            if not (-2.0 <= self.frequency_penalty <= 2.0):
                errors.append("frequency_penalty must be between -2.0 and 2.0")

        if self.presence_penalty is not None:
            if not (-2.0 <= self.presence_penalty <= 2.0):
                errors.append("presence_penalty must be between -2.0 and 2.0")

        # Validate stop sequences
        if len(self.stop_sequences) > 4:
            warnings.append("More than 4 stop sequences may not be supported by all models")

        if any(not isinstance(seq, str) for seq in self.stop_sequences):
            errors.append("All stop sequences must be strings")

        # Create and return validation result
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=tuple(errors), warnings=tuple(warnings))

    def get_model_parameters(self) -> Dict[str, Any]:
        """Get parameters dictionary for model calls."""
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.top_p is not None:
            params["top_p"] = self.top_p

        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty

        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty

        if self.stop_sequences:
            params["stop"] = self.stop_sequences

        if self.system_prompt:
            params["system"] = self.system_prompt

        # Add custom parameters
        params.update(self.custom_parameters)

        return params

    def estimate_cost_per_sample(self) -> float:
        """Estimate cost per sample based on configuration."""
        # Simplified cost estimation - in real implementation would use actual model pricing
        base_cost_per_token = 0.0001  # $0.0001 per token
        total_tokens_estimate = self.max_tokens * len(self.models)
        return total_tokens_estimate * base_cost_per_token

    def supports_streaming(self) -> bool:
        """Check if configuration supports streaming responses."""
        # Check if any models support streaming (simplified check)
        streaming_models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
        return any(model in streaming_models for model in self.models)

    def clone_with_modifications(self, **modifications) -> "TestConfiguration":
        """Create a copy of the configuration with modifications."""
        current_dict = {
            "models": self.models.copy(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop_sequences": self.stop_sequences.copy(),
            "system_prompt": self.system_prompt,
            "custom_parameters": self.custom_parameters.copy(),
        }

        current_dict.update(modifications)
        return TestConfiguration(**current_dict)

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"TestConfiguration(models={len(self.models)}, "
            f"max_tokens={self.max_tokens}, temp={self.temperature})"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, TestConfiguration):
            return False

        return (
            self.models == other.models
            and self.max_tokens == other.max_tokens
            and self.temperature == other.temperature
            and self.timeout_seconds == other.timeout_seconds
            and self.top_p == other.top_p
            and self.frequency_penalty == other.frequency_penalty
            and self.presence_penalty == other.presence_penalty
            and self.stop_sequences == other.stop_sequences
            and self.system_prompt == other.system_prompt
            and self.custom_parameters == other.custom_parameters
        )
