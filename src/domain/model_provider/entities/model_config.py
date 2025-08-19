"""Model configuration entity."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict

from ..exceptions import ValidationError
from ..value_objects.model_category import ModelCategory


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration entity defining a specific LLM model."""

    model_id: str
    display_name: str
    max_tokens: int
    cost_per_input_token: Decimal
    cost_per_output_token: Decimal
    supports_streaming: bool
    model_category: ModelCategory
    parameters: Dict[str, Any]

    def __post_init__(self):
        """Validate model configuration after creation."""
        self.validate()

    def validate(self) -> None:
        """Validate model configuration."""
        if not self.model_id:
            raise ValidationError("Model ID is required")

        if not self.display_name:
            raise ValidationError("Display name is required")

        if self.max_tokens <= 0:
            raise ValidationError("Max tokens must be positive")

        if self.cost_per_input_token < 0 or self.cost_per_output_token < 0:
            raise ValidationError("Token costs cannot be negative")

        if not isinstance(self.model_category, ModelCategory):
            raise ValidationError("Model category must be a valid ModelCategory")

        # Validate parameters based on model category
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate model-specific parameters."""
        if "temperature" in self.parameters:
            temp = self.parameters["temperature"]
            if not isinstance(temp, (int, float)) or not (0.0 <= temp <= 2.0):
                raise ValidationError("Temperature must be between 0.0 and 2.0")

        if "top_p" in self.parameters:
            top_p = self.parameters["top_p"]
            if not isinstance(top_p, (int, float)) or not (0.0 <= top_p <= 1.0):
                raise ValidationError("top_p must be between 0.0 and 1.0")

        if "top_k" in self.parameters:
            top_k = self.parameters["top_k"]
            if not isinstance(top_k, int) or top_k < 1:
                raise ValidationError("top_k must be a positive integer")

        if "frequency_penalty" in self.parameters:
            freq_penalty = self.parameters["frequency_penalty"]
            if not isinstance(freq_penalty, (int, float)) or not (-2.0 <= freq_penalty <= 2.0):
                raise ValidationError("frequency_penalty must be between -2.0 and 2.0")

        if "presence_penalty" in self.parameters:
            pres_penalty = self.parameters["presence_penalty"]
            if not isinstance(pres_penalty, (int, float)) or not (-2.0 <= pres_penalty <= 2.0):
                raise ValidationError("presence_penalty must be between -2.0 and 2.0")

        if "max_output_tokens" in self.parameters:
            max_output = self.parameters["max_output_tokens"]
            if not isinstance(max_output, int) or max_output <= 0:
                raise ValidationError("max_output_tokens must be a positive integer")

        if "stop_sequences" in self.parameters:
            stop_seq = self.parameters["stop_sequences"]
            if not isinstance(stop_seq, list):
                raise ValidationError("stop_sequences must be a list")

            if len(stop_seq) > 10:
                raise ValidationError("stop_sequences cannot have more than 10 items")

            if not all(isinstance(seq, str) for seq in stop_seq):
                raise ValidationError("All stop sequences must be strings")

    def calculate_estimated_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Calculate estimated cost for token usage."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValidationError("Token counts cannot be negative")

        input_cost = Decimal(input_tokens) * self.cost_per_input_token
        output_cost = Decimal(output_tokens) * self.cost_per_output_token
        return input_cost + output_cost

    def supports_parameter(self, parameter_name: str) -> bool:
        """Check if parameter is supported by this model."""
        return parameter_name in self.parameters

    def get_parameter_value(self, parameter_name: str, default_value=None):
        """Get parameter value with optional default."""
        return self.parameters.get(parameter_name, default_value)

    def get_effective_max_tokens(self) -> int:
        """Get effective maximum tokens (considering max_output_tokens parameter)."""
        max_output = self.get_parameter_value("max_output_tokens")
        if max_output is not None:
            return min(self.max_tokens, max_output)
        return self.max_tokens

    def is_compatible_with_use_case(self, use_case: str) -> bool:
        """Check if model is compatible with a specific use case."""
        use_case_lower = use_case.lower()
        category_use_cases = [uc.lower() for uc in self.model_category.typical_use_cases]

        return any(use_case_lower in category_use_case for category_use_case in category_use_cases)

    def get_recommended_parameters(self) -> Dict[str, Any]:
        """Get recommended parameters for this model."""
        recommendations = {}

        # Add category-specific recommendations
        if self.model_category == ModelCategory.CHAT_COMPLETION:
            recommendations.update(
                {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                }
            )
        elif self.model_category == ModelCategory.CODE_GENERATION:
            recommendations.update(
                {"temperature": 0.2, "top_p": 0.95, "stop_sequences": ["\n\n", "```"]}
            )
        elif self.model_category == ModelCategory.TEXT_GENERATION:
            recommendations.update({"temperature": 1.0, "top_p": 1.0})

        # Only include supported parameters
        return {k: v for k, v in recommendations.items() if self.supports_parameter(k)}

    def clone_with_parameters(self, **new_parameters) -> "ModelConfig":
        """Create a new ModelConfig with updated parameters."""
        updated_parameters = self.parameters.copy()
        updated_parameters.update(new_parameters)

        # Create new instance with updated parameters
        return ModelConfig(
            model_id=self.model_id,
            display_name=self.display_name,
            max_tokens=self.max_tokens,
            cost_per_input_token=self.cost_per_input_token,
            cost_per_output_token=self.cost_per_output_token,
            supports_streaming=self.supports_streaming,
            model_category=self.model_category,
            parameters=updated_parameters,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "max_tokens": self.max_tokens,
            "cost_per_input_token": str(self.cost_per_input_token),
            "cost_per_output_token": str(self.cost_per_output_token),
            "supports_streaming": self.supports_streaming,
            "model_category": self.model_category.value,
            "parameters": self.parameters.copy(),
        }

    def __str__(self) -> str:
        """String representation."""
        return f"ModelConfig(id='{self.model_id}', category={self.model_category.name})"
