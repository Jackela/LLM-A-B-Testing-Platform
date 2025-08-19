"""Scoring scale value object."""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, Union

from ..exceptions import ValidationError


@dataclass(frozen=True)
class ScoringScale:
    """Value object representing evaluation scoring scale."""

    min_score: Decimal
    max_score: Decimal
    scale_type: str  # "continuous", "discrete", "likert"
    scale_steps: int  # Number of discrete steps for discrete scales
    description: str

    def __post_init__(self):
        """Validate scoring scale parameters."""
        if self.min_score >= self.max_score:
            raise ValidationError("Minimum score must be less than maximum score")

        if self.scale_type not in ["continuous", "discrete", "likert"]:
            raise ValidationError(f"Invalid scale type: {self.scale_type}")

        if self.scale_type in ["discrete", "likert"] and self.scale_steps < 2:
            raise ValidationError("Discrete scales must have at least 2 steps")

        if self.scale_type == "likert" and self.scale_steps > 10:
            raise ValidationError("Likert scales should not exceed 10 points")

    @classmethod
    def create_five_point_likert(cls) -> "ScoringScale":
        """Factory method for standard 5-point Likert scale."""
        return cls(
            min_score=Decimal("1"),
            max_score=Decimal("5"),
            scale_type="likert",
            scale_steps=5,
            description="5-point Likert scale (1=Strongly Disagree, 5=Strongly Agree)",
        )

    @classmethod
    def create_continuous_zero_to_one(cls) -> "ScoringScale":
        """Factory method for continuous 0-1 scale."""
        return cls(
            min_score=Decimal("0.0"),
            max_score=Decimal("1.0"),
            scale_type="continuous",
            scale_steps=0,  # Not applicable for continuous
            description="Continuous scale from 0.0 to 1.0",
        )

    @classmethod
    def create_percentage_scale(cls) -> "ScoringScale":
        """Factory method for percentage scale."""
        return cls(
            min_score=Decimal("0"),
            max_score=Decimal("100"),
            scale_type="continuous",
            scale_steps=0,  # Not applicable for continuous
            description="Percentage scale from 0 to 100",
        )

    def is_valid_score(self, score: Union[float, int, Decimal]) -> bool:
        """Check if score is valid for this scale."""
        try:
            score_decimal = Decimal(str(score))

            # Check bounds
            if score_decimal < self.min_score or score_decimal > self.max_score:
                return False

            # Check discrete step requirements
            if self.scale_type == "discrete" or self.scale_type == "likert":
                step_size = (self.max_score - self.min_score) / (self.scale_steps - 1)
                # Check if score aligns with discrete steps
                relative_score = score_decimal - self.min_score
                steps_from_min = relative_score / step_size
                return abs(steps_from_min - round(steps_from_min)) < Decimal("0.001")

            return True

        except (ValueError, TypeError):
            return False

    def normalize_score(self, score: Union[float, int, Decimal]) -> Decimal:
        """Normalize score to 0-1 range."""
        if not self.is_valid_score(score):
            raise ValidationError(f"Invalid score {score} for scale {self.description}")

        score_decimal = Decimal(str(score))
        range_size = self.max_score - self.min_score
        normalized = (score_decimal - self.min_score) / range_size

        # Round to avoid floating point precision issues
        return normalized.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def denormalize_score(self, normalized_score: Union[float, Decimal]) -> Decimal:
        """Convert normalized score (0-1) back to original scale."""
        normalized_decimal = Decimal(str(normalized_score))

        if normalized_decimal < 0 or normalized_decimal > 1:
            raise ValidationError("Normalized score must be between 0 and 1")

        range_size = self.max_score - self.min_score
        original_score = self.min_score + (normalized_decimal * range_size)

        # Round to appropriate precision for scale type
        if self.scale_type == "likert":
            return original_score.quantize(Decimal("1"))
        else:
            return original_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def get_scale_midpoint(self) -> Decimal:
        """Get the midpoint of the scale."""
        return (self.min_score + self.max_score) / 2

    def get_scale_range(self) -> Decimal:
        """Get the range of the scale."""
        return self.max_score - self.min_score

    def get_discrete_values(self) -> list[Decimal]:
        """Get all valid discrete values for discrete/Likert scales."""
        if self.scale_type == "continuous":
            raise ValidationError("Cannot get discrete values for continuous scale")

        values = []
        step_size = self.get_scale_range() / (self.scale_steps - 1)

        for i in range(self.scale_steps):
            value = self.min_score + (step_size * i)
            values.append(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        return values

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "min_score": str(self.min_score),
            "max_score": str(self.max_score),
            "scale_type": self.scale_type,
            "scale_steps": self.scale_steps,
            "description": self.description,
        }

        if self.scale_type in ["discrete", "likert"]:
            result["discrete_values"] = [str(v) for v in self.get_discrete_values()]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringScale":
        """Create from dictionary representation."""
        return cls(
            min_score=Decimal(data["min_score"]),
            max_score=Decimal(data["max_score"]),
            scale_type=data["scale_type"],
            scale_steps=data["scale_steps"],
            description=data["description"],
        )

    def __str__(self) -> str:
        """String representation."""
        return f"ScoringScale({self.scale_type}, {self.min_score}-{self.max_score})"
