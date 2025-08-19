"""Evaluation dimension entity."""

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from ..exceptions import ValidationError


@dataclass
class Dimension:
    """Entity representing an evaluation dimension."""

    dimension_id: UUID
    name: str
    description: str
    weight: Decimal  # Must sum to 1.0 across all dimensions in template
    scoring_criteria: Dict[int, str]  # score -> description
    is_required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dimension after creation."""
        if not self.dimension_id:
            self.dimension_id = uuid4()

        self._validate_dimension()

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        weight: Decimal,
        scoring_criteria: Dict[int, str],
        is_required: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Dimension":
        """Factory method to create dimension."""
        return cls(
            dimension_id=uuid4(),
            name=name,
            description=description,
            weight=weight,
            scoring_criteria=scoring_criteria,
            is_required=is_required,
            metadata=metadata or {},
        )

    def _validate_dimension(self) -> None:
        """Validate dimension properties."""
        if not self.name.strip():
            raise ValidationError("Dimension name cannot be empty")

        if not self.description.strip():
            raise ValidationError("Dimension description cannot be empty")

        if not (0 < self.weight <= 1):
            raise ValidationError("Dimension weight must be between 0 and 1")

        if not self.scoring_criteria:
            raise ValidationError("Dimension must have scoring criteria")

        # Validate scoring criteria
        scores = list(self.scoring_criteria.keys())
        if min(scores) < 1 or max(scores) > 5:
            raise ValidationError("Score keys must be between 1 and 5")

        # Check for consecutive scores
        scores.sort()
        if scores != list(range(min(scores), max(scores) + 1)):
            raise ValidationError("Scoring criteria must have consecutive integer keys")

        # Validate descriptions are not empty
        for score, description in self.scoring_criteria.items():
            if not description.strip():
                raise ValidationError(f"Empty description for score {score}")

    def get_score_range(self) -> tuple[int, int]:
        """Get the min and max scores for this dimension."""
        scores = list(self.scoring_criteria.keys())
        return (min(scores), max(scores))

    def is_valid_score(self, score: int) -> bool:
        """Check if score is valid for this dimension."""
        return score in self.scoring_criteria

    def get_score_description(self, score: int) -> Optional[str]:
        """Get description for a specific score."""
        return self.scoring_criteria.get(score)

    def normalize_score(self, score: int) -> Decimal:
        """Normalize score to 0-1 range."""
        if not self.is_valid_score(score):
            raise ValidationError(f"Invalid score {score} for dimension {self.name}")

        min_score, max_score = self.get_score_range()
        range_size = max_score - min_score

        if range_size == 0:
            return Decimal("1.0")

        normalized = Decimal(score - min_score) / Decimal(range_size)
        return normalized.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def denormalize_score(self, normalized_score: Decimal) -> int:
        """Convert normalized score back to original scale."""
        if not (0 <= normalized_score <= 1):
            raise ValidationError("Normalized score must be between 0 and 1")

        min_score, max_score = self.get_score_range()
        range_size = max_score - min_score

        original_score = min_score + int(normalized_score * range_size)

        # Ensure result is within valid range
        return max(min_score, min(max_score, original_score))

    def calculate_weighted_score(self, raw_score: int) -> Decimal:
        """Calculate weighted score for this dimension."""
        normalized = self.normalize_score(raw_score)
        weighted = normalized * self.weight
        return weighted.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def update_weight(self, new_weight: Decimal) -> None:
        """Update dimension weight."""
        if not (0 < new_weight <= 1):
            raise ValidationError("Dimension weight must be between 0 and 1")

        self.weight = new_weight.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def add_scoring_criterion(self, score: int, description: str) -> None:
        """Add or update scoring criterion."""
        if not (1 <= score <= 5):
            raise ValidationError("Score must be between 1 and 5")

        if not description.strip():
            raise ValidationError("Description cannot be empty")

        self.scoring_criteria[score] = description
        self._validate_dimension()

    def remove_scoring_criterion(self, score: int) -> None:
        """Remove scoring criterion."""
        if score not in self.scoring_criteria:
            raise ValidationError(f"Score {score} not found in criteria")

        if len(self.scoring_criteria) <= 2:
            raise ValidationError("Must have at least 2 scoring criteria")

        del self.scoring_criteria[score]
        self._validate_dimension()

    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata field."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata field value."""
        return self.metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimension_id": str(self.dimension_id),
            "name": self.name,
            "description": self.description,
            "weight": str(self.weight),
            "scoring_criteria": self.scoring_criteria.copy(),
            "is_required": self.is_required,
            "metadata": self.metadata.copy(),
            "score_range": self.get_score_range(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dimension":
        """Create from dictionary representation."""
        return cls(
            dimension_id=UUID(data["dimension_id"]),
            name=data["name"],
            description=data["description"],
            weight=Decimal(data["weight"]),
            scoring_criteria=data["scoring_criteria"],
            is_required=data.get("is_required", True),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """String representation."""
        return f"Dimension(name='{self.name}', weight={self.weight})"

    def __eq__(self, other) -> bool:
        """Equality comparison based on ID."""
        if not isinstance(other, Dimension):
            return False
        return self.dimension_id == other.dimension_id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.dimension_id)


# Standard evaluation dimensions
STANDARD_DIMENSIONS = {
    "accuracy": Dimension.create(
        name="accuracy",
        description="Factual correctness and truthfulness of the response",
        weight=Decimal("0.3"),
        scoring_criteria={
            5: "Completely accurate with all facts correct",
            4: "Mostly accurate with minor factual issues",
            3: "Partially accurate with some significant errors",
            2: "Many factual errors but some correct information",
            1: "Largely inaccurate or misleading",
        },
    ),
    "relevance": Dimension.create(
        name="relevance",
        description="How well the response addresses the prompt",
        weight=Decimal("0.25"),
        scoring_criteria={
            5: "Directly and completely addresses all aspects",
            4: "Addresses most aspects with good relevance",
            3: "Addresses some aspects but misses key points",
            2: "Partially relevant but significant gaps",
            1: "Largely irrelevant or off-topic",
        },
    ),
    "clarity": Dimension.create(
        name="clarity",
        description="Clarity and understandability of communication",
        weight=Decimal("0.25"),
        scoring_criteria={
            5: "Exceptionally clear and well-structured",
            4: "Clear and easy to understand",
            3: "Generally clear with some confusing parts",
            2: "Somewhat unclear but understandable",
            1: "Unclear and difficult to understand",
        },
    ),
    "usefulness": Dimension.create(
        name="usefulness",
        description="Practical value and actionability",
        weight=Decimal("0.2"),
        scoring_criteria={
            5: "Highly useful with actionable insights",
            4: "Useful with good practical value",
            3: "Moderately useful",
            2: "Limited usefulness",
            1: "Not useful or actionable",
        },
    ),
}
