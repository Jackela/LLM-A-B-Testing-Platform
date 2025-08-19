"""Test sample entity for Test Management domain."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..exceptions import BusinessRuleViolation
from ..value_objects.difficulty_level import DifficultyLevel


@dataclass
class TestSample:
    """Test sample entity representing a single test case."""

    prompt: str
    difficulty: DifficultyLevel
    expected_output: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    evaluation_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _is_frozen: bool = field(default=False, init=False)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to implement immutability after evaluation starts."""
        # Allow setting of evaluation results and private attributes
        if name in ("evaluation_results", "_is_frozen") or name.startswith("_"):
            super().__setattr__(name, value)
            return

        # Check if sample is frozen (has evaluations)
        if hasattr(self, "_is_frozen") and self._is_frozen:
            raise BusinessRuleViolation(
                f"Cannot modify {name} after evaluation has started. "
                "Sample properties are immutable once evaluation begins."
            )

        super().__setattr__(name, value)

    @property
    def is_evaluated(self) -> bool:
        """Check if sample has any evaluation results."""
        return len(self.evaluation_results) > 0

    def add_evaluation_result(self, model_name: str, result: Dict[str, Any]) -> None:
        """Add evaluation result for a specific model."""
        # Validate result format
        if not isinstance(result, dict):
            raise BusinessRuleViolation("Evaluation result must be a dictionary")

        if "score" not in result:
            raise BusinessRuleViolation("Evaluation result must contain a 'score' field")

        score = result["score"]
        if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
            raise BusinessRuleViolation("Score must be a number between 0.0 and 1.0")

        # Freeze the sample after first evaluation
        if not self._is_frozen and not self.is_evaluated:
            self._is_frozen = True

        # Store the evaluation result
        self.evaluation_results[model_name] = result.copy()

    def get_evaluation_result(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get evaluation result for a specific model."""
        return self.evaluation_results.get(model_name)

    def get_average_score(self) -> float:
        """Calculate average score across all evaluations."""
        if not self.evaluation_results:
            return 0.0

        scores = [result["score"] for result in self.evaluation_results.values()]
        return sum(scores) / len(scores)

    def get_weighted_score(self) -> float:
        """Calculate weighted score based on difficulty level."""
        average_score = self.get_average_score()
        return average_score * self.difficulty.score_factor()

    def get_model_scores(self) -> Dict[str, float]:
        """Get scores for each model."""
        return {model: result["score"] for model, result in self.evaluation_results.items()}

    def has_evaluation_for_model(self, model_name: str) -> bool:
        """Check if sample has evaluation for specific model."""
        return model_name in self.evaluation_results

    def get_evaluation_count(self) -> int:
        """Get number of models that have evaluated this sample."""
        return len(self.evaluation_results)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the sample."""
        if self._is_frozen:
            raise BusinessRuleViolation("Cannot modify tags after evaluation has started")

        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the sample."""
        if self._is_frozen:
            raise BusinessRuleViolation("Cannot modify tags after evaluation has started")

        if tag in self.tags:
            self.tags.remove(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if sample has a specific tag."""
        return tag in self.tags

    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata field."""
        if self._is_frozen:
            raise BusinessRuleViolation("Cannot modify metadata after evaluation has started")

        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata field value."""
        return self.metadata.get(key, default)

    def calculate_evaluation_stats(self) -> Dict[str, Any]:
        """Calculate evaluation statistics."""
        if not self.evaluation_results:
            return {
                "count": 0,
                "average_score": 0.0,
                "weighted_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "score_variance": 0.0,
            }

        scores = [result["score"] for result in self.evaluation_results.values()]
        average = sum(scores) / len(scores)
        variance = sum((score - average) ** 2 for score in scores) / len(scores)

        return {
            "count": len(scores),
            "average_score": average,
            "weighted_score": average * self.difficulty.score_factor(),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_variance": variance,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary representation."""
        return {
            "id": str(self.id),
            "prompt": self.prompt,
            "difficulty": self.difficulty.value,
            "expected_output": self.expected_output,
            "tags": self.tags.copy(),
            "metadata": self.metadata.copy(),
            "evaluation_results": {
                model: result.copy() for model, result in self.evaluation_results.items()
            },
            "is_evaluated": self.is_evaluated,
            "is_frozen": self._is_frozen,
            "average_score": self.get_average_score(),
            "weighted_score": self.get_weighted_score(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestSample":
        """Create sample from dictionary representation."""
        sample = cls(
            prompt=data["prompt"],
            difficulty=DifficultyLevel.from_string(data["difficulty"]),
            expected_output=data.get("expected_output"),
            tags=data.get("tags", []).copy(),
            metadata=data.get("metadata", {}).copy(),
            id=UUID(data["id"]) if "id" in data else uuid4(),
        )

        # Restore evaluation results if present
        if "evaluation_results" in data and data["evaluation_results"]:
            for model, result in data["evaluation_results"].items():
                sample.add_evaluation_result(model, result)

        return sample

    def __str__(self) -> str:
        """String representation of sample."""
        prompt_preview = self.prompt[:50] + "..." if len(self.prompt) > 50 else self.prompt
        return (
            f"TestSample(id={str(self.id)[:8]}..., "
            f"prompt='{prompt_preview}', "
            f"difficulty={self.difficulty.value}, "
            f"evaluated={self.is_evaluated})"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison based on ID."""
        if not isinstance(other, TestSample):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)
