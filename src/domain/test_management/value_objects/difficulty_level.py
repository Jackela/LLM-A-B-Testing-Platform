"""Difficulty level value object for Test Management domain."""

from enum import Enum
from typing import Union


class DifficultyLevel(Enum):
    """Sample difficulty level enumeration."""

    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"

    def weight(self) -> float:
        """Get difficulty weight for calculations."""
        weight_map = {
            DifficultyLevel.EASY: 1.0,
            DifficultyLevel.MEDIUM: 1.5,
            DifficultyLevel.HARD: 2.0,
        }
        return weight_map[self]

    def score_factor(self) -> float:
        """Get score adjustment factor based on difficulty."""
        factor_map = {
            DifficultyLevel.EASY: 0.8,
            DifficultyLevel.MEDIUM: 1.0,
            DifficultyLevel.HARD: 1.2,
        }
        return factor_map[self]

    @classmethod
    def from_string(cls, value: str) -> "DifficultyLevel":
        """Create difficulty level from string value."""
        normalized_value = value.upper().strip()

        for level in cls:
            if level.value == normalized_value:
                return level

        raise ValueError(f"Invalid difficulty level: {value}")

    def __lt__(self, other: "DifficultyLevel") -> bool:
        """Compare difficulty levels for ordering."""
        if not isinstance(other, DifficultyLevel):
            raise TypeError(f"Cannot compare DifficultyLevel with {type(other)}")

        order_map = {
            DifficultyLevel.EASY: 1,
            DifficultyLevel.MEDIUM: 2,
            DifficultyLevel.HARD: 3,
        }
        return order_map[self] < order_map[other]

    def __le__(self, other: "DifficultyLevel") -> bool:
        """Less than or equal comparison."""
        return self < other or self == other

    def __gt__(self, other: "DifficultyLevel") -> bool:
        """Greater than comparison."""
        return not self <= other

    def __ge__(self, other: "DifficultyLevel") -> bool:
        """Greater than or equal comparison."""
        return not self < other

    def __str__(self) -> str:
        """String representation of difficulty."""
        return self.value

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"DifficultyLevel.{self.name}"
