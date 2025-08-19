"""Confidence interval value object."""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Tuple

from ..exceptions import ValidationError


@dataclass(frozen=True)
class ConfidenceInterval:
    """Confidence interval value object."""

    lower_bound: Decimal
    upper_bound: Decimal
    confidence_level: Decimal

    def __post_init__(self):
        """Validate confidence interval."""
        if self.lower_bound > self.upper_bound:
            raise ValidationError(
                f"Lower bound ({self.lower_bound}) must be <= upper bound ({self.upper_bound})"
            )

        if not (Decimal("0") < self.confidence_level < Decimal("1")):
            raise ValidationError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )

    def width(self) -> Decimal:
        """Get width of confidence interval."""
        return self.upper_bound - self.lower_bound

    def midpoint(self) -> Decimal:
        """Get midpoint of confidence interval."""
        return (self.lower_bound + self.upper_bound) / Decimal("2")

    def contains(self, value: Decimal) -> bool:
        """Check if interval contains a value."""
        return self.lower_bound <= value <= self.upper_bound

    def contains_zero(self) -> bool:
        """Check if interval contains zero."""
        return self.contains(Decimal("0"))

    def margin_of_error(self) -> Decimal:
        """Get margin of error (half-width)."""
        return self.width() / Decimal("2")

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple of floats."""
        return (float(self.lower_bound), float(self.upper_bound))

    def round(self, places: int = 3) -> "ConfidenceInterval":
        """Round bounds to specified decimal places."""
        return ConfidenceInterval(
            lower_bound=self.lower_bound.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP),
            upper_bound=self.upper_bound.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP),
            confidence_level=self.confidence_level,
        )

    def __str__(self) -> str:
        """String representation."""
        confidence_pct = int(self.confidence_level * 100)
        return f"{confidence_pct}% CI: [{self.lower_bound}, {self.upper_bound}]"
