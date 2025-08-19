"""Performance score value object."""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from ..exceptions import ValidationError


@dataclass(frozen=True)
class PerformanceScore:
    """Performance score with confidence metrics."""

    score: Decimal
    confidence: Decimal
    sample_size: int
    standard_error: Optional[Decimal] = None

    def __post_init__(self):
        """Validate performance score."""
        if not (Decimal("0") <= self.score <= Decimal("1")):
            raise ValidationError(f"Score must be between 0 and 1, got {self.score}")

        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValidationError(f"Confidence must be between 0 and 1, got {self.confidence}")

        if self.sample_size < 0:
            raise ValidationError(f"Sample size must be non-negative, got {self.sample_size}")

        if self.standard_error is not None and self.standard_error < 0:
            raise ValidationError(f"Standard error must be non-negative, got {self.standard_error}")

    def is_high_confidence(self, threshold: Decimal = Decimal("0.8")) -> bool:
        """Check if score has high confidence."""
        return self.confidence >= threshold

    def is_reliable_sample_size(self, min_size: int = 30) -> bool:
        """Check if sample size is reliable for statistical inference."""
        return self.sample_size >= min_size

    def get_score_range(
        self, confidence_level: Decimal = Decimal("0.95")
    ) -> tuple[Decimal, Decimal]:
        """Get approximate score range based on standard error."""
        if self.standard_error is None:
            # If no standard error, use confidence as proxy for uncertainty
            margin = (Decimal("1") - self.confidence) * self.score
            return (max(Decimal("0"), self.score - margin), min(Decimal("1"), self.score + margin))

        # Use standard error for confidence bounds
        # Approximate 95% CI uses ~1.96 * SE, 90% uses ~1.645 * SE
        z_score = Decimal("1.96") if confidence_level >= Decimal("0.95") else Decimal("1.645")
        margin = z_score * self.standard_error

        return (max(Decimal("0"), self.score - margin), min(Decimal("1"), self.score + margin))

    def round(self, places: int = 3) -> "PerformanceScore":
        """Round score and confidence to specified decimal places."""
        return PerformanceScore(
            score=self.score.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP),
            confidence=self.confidence.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP),
            sample_size=self.sample_size,
            standard_error=(
                self.standard_error.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
                if self.standard_error
                else None
            ),
        )

    def __str__(self) -> str:
        """String representation."""
        confidence_pct = int(self.confidence * 100)
        return f"Score: {self.score} (confidence: {confidence_pct}%, n={self.sample_size})"
