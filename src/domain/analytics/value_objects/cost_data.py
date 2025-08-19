"""Cost data value object."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict

from ...model_provider.value_objects.money import Money
from ..exceptions import ValidationError


@dataclass(frozen=True)
class CostData:
    """Cost data for model performance analysis."""

    total_cost: Money
    cost_per_request: Money
    request_count: int
    cost_per_token: Money
    token_count: int

    def __post_init__(self):
        """Validate cost data."""
        if self.request_count < 0:
            raise ValidationError(f"Request count must be non-negative, got {self.request_count}")

        if self.token_count < 0:
            raise ValidationError(f"Token count must be non-negative, got {self.token_count}")

        # Validate consistency
        if self.request_count > 0:
            expected_cost_per_request = self.total_cost.amount / Decimal(str(self.request_count))
            if abs(expected_cost_per_request - self.cost_per_request.amount) > Decimal("0.001"):
                raise ValidationError(
                    "Cost per request is inconsistent with total cost and request count"
                )

        if self.token_count > 0:
            expected_cost_per_token = self.total_cost.amount / Decimal(str(self.token_count))
            if abs(expected_cost_per_token - self.cost_per_token.amount) > Decimal("0.0001"):
                raise ValidationError(
                    "Cost per token is inconsistent with total cost and token count"
                )

    def cost_efficiency_score(self, max_cost_per_request: Money) -> Decimal:
        """Calculate cost efficiency score (0-1, higher is more efficient)."""
        if max_cost_per_request.amount <= Decimal("0"):
            raise ValidationError("Max cost per request must be positive")

        if self.cost_per_request.amount <= Decimal("0"):
            return Decimal("1")  # Free is maximally efficient

        # Inverse relationship: lower cost = higher efficiency
        efficiency = max_cost_per_request.amount / self.cost_per_request.amount
        return min(Decimal("1"), efficiency)  # Cap at 1.0

    def tokens_per_dollar(self) -> Decimal:
        """Calculate tokens per dollar spent."""
        if self.total_cost.amount <= Decimal("0"):
            return Decimal("0")  # Avoid division by zero

        return Decimal(str(self.token_count)) / self.total_cost.amount

    def requests_per_dollar(self) -> Decimal:
        """Calculate requests per dollar spent."""
        if self.total_cost.amount <= Decimal("0"):
            return Decimal("0")  # Avoid division by zero

        return Decimal(str(self.request_count)) / self.total_cost.amount

    def average_tokens_per_request(self) -> Decimal:
        """Calculate average tokens per request."""
        if self.request_count <= 0:
            return Decimal("0")

        return Decimal(str(self.token_count)) / Decimal(str(self.request_count))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_cost": {
                "amount": str(self.total_cost.amount),
                "currency": self.total_cost.currency,
            },
            "cost_per_request": {
                "amount": str(self.cost_per_request.amount),
                "currency": self.cost_per_request.currency,
            },
            "request_count": self.request_count,
            "cost_per_token": {
                "amount": str(self.cost_per_token.amount),
                "currency": self.cost_per_token.currency,
            },
            "token_count": self.token_count,
            "tokens_per_dollar": str(self.tokens_per_dollar()),
            "requests_per_dollar": str(self.requests_per_dollar()),
            "average_tokens_per_request": str(self.average_tokens_per_request()),
        }

    @classmethod
    def create_from_totals(
        cls, total_cost: Money, request_count: int, token_count: int
    ) -> "CostData":
        """Create cost data from totals."""
        if request_count <= 0:
            raise ValidationError("Request count must be positive")

        if token_count <= 0:
            raise ValidationError("Token count must be positive")

        cost_per_request = Money(
            amount=total_cost.amount / Decimal(str(request_count)), currency=total_cost.currency
        )

        cost_per_token = Money(
            amount=total_cost.amount / Decimal(str(token_count)), currency=total_cost.currency
        )

        return cls(
            total_cost=total_cost,
            cost_per_request=cost_per_request,
            request_count=request_count,
            cost_per_token=cost_per_token,
            token_count=token_count,
        )

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Cost: {self.total_cost} ({self.request_count} requests, "
            f"{self.cost_per_request} per request)"
        )
