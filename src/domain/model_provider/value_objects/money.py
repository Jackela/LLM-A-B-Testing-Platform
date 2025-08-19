"""Money value object for cost calculations."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Union

from ..exceptions import ValidationError


@dataclass(frozen=True)
class Money:
    """Money value object for handling currency amounts."""

    amount: Decimal
    currency: str

    def __post_init__(self):
        """Validate money object after initialization."""
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", Decimal(str(self.amount)))

        if not isinstance(self.currency, str):
            raise ValidationError("Currency must be a string")

        if len(self.currency) != 3:
            raise ValidationError("Currency must be a 3-letter code")

    def __add__(self, other: "Money") -> "Money":
        """Add two money amounts."""
        if not isinstance(other, Money):
            raise ValidationError("Can only add Money to Money")

        if self.currency != other.currency:
            raise ValidationError("Cannot add money with different currencies")

        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        """Subtract two money amounts."""
        if not isinstance(other, Money):
            raise ValidationError("Can only subtract Money from Money")

        if self.currency != other.currency:
            raise ValidationError("Cannot subtract money with different currencies")

        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, multiplier: Union[int, float, Decimal]) -> "Money":
        """Multiply money by a number."""
        if not isinstance(multiplier, (int, float, Decimal)):
            raise ValidationError("Can only multiply money by a number")

        return Money(self.amount * Decimal(str(multiplier)), self.currency)

    def __truediv__(self, divisor: Union[int, float, Decimal]) -> "Money":
        """Divide money by a number."""
        if not isinstance(divisor, (int, float, Decimal)):
            raise ValidationError("Can only divide money by a number")

        if divisor == 0:
            raise ValidationError("Cannot divide by zero")

        return Money(self.amount / Decimal(str(divisor)), self.currency)

    def __eq__(self, other) -> bool:
        """Check equality with another Money object."""
        if not isinstance(other, Money):
            return False

        return self.amount == other.amount and self.currency == other.currency

    def __lt__(self, other: "Money") -> bool:
        """Less than comparison."""
        if not isinstance(other, Money):
            raise ValidationError("Can only compare Money with Money")

        if self.currency != other.currency:
            raise ValidationError("Cannot compare money with different currencies")

        return self.amount < other.amount

    def __le__(self, other: "Money") -> bool:
        """Less than or equal comparison."""
        return self < other or self == other

    def __gt__(self, other: "Money") -> bool:
        """Greater than comparison."""
        if not isinstance(other, Money):
            raise ValidationError("Can only compare Money with Money")

        if self.currency != other.currency:
            raise ValidationError("Cannot compare money with different currencies")

        return self.amount > other.amount

    def __ge__(self, other: "Money") -> bool:
        """Greater than or equal comparison."""
        return self > other or self == other

    def __str__(self) -> str:
        """String representation."""
        return f"{self.amount} {self.currency}"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Money(amount={self.amount}, currency='{self.currency}')"

    def is_zero(self) -> bool:
        """Check if amount is zero."""
        return self.amount == Decimal("0.00")

    def is_positive(self) -> bool:
        """Check if amount is positive."""
        return self.amount > Decimal("0.00")

    def is_negative(self) -> bool:
        """Check if amount is negative."""
        return self.amount < Decimal("0.00")

    def abs(self) -> "Money":
        """Get absolute value."""
        return Money(abs(self.amount), self.currency)

    def round_to_cents(self) -> "Money":
        """Round to nearest cent (2 decimal places)."""
        rounded_amount = self.amount.quantize(Decimal("0.01"))
        return Money(rounded_amount, self.currency)

    @classmethod
    def zero(cls, currency: str) -> "Money":
        """Create zero money in the specified currency."""
        return cls(Decimal("0.00"), currency)

    @classmethod
    def from_float(cls, amount: float, currency: str) -> "Money":
        """Create Money from float amount."""
        return cls(Decimal(str(amount)), currency)

    @classmethod
    def from_cents(cls, cents: int, currency: str) -> "Money":
        """Create Money from cents (integer)."""
        return cls(Decimal(cents) / Decimal("100"), currency)
