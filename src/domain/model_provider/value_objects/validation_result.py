"""Validation result value object."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def __post_init__(self):
        """Validate the validation result itself."""
        if not isinstance(self.errors, tuple):
            object.__setattr__(self, "errors", tuple(self.errors))

        if not isinstance(self.warnings, tuple):
            object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def error_count(self) -> int:
        """Get the number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Get the number of warnings."""
        return len(self.warnings)

    def add_error(self, error: str) -> "ValidationResult":
        """Add an error and return new ValidationResult."""
        new_errors = self.errors + (error,)
        return ValidationResult(
            is_valid=False, errors=new_errors, warnings=self.warnings  # Any errors make it invalid
        )

    def add_warning(self, warning: str) -> "ValidationResult":
        """Add a warning and return new ValidationResult."""
        new_warnings = self.warnings + (warning,)
        return ValidationResult(is_valid=self.is_valid, errors=self.errors, warnings=new_warnings)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge with another validation result."""
        merged_errors = self.errors + other.errors
        merged_warnings = self.warnings + other.warnings
        merged_valid = self.is_valid and other.is_valid

        return ValidationResult(
            is_valid=merged_valid, errors=merged_errors, warnings=merged_warnings
        )

    @classmethod
    def valid(cls) -> "ValidationResult":
        """Create a valid result with no errors or warnings."""
        return cls(is_valid=True)

    @classmethod
    def invalid(cls, errors: list = None) -> "ValidationResult":
        """Create an invalid result with errors."""
        return cls(is_valid=False, errors=tuple(errors or []))

    @classmethod
    def with_warnings(cls, warnings: list = None) -> "ValidationResult":
        """Create a valid result with warnings."""
        return cls(is_valid=True, warnings=tuple(warnings or []))

    def __str__(self) -> str:
        """String representation."""
        if self.is_valid and not self.has_warnings:
            return "Valid"

        parts = []
        if not self.is_valid:
            parts.append(f"Invalid ({self.error_count} errors)")
        else:
            parts.append("Valid")

        if self.has_warnings:
            parts.append(f"{self.warning_count} warnings")

        return " - ".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }
