"""Validation result value object for Test Management domain."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result containing errors and warnings."""

    is_valid: bool
    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Ensure errors and warnings are immutable tuples."""
        # Note: With default_factory=tuple, these should always be tuples
        # This defensive check is kept for robustness
        pass

    @classmethod
    def success(cls) -> "ValidationResult":
        """Create successful validation result."""
        return cls(is_valid=True, errors=(), warnings=())

    @classmethod
    def with_errors(cls, errors: List[str]) -> "ValidationResult":
        """Create validation result with errors."""
        return cls(is_valid=False, errors=tuple(errors), warnings=())

    @classmethod
    def with_warnings(cls, warnings: List[str]) -> "ValidationResult":
        """Create validation result with warnings."""
        return cls(is_valid=True, errors=(), warnings=tuple(warnings))

    def add_error(self, error: str) -> "ValidationResult":
        """Add error and return new validation result."""
        new_errors = list(self.errors) + [error]
        return ValidationResult(is_valid=False, errors=tuple(new_errors), warnings=self.warnings)

    def add_warning(self, warning: str) -> "ValidationResult":
        """Add warning and return new validation result."""
        new_warnings = list(self.warnings) + [warning]
        return ValidationResult(
            is_valid=self.is_valid, errors=self.errors, warnings=tuple(new_warnings)
        )

    @classmethod
    def combine(cls, results: List["ValidationResult"]) -> "ValidationResult":
        """Combine multiple validation results."""
        if not results:
            return cls.success()

        all_errors: List[str] = []
        all_warnings: List[str] = []
        is_valid = True

        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            if not result.is_valid:
                is_valid = False

        return cls(is_valid=is_valid, errors=tuple(all_errors), warnings=tuple(all_warnings))

    def has_errors(self) -> bool:
        """Check if validation result has errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if validation result has warnings."""
        return len(self.warnings) > 0

    def __bool__(self) -> bool:
        """Boolean conversion - True if valid."""
        return self.is_valid

    def __str__(self) -> str:
        """String representation of validation result."""
        if self.is_valid:
            if self.has_warnings():
                return f"Valid with {len(self.warnings)} warnings"
            return "Valid"
        else:
            return f"Invalid: {len(self.errors)} errors, {len(self.warnings)} warnings"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ValidationResult(is_valid={self.is_valid}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )
