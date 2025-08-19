"""Exceptions for evaluation domain."""


class EvaluationDomainError(Exception):
    """Base exception for evaluation domain."""

    pass


class JudgeNotCalibratedError(EvaluationDomainError):
    """Raised when judge is not properly calibrated for production use."""

    pass


class ValidationError(EvaluationDomainError):
    """Raised when validation fails."""

    pass


class TemplateRenderError(EvaluationDomainError):
    """Raised when template rendering fails."""

    pass


class MissingDimensionScore(EvaluationDomainError):
    """Raised when dimension score is missing from evaluation."""

    pass


class InvalidScore(EvaluationDomainError):
    """Raised when score is outside valid range."""

    pass


class ConsensusCalculationError(EvaluationDomainError):
    """Raised when consensus calculation fails."""

    pass


class InsufficientDataError(EvaluationDomainError):
    """Raised when insufficient data for statistical calculations."""

    pass


class QualityControlError(EvaluationDomainError):
    """Raised when quality control checks fail."""

    pass


class CalibrationError(EvaluationDomainError):
    """Raised when judge calibration fails."""

    pass


class BusinessRuleViolation(EvaluationDomainError):
    """Raised when business rule is violated."""

    pass
