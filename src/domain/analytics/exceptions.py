"""Analytics domain exceptions."""

from typing import Optional


class AnalyticsError(Exception):
    """Base exception for analytics domain."""

    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ValidationError(AnalyticsError):
    """Raised when validation fails."""

    pass


class StatisticalError(AnalyticsError):
    """Base exception for statistical analysis errors."""

    pass


class InsufficientDataError(StatisticalError):
    """Raised when there's insufficient data for statistical analysis."""

    pass


class InvalidTestTypeError(StatisticalError):
    """Raised when an unsupported statistical test type is requested."""

    pass


class InvalidDataError(StatisticalError):
    """Raised when input data is invalid for statistical analysis."""

    pass


class NumericalInstabilityError(StatisticalError):
    """Raised when numerical calculations become unstable."""

    pass


class MultipleComparisonsError(StatisticalError):
    """Raised when multiple comparisons correction fails."""

    pass


class AggregationError(AnalyticsError):
    """Base exception for data aggregation errors."""

    pass


class InvalidAggregationRule(AggregationError):
    """Raised when aggregation rule is invalid."""

    pass


class MissingDataError(AggregationError):
    """Raised when required data is missing for aggregation."""

    pass


class InsightGenerationError(AnalyticsError):
    """Raised when insight generation fails."""

    pass


class ModelComparisonError(AnalyticsError):
    """Raised when model comparison fails."""

    pass


class CostAnalysisError(AnalyticsError):
    """Raised when cost analysis fails."""

    pass


class ReportGenerationError(AnalyticsError):
    """Raised when report generation fails."""

    pass


class VisualizationError(AnalyticsError):
    """Raised when visualization generation fails."""

    pass


class CalculationError(AnalyticsError):
    """Raised when metric calculation fails."""

    pass
