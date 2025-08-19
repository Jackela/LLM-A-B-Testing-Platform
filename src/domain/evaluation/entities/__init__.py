"""Entities for evaluation domain."""

from .dimension import Dimension
from .evaluation_result import EvaluationResult
from .evaluation_template import EvaluationTemplate
from .judge import Judge

__all__ = [
    "Dimension",
    "EvaluationTemplate",
    "EvaluationResult",
    "Judge",
]
