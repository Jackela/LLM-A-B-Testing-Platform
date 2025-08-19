"""Model performance entity for comprehensive analysis."""

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ...evaluation.entities.evaluation_result import EvaluationResult
from ..exceptions import ModelComparisonError, ValidationError
from ..value_objects.cost_data import CostData
from ..value_objects.performance_score import PerformanceScore
from ..value_objects.test_result import TestResult


@dataclass
class DimensionPerformance:
    """Performance metrics for a specific evaluation dimension."""

    dimension_name: str
    score: PerformanceScore
    percentile_rank: Optional[Decimal] = None
    strength_level: str = "average"  # weak, average, strong, excellent
    improvement_potential: Decimal = Decimal("0")

    def __post_init__(self):
        """Validate dimension performance."""
        if not self.dimension_name.strip():
            raise ValidationError("Dimension name cannot be empty")

        if self.percentile_rank is not None:
            if not (Decimal("0") <= self.percentile_rank <= Decimal("100")):
                raise ValidationError("Percentile rank must be between 0 and 100")

        if not (Decimal("0") <= self.improvement_potential <= Decimal("1")):
            raise ValidationError("Improvement potential must be between 0 and 1")


@dataclass
class CostEffectivenessAnalysis:
    """Cost-effectiveness analysis for model performance."""

    cost_per_quality_point: Decimal
    efficiency_rank: int
    is_pareto_optimal: bool
    cost_saving_potential: Decimal
    recommendation: str
    budget_utilization: Decimal = Decimal("0")

    def __post_init__(self):
        """Validate cost-effectiveness analysis."""
        if self.cost_per_quality_point < 0:
            raise ValidationError("Cost per quality point cannot be negative")

        if self.efficiency_rank < 1:
            raise ValidationError("Efficiency rank must be at least 1")

        if not (Decimal("0") <= self.cost_saving_potential <= Decimal("1")):
            raise ValidationError("Cost saving potential must be between 0 and 1")


@dataclass
class ModelPerformance:
    """Comprehensive model performance analysis entity."""

    performance_id: UUID
    model_id: str
    model_name: str
    overall_performance: PerformanceScore
    dimension_performances: Dict[str, DimensionPerformance]
    evaluation_results: List[EvaluationResult]

    # Cost analysis
    cost_data: Optional[CostData] = None
    cost_effectiveness: Optional[CostEffectivenessAnalysis] = None

    # Performance breakdown
    performance_by_category: Dict[str, PerformanceScore] = field(default_factory=dict)
    performance_by_difficulty: Dict[str, PerformanceScore] = field(default_factory=dict)

    # Quality indicators
    reliability_score: Decimal = Decimal("0")
    consistency_score: Decimal = Decimal("0")

    # Metadata
    analysis_timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _domain_events: List[object] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate model performance after creation."""
        if not self.performance_id:
            self.performance_id = uuid4()

        if not self.model_id.strip():
            raise ValidationError("Model ID cannot be empty")

        if not self.model_name.strip():
            raise ValidationError("Model name cannot be empty")

        if not self.evaluation_results:
            raise ValidationError("Evaluation results cannot be empty")

    @classmethod
    def create_from_evaluation_results(
        cls,
        model_id: str,
        model_name: str,
        evaluation_results: List[EvaluationResult],
        cost_data: Optional[CostData] = None,
    ) -> "ModelPerformance":
        """Factory method to create model performance from evaluation results."""

        if not evaluation_results:
            raise ValidationError("Evaluation results cannot be empty")

        # Calculate overall performance
        overall_performance = cls._calculate_overall_performance(evaluation_results)

        # Calculate dimension performances
        dimension_performances = cls._calculate_dimension_performances(evaluation_results)

        # Calculate performance breakdowns
        performance_by_category = cls._calculate_performance_by_category(evaluation_results)
        performance_by_difficulty = cls._calculate_performance_by_difficulty(evaluation_results)

        # Calculate quality indicators
        reliability_score = cls._calculate_reliability_score(evaluation_results)
        consistency_score = cls._calculate_consistency_score(evaluation_results)

        return cls(
            performance_id=uuid4(),
            model_id=model_id,
            model_name=model_name,
            overall_performance=overall_performance,
            dimension_performances=dimension_performances,
            evaluation_results=evaluation_results,
            cost_data=cost_data,
            performance_by_category=performance_by_category,
            performance_by_difficulty=performance_by_difficulty,
            reliability_score=reliability_score,
            consistency_score=consistency_score,
        )

    def compare_with(self, other: "ModelPerformance") -> "ModelComparison":
        """Compare this model's performance with another model."""

        if not isinstance(other, ModelPerformance):
            raise ModelComparisonError("Can only compare with another ModelPerformance instance")

        # Calculate overall score difference
        score_difference = self.overall_performance.score - other.overall_performance.score

        # Calculate cost difference
        cost_difference = Decimal("0")
        if self.cost_data and other.cost_data:
            cost_difference = (
                self.cost_data.cost_per_request.amount - other.cost_data.cost_per_request.amount
            )

        # Perform statistical test for significance
        from ..services.significance_tester import SignificanceTester

        tester = SignificanceTester()
        statistical_test_result = tester.test_model_performance_difference(
            self.evaluation_results, other.evaluation_results
        )

        # Generate comparison recommendation
        recommendation = self._generate_comparison_recommendation(
            score_difference, cost_difference, statistical_test_result
        )

        return ModelComparison(
            model_a=self,
            model_b=other,
            score_difference=score_difference,
            cost_difference=cost_difference,
            statistical_test_result=statistical_test_result,
            dimension_comparisons=self._compare_dimensions(other),
            recommendation=recommendation,
        )

    def calculate_improvement_opportunities(self) -> Dict[str, Decimal]:
        """Calculate improvement opportunities for each dimension."""

        opportunities = {}

        for dimension_name, dim_perf in self.dimension_performances.items():
            # Opportunity is inversely related to current score
            current_score = dim_perf.score.score
            max_improvement = Decimal("1") - current_score

            # Weight by sample size confidence
            confidence_weight = dim_perf.score.confidence

            opportunity = max_improvement * confidence_weight
            opportunities[dimension_name] = opportunity.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        return opportunities

    def get_strengths(self, threshold: Decimal = Decimal("0.75")) -> List[str]:
        """Get model strengths (dimensions performing above threshold)."""

        strengths = []

        for dimension_name, dim_perf in self.dimension_performances.items():
            if dim_perf.score.score >= threshold:
                strengths.append(dimension_name)

        return strengths

    def get_weaknesses(self, threshold: Decimal = Decimal("0.5")) -> List[str]:
        """Get model weaknesses (dimensions performing below threshold)."""

        weaknesses = []

        for dimension_name, dim_perf in self.dimension_performances.items():
            if dim_perf.score.score <= threshold:
                weaknesses.append(dimension_name)

        return weaknesses

    def is_reliable(self, threshold: Decimal = Decimal("0.7")) -> bool:
        """Check if model performance is reliable."""
        return self.reliability_score >= threshold

    def is_consistent(self, threshold: Decimal = Decimal("0.7")) -> bool:
        """Check if model performance is consistent."""
        return self.consistency_score >= threshold

    def is_cost_effective(
        self,
        max_cost_per_request: Optional[Decimal] = None,
        min_performance_threshold: Decimal = Decimal("0.6"),
    ) -> bool:
        """Check if model is cost-effective."""

        # Performance threshold check
        if self.overall_performance.score < min_performance_threshold:
            return False

        # Cost check if cost data available
        if self.cost_data and max_cost_per_request:
            return self.cost_data.cost_per_request.amount <= max_cost_per_request

        # If no cost constraints, consider high-performing models cost-effective
        return self.overall_performance.score >= Decimal("0.8")

    @classmethod
    def _calculate_overall_performance(
        cls, evaluation_results: List[EvaluationResult]
    ) -> PerformanceScore:
        """Calculate overall performance score."""

        if not evaluation_results:
            return PerformanceScore(score=Decimal("0"), confidence=Decimal("0"), sample_size=0)

        # Weight scores by confidence
        weighted_sum = Decimal("0")
        total_confidence = Decimal("0")

        for result in evaluation_results:
            score = result.overall_score
            confidence = result.confidence_score

            weighted_sum += score * confidence
            total_confidence += confidence

        if total_confidence == 0:
            # Fall back to simple average
            scores = [r.overall_score for r in evaluation_results]
            avg_score = sum(scores) / len(scores)
            return PerformanceScore(
                score=avg_score, confidence=Decimal("0.5"), sample_size=len(evaluation_results)
            )

        weighted_average = weighted_sum / total_confidence
        avg_confidence = total_confidence / len(evaluation_results)

        # Calculate standard error
        scores = [float(r.overall_score) for r in evaluation_results]
        if len(scores) > 1:
            import statistics

            std_error = Decimal(str(statistics.stdev(scores) / (len(scores) ** 0.5)))
        else:
            std_error = Decimal("0")

        return PerformanceScore(
            score=weighted_average.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            confidence=avg_confidence.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            sample_size=len(evaluation_results),
            standard_error=std_error,
        )

    @classmethod
    def _calculate_dimension_performances(
        cls, evaluation_results: List[EvaluationResult]
    ) -> Dict[str, DimensionPerformance]:
        """Calculate performance for each dimension."""

        import statistics
        from collections import defaultdict

        dimension_scores = defaultdict(list)
        dimension_confidences = defaultdict(list)

        # Collect scores for each dimension
        for result in evaluation_results:
            for dimension, score in result.dimension_scores.items():
                dimension_scores[dimension].append(score)
                dimension_confidences[dimension].append(float(result.confidence_score))

        dimension_performances = {}

        for dimension, scores in dimension_scores.items():
            if not scores:
                continue

            # Calculate weighted average
            confidences = dimension_confidences[dimension]
            weighted_sum = sum(s * c for s, c in zip(scores, confidences))
            total_confidence = sum(confidences)

            if total_confidence == 0:
                avg_score = statistics.mean(scores)
                avg_confidence = 0.5
            else:
                avg_score = weighted_sum / total_confidence
                avg_confidence = total_confidence / len(confidences)

            # Normalize score to 0-1 range (assuming scores are 1-5)
            normalized_score = (avg_score - 1) / 4  # Convert 1-5 to 0-1
            normalized_score = max(0.0, min(1.0, normalized_score))

            # Calculate standard error
            std_error = statistics.stdev(scores) / (len(scores) ** 0.5) if len(scores) > 1 else 0.0
            normalized_std_error = std_error / 4  # Normalize to 0-1 scale

            performance_score = PerformanceScore(
                score=Decimal(str(normalized_score)).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ),
                confidence=Decimal(str(avg_confidence)).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ),
                sample_size=len(scores),
                standard_error=Decimal(str(normalized_std_error)).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ),
            )

            # Determine strength level
            strength_level = cls._determine_strength_level(normalized_score)

            dimension_performances[dimension] = DimensionPerformance(
                dimension_name=dimension, score=performance_score, strength_level=strength_level
            )

        return dimension_performances

    @classmethod
    def _calculate_performance_by_category(
        cls, evaluation_results: List[EvaluationResult]
    ) -> Dict[str, PerformanceScore]:
        """Calculate performance breakdown by category."""

        import statistics
        from collections import defaultdict

        category_results = defaultdict(list)

        for result in evaluation_results:
            category = result.metadata.get("category", "unknown")
            category_results[category].append(result)

        category_performances = {}

        for category, results in category_results.items():
            if results:
                performance = cls._calculate_overall_performance(results)
                category_performances[category] = performance

        return category_performances

    @classmethod
    def _calculate_performance_by_difficulty(
        cls, evaluation_results: List[EvaluationResult]
    ) -> Dict[str, PerformanceScore]:
        """Calculate performance breakdown by difficulty level."""

        from collections import defaultdict

        difficulty_results = defaultdict(list)

        for result in evaluation_results:
            difficulty = result.metadata.get("difficulty_level", "unknown")
            difficulty_results[difficulty].append(result)

        difficulty_performances = {}

        for difficulty, results in difficulty_results.items():
            if results:
                performance = cls._calculate_overall_performance(results)
                difficulty_performances[difficulty] = performance

        return difficulty_performances

    @classmethod
    def _calculate_reliability_score(cls, evaluation_results: List[EvaluationResult]) -> Decimal:
        """Calculate reliability score based on confidence and quality."""

        if not evaluation_results:
            return Decimal("0")

        # Reliability based on average confidence and quality pass rate
        confidences = [float(r.confidence_score) for r in evaluation_results]
        avg_confidence = sum(confidences) / len(confidences)

        # Quality pass rate
        quality_passing = len([r for r in evaluation_results if not r.has_quality_issues()])
        quality_rate = quality_passing / len(evaluation_results)

        # Combine confidence and quality (weighted average)
        reliability = avg_confidence * 0.7 + quality_rate * 0.3

        return Decimal(str(reliability)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    @classmethod
    def _calculate_consistency_score(cls, evaluation_results: List[EvaluationResult]) -> Decimal:
        """Calculate consistency score based on score variance."""

        if len(evaluation_results) < 2:
            return Decimal("1")  # Perfect consistency with single result

        scores = [float(r.overall_score) for r in evaluation_results]

        import statistics

        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores)

        # Consistency is inversely related to coefficient of variation
        if mean_score == 0:
            return Decimal("0")

        cv = std_dev / mean_score  # Coefficient of variation
        consistency = max(0.0, 1.0 - cv)  # Higher CV = lower consistency

        return Decimal(str(consistency)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    @classmethod
    def _determine_strength_level(cls, score: float) -> str:
        """Determine strength level based on score."""

        if score >= 0.9:
            return "excellent"
        elif score >= 0.75:
            return "strong"
        elif score >= 0.5:
            return "average"
        else:
            return "weak"

    def _compare_dimensions(self, other: "ModelPerformance") -> Dict[str, Dict[str, Any]]:
        """Compare dimensions with another model."""

        comparisons = {}

        # Get all dimensions from both models
        all_dimensions = set(self.dimension_performances.keys()) | set(
            other.dimension_performances.keys()
        )

        for dimension in all_dimensions:
            self_perf = self.dimension_performances.get(dimension)
            other_perf = other.dimension_performances.get(dimension)

            if self_perf and other_perf:
                score_diff = self_perf.score.score - other_perf.score.score
                comparisons[dimension] = {
                    "score_difference": float(score_diff),
                    "self_score": float(self_perf.score.score),
                    "other_score": float(other_perf.score.score),
                    "advantage": (
                        "self" if score_diff > 0 else "other" if score_diff < 0 else "equal"
                    ),
                }
            elif self_perf:
                comparisons[dimension] = {
                    "score_difference": float(self_perf.score.score),
                    "self_score": float(self_perf.score.score),
                    "other_score": 0.0,
                    "advantage": "self",
                }
            elif other_perf:
                comparisons[dimension] = {
                    "score_difference": -float(other_perf.score.score),
                    "self_score": 0.0,
                    "other_score": float(other_perf.score.score),
                    "advantage": "other",
                }

        return comparisons

    def _generate_comparison_recommendation(
        self,
        score_difference: Decimal,
        cost_difference: Decimal,
        statistical_test_result: TestResult,
    ) -> str:
        """Generate recommendation based on comparison results."""

        is_significant = statistical_test_result.is_significant()
        has_practical_significance = statistical_test_result.has_practical_significance()

        if is_significant and has_practical_significance:
            if score_difference > 0:
                if cost_difference <= 0:
                    return (
                        f"Strong recommendation for {self.model_name}: "
                        "Significantly better performance with equal or lower cost."
                    )
                else:
                    return (
                        f"Conditional recommendation for {self.model_name}: "
                        "Better performance but higher cost. Evaluate cost-benefit trade-offs."
                    )
            else:
                return (
                    f"Recommendation against {self.model_name}: " "Significantly worse performance."
                )
        elif is_significant:
            return (
                "Statistically significant difference detected but effect size is small. "
                "Consider practical importance and costs."
            )
        else:
            return (
                "No significant performance difference detected. "
                "Choose based on cost, reliability, or other factors."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""

        return {
            "performance_id": str(self.performance_id),
            "model_id": self.model_id,
            "model_name": self.model_name,
            "overall_performance": {
                "score": str(self.overall_performance.score),
                "confidence": str(self.overall_performance.confidence),
                "sample_size": self.overall_performance.sample_size,
                "standard_error": (
                    str(self.overall_performance.standard_error)
                    if self.overall_performance.standard_error
                    else None
                ),
            },
            "dimension_performances": {
                dim: {
                    "score": str(perf.score.score),
                    "confidence": str(perf.score.confidence),
                    "sample_size": perf.score.sample_size,
                    "strength_level": perf.strength_level,
                    "percentile_rank": str(perf.percentile_rank) if perf.percentile_rank else None,
                }
                for dim, perf in self.dimension_performances.items()
            },
            "performance_by_category": {
                cat: {
                    "score": str(perf.score),
                    "confidence": str(perf.confidence),
                    "sample_size": perf.sample_size,
                }
                for cat, perf in self.performance_by_category.items()
            },
            "performance_by_difficulty": {
                diff: {
                    "score": str(perf.score),
                    "confidence": str(perf.confidence),
                    "sample_size": perf.sample_size,
                }
                for diff, perf in self.performance_by_difficulty.items()
            },
            "cost_data": self.cost_data.to_dict() if self.cost_data else None,
            "cost_effectiveness": (
                {
                    "cost_per_quality_point": str(self.cost_effectiveness.cost_per_quality_point),
                    "efficiency_rank": self.cost_effectiveness.efficiency_rank,
                    "is_pareto_optimal": self.cost_effectiveness.is_pareto_optimal,
                    "recommendation": self.cost_effectiveness.recommendation,
                }
                if self.cost_effectiveness
                else None
            ),
            "reliability_score": str(self.reliability_score),
            "consistency_score": str(self.consistency_score),
            "strengths": self.get_strengths(),
            "weaknesses": self.get_weaknesses(),
            "improvement_opportunities": {
                dim: str(opp) for dim, opp in self.calculate_improvement_opportunities().items()
            },
            "is_reliable": self.is_reliable(),
            "is_consistent": self.is_consistent(),
            "is_cost_effective": self.is_cost_effective(),
            "analysis_timestamp": self.analysis_timestamp,
            "metadata": self.metadata.copy(),
        }


@dataclass(frozen=True)
class ModelComparison:
    """Model comparison result value object."""

    model_a: ModelPerformance
    model_b: ModelPerformance
    score_difference: Decimal
    cost_difference: Decimal
    statistical_test_result: TestResult
    dimension_comparisons: Dict[str, Dict[str, Any]]
    recommendation: str

    def get_winner(self) -> Optional[str]:
        """Get the winning model based on statistical significance."""

        if self.statistical_test_result.is_significant():
            if self.score_difference > 0:
                return self.model_a.model_id
            elif self.score_difference < 0:
                return self.model_b.model_id

        return None  # No clear winner

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""

        return {
            "model_a": {
                "id": self.model_a.model_id,
                "name": self.model_a.model_name,
                "score": str(self.model_a.overall_performance.score),
            },
            "model_b": {
                "id": self.model_b.model_id,
                "name": self.model_b.model_name,
                "score": str(self.model_b.overall_performance.score),
            },
            "score_difference": str(self.score_difference),
            "cost_difference": str(self.cost_difference),
            "statistical_test": self.statistical_test_result.to_dict(),
            "dimension_comparisons": self.dimension_comparisons,
            "recommendation": self.recommendation,
            "winner": self.get_winner(),
            "is_significant": self.statistical_test_result.is_significant(),
            "has_practical_significance": self.statistical_test_result.has_practical_significance(),
        }
