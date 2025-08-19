"""Insight generation service for analytics domain."""

from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..entities.analysis_result import AnalysisResult, ModelPerformanceMetrics
from ..entities.model_performance import ModelComparison, ModelPerformance
from ..exceptions import InsightGenerationError, ValidationError
from ..value_objects.cost_data import CostData
from ..value_objects.insight import Insight, InsightSeverity, InsightType
from ..value_objects.test_result import TestResult


class InsightGenerator:
    """Domain service for automated insight generation."""

    def __init__(self):
        """Initialize insight generator."""
        self._confidence_thresholds = {
            "high": Decimal("0.8"),
            "medium": Decimal("0.6"),
            "low": Decimal("0.4"),
        }
        self._effect_size_thresholds = {"large": 0.8, "medium": 0.5, "small": 0.2}

    def generate_performance_insights(self, analysis_result: AnalysisResult) -> List[Insight]:
        """Generate performance-related insights from analysis results."""

        if not analysis_result.model_performances:
            raise InsightGenerationError("No model performances available for insight generation")

        insights = []

        # Generate insights for overall performance patterns
        insights.extend(self._generate_overall_performance_insights(analysis_result))

        # Generate insights for statistical test results
        insights.extend(self._generate_statistical_insights(analysis_result))

        # Generate insights for dimension-specific patterns
        insights.extend(self._generate_dimension_insights(analysis_result))

        # Generate insights for performance variability
        insights.extend(self._generate_variability_insights(analysis_result))

        # Generate cost-effectiveness insights if cost data available
        insights.extend(self._generate_cost_insights(analysis_result))

        return insights

    def generate_model_comparison_insights(
        self, model_comparison: ModelComparison
    ) -> List[Insight]:
        """Generate insights from model comparison results."""

        insights = []

        # Statistical significance insight
        if model_comparison.statistical_test_result.is_significant():
            insights.append(self._create_significance_insight(model_comparison))

        # Effect size insight
        insights.append(self._create_effect_size_insight(model_comparison))

        # Cost-benefit insight
        if model_comparison.model_a.cost_data and model_comparison.model_b.cost_data:
            insights.append(self._create_cost_benefit_insight(model_comparison))

        # Dimension-specific insights
        insights.extend(self._create_dimension_comparison_insights(model_comparison))

        return insights

    def generate_quality_insights(
        self, model_performances: List[ModelPerformance]
    ) -> List[Insight]:
        """Generate quality-related insights."""

        if not model_performances:
            return []

        insights = []

        # Reliability insights
        insights.extend(self._generate_reliability_insights(model_performances))

        # Consistency insights
        insights.extend(self._generate_consistency_insights(model_performances))

        # Quality pattern insights
        insights.extend(self._generate_quality_pattern_insights(model_performances))

        return insights

    def generate_bias_detection_insights(self, analysis_result: AnalysisResult) -> List[Insight]:
        """Generate bias detection insights."""

        insights = []

        # Performance bias by category
        insights.extend(self._detect_category_bias(analysis_result))

        # Performance bias by difficulty
        insights.extend(self._detect_difficulty_bias(analysis_result))

        # Judge bias detection
        insights.extend(self._detect_judge_bias(analysis_result))

        return insights

    def generate_recommendations(
        self, analysis_result: AnalysisResult, business_context: Optional[Dict[str, Any]] = None
    ) -> List[Insight]:
        """Generate actionable recommendations."""

        insights = []

        # Model selection recommendations
        insights.extend(self._generate_model_selection_recommendations(analysis_result))

        # Performance improvement recommendations
        insights.extend(self._generate_improvement_recommendations(analysis_result))

        # Sample size recommendations
        insights.extend(self._generate_sample_size_recommendations(analysis_result))

        # Business context recommendations
        if business_context:
            insights.extend(
                self._generate_business_context_recommendations(analysis_result, business_context)
            )

        return insights

    def _generate_overall_performance_insights(
        self, analysis_result: AnalysisResult
    ) -> List[Insight]:
        """Generate insights about overall performance patterns."""

        insights = []
        performances = list(analysis_result.model_performances.values())

        if len(performances) < 2:
            return insights

        # Sort by performance score
        performances.sort(key=lambda p: p.overall_score, reverse=True)

        best_model = performances[0]
        worst_model = performances[-1]

        # Performance spread insight
        performance_spread = best_model.overall_score - worst_model.overall_score

        if performance_spread > Decimal("0.2"):
            insights.append(
                Insight(
                    insight_id=uuid4(),
                    insight_type=InsightType.PERFORMANCE,
                    severity=InsightSeverity.HIGH,
                    title="Significant Performance Variation Detected",
                    description=(
                        f"Large performance gap between best model ({best_model.model_name}: "
                        f"{best_model.overall_score:.3f}) and worst model "
                        f"({worst_model.model_name}: {worst_model.overall_score:.3f}). "
                        f"Spread: {performance_spread:.3f}"
                    ),
                    confidence_score=Decimal("0.9"),
                    evidence={
                        "performance_spread": str(performance_spread),
                        "best_model": best_model.model_name,
                        "worst_model": worst_model.model_name,
                        "best_score": str(best_model.overall_score),
                        "worst_score": str(worst_model.overall_score),
                    },
                    recommendations=[
                        f"Consider focusing on {best_model.model_name} for production use",
                        f"Investigate why {worst_model.model_name} underperforms",
                        "Analyze dimension-level differences to understand performance gaps",
                    ],
                    affected_models=[best_model.model_id, worst_model.model_id],
                )
            )

        # High performer insight
        high_performers = [p for p in performances if p.overall_score >= Decimal("0.8")]
        if high_performers:
            insights.append(
                Insight(
                    insight_id=uuid4(),
                    insight_type=InsightType.PERFORMANCE,
                    severity=InsightSeverity.INFO,
                    title=f"{len(high_performers)} High-Performing Model(s) Identified",
                    description=(
                        f"Found {len(high_performers)} model(s) with excellent performance "
                        f"(≥0.8 score): {', '.join([p.model_name for p in high_performers])}"
                    ),
                    confidence_score=Decimal("0.85"),
                    evidence={
                        "high_performer_count": len(high_performers),
                        "high_performers": [p.model_name for p in high_performers],
                        "scores": [str(p.overall_score) for p in high_performers],
                    },
                    recommendations=[
                        "Consider these models for production deployment",
                        "Analyze what makes these models successful",
                        "Use these models as benchmarks for others",
                    ],
                    affected_models=[p.model_id for p in high_performers],
                )
            )

        return insights

    def _generate_statistical_insights(self, analysis_result: AnalysisResult) -> List[Insight]:
        """Generate insights from statistical test results."""

        insights = []

        significant_tests = analysis_result.get_significant_tests()

        if significant_tests:
            insights.append(
                Insight(
                    insight_id=uuid4(),
                    insight_type=InsightType.STATISTICAL_SIGNIFICANCE,
                    severity=InsightSeverity.MEDIUM,
                    title=f"{len(significant_tests)} Statistically Significant Differences Found",
                    description=(
                        f"Found {len(significant_tests)} statistically significant differences "
                        f"out of {len(analysis_result.statistical_tests)} total tests. "
                        f"This indicates meaningful performance variations between models."
                    ),
                    confidence_score=Decimal("0.9"),
                    evidence={
                        "significant_tests": len(significant_tests),
                        "total_tests": len(analysis_result.statistical_tests),
                        "significance_rate": len(significant_tests)
                        / len(analysis_result.statistical_tests),
                        "test_names": list(significant_tests.keys()),
                    },
                    recommendations=[
                        "Focus on models showing significant advantages",
                        "Investigate causes of significant differences",
                        "Consider practical significance alongside statistical significance",
                    ],
                )
            )

        # Large effect size insights
        large_effects = [
            name
            for name, result in analysis_result.statistical_tests.items()
            if abs(result.effect_size) >= self._effect_size_thresholds["large"]
        ]

        if large_effects:
            insights.append(
                Insight(
                    insight_id=uuid4(),
                    insight_type=InsightType.STATISTICAL_SIGNIFICANCE,
                    severity=InsightSeverity.HIGH,
                    title=f"{len(large_effects)} Large Effect Size(s) Detected",
                    description=(
                        f"Found {len(large_effects)} comparison(s) with large effect sizes, "
                        "indicating not just statistical but also practical significance."
                    ),
                    confidence_score=Decimal("0.85"),
                    evidence={
                        "large_effect_tests": large_effects,
                        "effect_sizes": {
                            name: analysis_result.statistical_tests[name].effect_size
                            for name in large_effects
                        },
                    },
                    recommendations=[
                        "Prioritize decisions based on these comparisons",
                        "These differences are likely to be meaningful in practice",
                        "Consider implementing the better-performing options",
                    ],
                )
            )

        return insights

    def _generate_dimension_insights(self, analysis_result: AnalysisResult) -> List[Insight]:
        """Generate dimension-specific performance insights."""

        insights = []

        if not analysis_result.model_performances:
            return insights

        # Collect all dimensions
        all_dimensions = set()
        for perf in analysis_result.model_performances.values():
            all_dimensions.update(perf.dimension_scores.keys())

        for dimension in all_dimensions:
            dimension_scores = {}

            for model_id, perf in analysis_result.model_performances.items():
                if dimension in perf.dimension_scores:
                    dimension_scores[model_id] = perf.dimension_scores[dimension]

            if len(dimension_scores) >= 2:
                # Find best and worst performers for this dimension
                best_model = max(dimension_scores.items(), key=lambda x: x[1])
                worst_model = min(dimension_scores.items(), key=lambda x: x[1])

                score_gap = best_model[1] - worst_model[1]

                if score_gap > Decimal("0.3"):  # Significant gap
                    model_names = {
                        model_id: perf.model_name
                        for model_id, perf in analysis_result.model_performances.items()
                    }

                    insights.append(
                        Insight(
                            insight_id=uuid4(),
                            insight_type=InsightType.QUALITY_PATTERN,
                            severity=InsightSeverity.MEDIUM,
                            title=f"Large Performance Gap in {dimension.replace('_', ' ').title()}",
                            description=(
                                f"Significant variation in {dimension} performance: "
                                f"{model_names[best_model[0]]} ({best_model[1]:.3f}) vs "
                                f"{model_names[worst_model[0]]} ({worst_model[1]:.3f}). "
                                f"Gap: {score_gap:.3f}"
                            ),
                            confidence_score=Decimal("0.8"),
                            evidence={
                                "dimension": dimension,
                                "best_model": model_names[best_model[0]],
                                "worst_model": model_names[worst_model[0]],
                                "best_score": str(best_model[1]),
                                "worst_score": str(worst_model[1]),
                                "score_gap": str(score_gap),
                            },
                            recommendations=[
                                f"Investigate why {model_names[best_model[0]]} excels in {dimension}",
                                f"Consider improvements to {model_names[worst_model[0]]} for {dimension}",
                                f"Use {dimension} performance as a key differentiator",
                            ],
                            affected_models=[best_model[0], worst_model[0]],
                            category=dimension,
                        )
                    )

        return insights

    def _generate_variability_insights(self, analysis_result: AnalysisResult) -> List[Insight]:
        """Generate insights about performance variability and consistency."""

        insights = []

        for model_id, perf_metrics in analysis_result.model_performances.items():
            # Check consistency score
            if perf_metrics.consistency_score < Decimal("0.6"):
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.QUALITY_PATTERN,
                        severity=InsightSeverity.MEDIUM,
                        title=f"Low Consistency Detected: {perf_metrics.model_name}",
                        description=(
                            f"{perf_metrics.model_name} shows low consistency "
                            f"(score: {perf_metrics.consistency_score:.3f}). "
                            "This indicates variable performance across different test cases."
                        ),
                        confidence_score=Decimal("0.75"),
                        evidence={
                            "consistency_score": str(perf_metrics.consistency_score),
                            "model_name": perf_metrics.model_name,
                            "sample_size": perf_metrics.sample_count,
                        },
                        recommendations=[
                            "Investigate causes of performance variability",
                            "Consider additional training or fine-tuning",
                            "Monitor consistency in production usage",
                            "May need larger sample size to confirm pattern",
                        ],
                        affected_models=[model_id],
                    )
                )

            # Check reliability score
            if perf_metrics.reliability_score < Decimal("0.7"):
                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.QUALITY_PATTERN,
                        severity=InsightSeverity.HIGH,
                        title=f"Reliability Concerns: {perf_metrics.model_name}",
                        description=(
                            f"{perf_metrics.model_name} shows reliability concerns "
                            f"(score: {perf_metrics.reliability_score:.3f}). "
                            "This may indicate quality issues or low confidence evaluations."
                        ),
                        confidence_score=Decimal("0.8"),
                        evidence={
                            "reliability_score": str(perf_metrics.reliability_score),
                            "model_name": perf_metrics.model_name,
                            "sample_size": perf_metrics.sample_count,
                        },
                        recommendations=[
                            "Review evaluation quality and judge confidence",
                            "Consider additional quality controls",
                            "May require more rigorous testing before production use",
                            "Investigate evaluation methodology",
                        ],
                        affected_models=[model_id],
                    )
                )

        return insights

    def _generate_cost_insights(self, analysis_result: AnalysisResult) -> List[Insight]:
        """Generate cost-effectiveness insights."""

        insights = []

        # Find models with cost data
        models_with_costs = [
            perf
            for perf in analysis_result.model_performances.values()
            if perf.cost_data is not None
        ]

        if len(models_with_costs) < 2:
            return insights

        # Calculate cost-effectiveness scores
        cost_effectiveness_scores = []
        for perf in models_with_costs:
            if perf.cost_data.cost_per_request.amount > 0:
                ce_score = perf.overall_score / perf.cost_data.cost_per_request.amount
                cost_effectiveness_scores.append((perf, ce_score))

        if not cost_effectiveness_scores:
            return insights

        # Sort by cost-effectiveness
        cost_effectiveness_scores.sort(key=lambda x: x[1], reverse=True)

        most_efficient = cost_effectiveness_scores[0]
        least_efficient = cost_effectiveness_scores[-1]

        efficiency_ratio = (
            most_efficient[1] / least_efficient[1] if least_efficient[1] > 0 else float("inf")
        )

        if efficiency_ratio > 2.0:  # Significant difference
            insights.append(
                Insight(
                    insight_id=uuid4(),
                    insight_type=InsightType.COST_EFFECTIVENESS,
                    severity=InsightSeverity.HIGH,
                    title="Significant Cost-Effectiveness Difference Found",
                    description=(
                        f"{most_efficient[0].model_name} is {efficiency_ratio:.1f}x more "
                        f"cost-effective than {least_efficient[0].model_name}. "
                        f"Consider cost implications in model selection."
                    ),
                    confidence_score=Decimal("0.85"),
                    evidence={
                        "most_efficient_model": most_efficient[0].model_name,
                        "least_efficient_model": least_efficient[0].model_name,
                        "efficiency_ratio": efficiency_ratio,
                        "most_efficient_score": str(most_efficient[1]),
                        "least_efficient_score": str(least_efficient[1]),
                    },
                    recommendations=[
                        f"Consider {most_efficient[0].model_name} for cost-sensitive applications",
                        "Evaluate if performance difference justifies cost difference",
                        "Consider budget constraints in model selection",
                        "Monitor costs in production deployment",
                    ],
                    affected_models=[most_efficient[0].model_id, least_efficient[0].model_id],
                )
            )

        return insights

    def _create_significance_insight(self, model_comparison: ModelComparison) -> Insight:
        """Create insight for statistical significance."""

        winner = model_comparison.get_winner()
        winner_model = (
            model_comparison.model_a
            if winner == model_comparison.model_a.model_id
            else model_comparison.model_b
        )

        return Insight(
            insight_id=uuid4(),
            insight_type=InsightType.STATISTICAL_SIGNIFICANCE,
            severity=InsightSeverity.MEDIUM,
            title="Statistically Significant Performance Difference",
            description=(
                f"Significant performance difference detected between "
                f"{model_comparison.model_a.model_name} and {model_comparison.model_b.model_name}. "
                f"{winner_model.model_name} performs significantly better "
                f"(p={model_comparison.statistical_test_result.p_value:.4f})."
            ),
            confidence_score=Decimal(str(1 - model_comparison.statistical_test_result.p_value)),
            evidence={
                "p_value": model_comparison.statistical_test_result.p_value,
                "effect_size": model_comparison.statistical_test_result.effect_size,
                "test_type": model_comparison.statistical_test_result.test_type,
                "winner": winner_model.model_name,
            },
            recommendations=[
                f"Prefer {winner_model.model_name} based on statistical evidence",
                "Verify results with additional testing if needed",
                "Consider practical significance alongside statistical significance",
            ],
            affected_models=[model_comparison.model_a.model_id, model_comparison.model_b.model_id],
        )

    def _create_effect_size_insight(self, model_comparison: ModelComparison) -> Insight:
        """Create insight for effect size."""

        effect_size = abs(model_comparison.statistical_test_result.effect_size)

        if effect_size >= self._effect_size_thresholds["large"]:
            severity = InsightSeverity.HIGH
            magnitude = "large"
        elif effect_size >= self._effect_size_thresholds["medium"]:
            severity = InsightSeverity.MEDIUM
            magnitude = "medium"
        elif effect_size >= self._effect_size_thresholds["small"]:
            severity = InsightSeverity.LOW
            magnitude = "small"
        else:
            severity = InsightSeverity.INFO
            magnitude = "negligible"

        return Insight(
            insight_id=uuid4(),
            insight_type=InsightType.STATISTICAL_SIGNIFICANCE,
            severity=severity,
            title=f"{magnitude.title()} Effect Size Detected",
            description=(
                f"Effect size analysis shows {magnitude} practical difference "
                f"between models (Cohen's d = {effect_size:.3f}). "
                f"This indicates {'substantial' if effect_size >= 0.5 else 'modest'} "
                "real-world impact."
            ),
            confidence_score=Decimal("0.8"),
            evidence={
                "effect_size": effect_size,
                "magnitude": magnitude,
                "cohens_d": model_comparison.statistical_test_result.effect_size,
            },
            recommendations=[
                f"Effect size suggests {'strong' if effect_size >= 0.8 else 'moderate'} practical significance",
                "Consider both statistical and practical significance in decisions",
                "Validate findings with domain experts",
            ],
            affected_models=[model_comparison.model_a.model_id, model_comparison.model_b.model_id],
        )

    def _create_cost_benefit_insight(self, model_comparison: ModelComparison) -> Insight:
        """Create cost-benefit analysis insight."""

        better_model = (
            model_comparison.model_a
            if model_comparison.score_difference > 0
            else model_comparison.model_b
        )

        worse_model = (
            model_comparison.model_b
            if model_comparison.score_difference > 0
            else model_comparison.model_a
        )

        performance_improvement = abs(model_comparison.score_difference)
        cost_increase = (
            model_comparison.cost_difference
            if model_comparison.score_difference > 0
            else -model_comparison.cost_difference
        )

        if cost_increase <= 0:
            recommendation = f"Clear choice: {better_model.model_name} offers better performance at equal or lower cost"
            severity = InsightSeverity.HIGH
        elif performance_improvement > Decimal(
            "0.1"
        ) and cost_increase / performance_improvement < Decimal("10"):
            recommendation = f"Good value: {better_model.model_name} offers worthwhile improvement for the cost increase"
            severity = InsightSeverity.MEDIUM
        else:
            recommendation = f"Cost concerns: {better_model.model_name} improvement may not justify cost increase"
            severity = InsightSeverity.LOW

        return Insight(
            insight_id=uuid4(),
            insight_type=InsightType.COST_EFFECTIVENESS,
            severity=severity,
            title="Cost-Benefit Analysis Results",
            description=(
                f"Performance improvement of {performance_improvement:.3f} "
                f"comes with cost change of {cost_increase:.4f} per request. "
                f"{recommendation}"
            ),
            confidence_score=Decimal("0.75"),
            evidence={
                "performance_improvement": str(performance_improvement),
                "cost_change": str(cost_increase),
                "better_model": better_model.model_name,
                "worse_model": worse_model.model_name,
            },
            recommendations=[
                recommendation,
                "Consider budget constraints and performance requirements",
                "Monitor actual costs in production use",
            ],
            affected_models=[better_model.model_id, worse_model.model_id],
        )

    def _create_dimension_comparison_insights(
        self, model_comparison: ModelComparison
    ) -> List[Insight]:
        """Create insights for dimension-specific comparisons."""

        insights = []

        for dimension, comparison in model_comparison.dimension_comparisons.items():
            if abs(comparison["score_difference"]) > 0.3:  # Significant dimension difference

                advantage_model = (
                    model_comparison.model_a.model_name
                    if comparison["advantage"] == "self"
                    else model_comparison.model_b.model_name
                )

                insights.append(
                    Insight(
                        insight_id=uuid4(),
                        insight_type=InsightType.QUALITY_PATTERN,
                        severity=InsightSeverity.MEDIUM,
                        title=f"Strong {dimension.replace('_', ' ').title()} Advantage: {advantage_model}",
                        description=(
                            f"{advantage_model} shows significant advantage in {dimension} "
                            f"(difference: {comparison['score_difference']:+.3f}). "
                            f"This dimension may be a key differentiator."
                        ),
                        confidence_score=Decimal("0.8"),
                        evidence={
                            "dimension": dimension,
                            "advantage_model": advantage_model,
                            "score_difference": comparison["score_difference"],
                            "advantage_score": (
                                comparison["self_score"]
                                if comparison["advantage"] == "self"
                                else comparison["other_score"]
                            ),
                        },
                        recommendations=[
                            f"Leverage {advantage_model} for tasks requiring strong {dimension}",
                            f"Investigate what makes {advantage_model} excel in {dimension}",
                            "Consider dimension-specific model selection",
                        ],
                        affected_models=[
                            model_comparison.model_a.model_id,
                            model_comparison.model_b.model_id,
                        ],
                        category=dimension,
                    )
                )

        return insights

    # Additional helper methods would be implemented here...
    def _generate_reliability_insights(
        self, model_performances: List[ModelPerformance]
    ) -> List[Insight]:
        """Generate reliability insights - placeholder."""
        return []

    def _generate_consistency_insights(
        self, model_performances: List[ModelPerformance]
    ) -> List[Insight]:
        """Generate consistency insights - placeholder."""
        return []

    def _generate_quality_pattern_insights(
        self, model_performances: List[ModelPerformance]
    ) -> List[Insight]:
        """Generate quality pattern insights - placeholder."""
        return []

    def _detect_category_bias(self, analysis_result: AnalysisResult) -> List[Insight]:
        """Detect category bias - placeholder."""
        return []

    def _detect_difficulty_bias(self, analysis_result: AnalysisResult) -> List[Insight]:
        """Detect difficulty bias - placeholder."""
        return []

    def _detect_judge_bias(self, analysis_result: AnalysisResult) -> List[Insight]:
        """Detect judge bias - placeholder."""
        return []

    def _generate_model_selection_recommendations(
        self, analysis_result: AnalysisResult
    ) -> List[Insight]:
        """Generate model selection recommendations - placeholder."""
        return []

    def _generate_improvement_recommendations(
        self, analysis_result: AnalysisResult
    ) -> List[Insight]:
        """Generate improvement recommendations - placeholder."""
        return []

    def _generate_sample_size_recommendations(
        self, analysis_result: AnalysisResult
    ) -> List[Insight]:
        """Generate sample size recommendations - placeholder."""
        return []

    def _generate_business_context_recommendations(
        self, analysis_result: AnalysisResult, business_context: Dict[str, Any]
    ) -> List[Insight]:
        """Generate business context recommendations - placeholder."""
        return []
