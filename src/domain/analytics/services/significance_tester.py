"""Significance testing service for A/B test analysis."""

from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...evaluation.entities.evaluation_result import EvaluationResult
from ...model_provider.entities.model_response import ModelResponse
from ..entities.statistical_test import CorrectionMethod, StatisticalTest, TestType
from ..exceptions import InsufficientDataError, InvalidDataError, StatisticalError, ValidationError
from ..value_objects.test_result import TestResult


class SignificanceTester:
    """Domain service for statistical significance testing."""

    def __init__(self):
        """Initialize significance tester."""
        self._default_confidence_level = 0.95
        self._minimum_sample_size = 30

    def test_model_performance_difference(
        self,
        model_a_results: List[EvaluationResult],
        model_b_results: List[EvaluationResult],
        test_type: TestType = TestType.TTEST_INDEPENDENT,
        confidence_level: float = 0.95,
        dimension: Optional[str] = None,
    ) -> TestResult:
        """Test for significant difference in model performance."""

        # Validate inputs
        self._validate_evaluation_results(model_a_results, model_b_results)

        # Extract performance scores
        scores_a = self._extract_scores(model_a_results, dimension)
        scores_b = self._extract_scores(model_b_results, dimension)

        # Prepare data for statistical test
        data = {"model_a": scores_a, "model_b": scores_b}

        # Create and run statistical test
        statistical_test = StatisticalTest.create(
            test_type=test_type,
            confidence_level=confidence_level,
            minimum_sample_size=self._minimum_sample_size,
        )

        return statistical_test.run_test(data)

    def test_multiple_models(
        self,
        model_results: Dict[str, List[EvaluationResult]],
        test_type: TestType = TestType.ANOVA_ONEWAY,
        confidence_level: float = 0.95,
        dimension: Optional[str] = None,
        correction_method: CorrectionMethod = CorrectionMethod.BONFERRONI,
    ) -> Dict[str, TestResult]:
        """Test for significant differences among multiple models."""

        if len(model_results) < 2:
            raise ValidationError("At least 2 models required for comparison")

        model_names = list(model_results.keys())
        results = {}

        # If more than 2 models, perform pairwise comparisons
        if len(model_results) == 2:
            # Simple two-model comparison
            model_a, model_b = model_names
            results[f"{model_a}_vs_{model_b}"] = self.test_model_performance_difference(
                model_results[model_a],
                model_results[model_b],
                test_type=TestType.TTEST_INDEPENDENT,
                confidence_level=confidence_level,
                dimension=dimension,
            )
        else:
            # Multiple pairwise comparisons
            p_values = []
            pairwise_results = {}

            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model_a, model_b = model_names[i], model_names[j]
                    comparison_key = f"{model_a}_vs_{model_b}"

                    test_result = self.test_model_performance_difference(
                        model_results[model_a],
                        model_results[model_b],
                        test_type=TestType.TTEST_INDEPENDENT,
                        confidence_level=confidence_level,
                        dimension=dimension,
                    )

                    pairwise_results[comparison_key] = test_result
                    p_values.append(test_result.p_value)

            # Apply multiple comparisons correction
            if correction_method != CorrectionMethod.NONE:
                statistical_test = StatisticalTest.create(
                    test_type=test_type,
                    confidence_level=confidence_level,
                    correction_method=correction_method,
                )

                corrected_p_values = statistical_test.apply_multiple_comparisons_correction(
                    p_values
                )

                # Update results with corrected p-values
                for idx, (comparison_key, original_result) in enumerate(pairwise_results.items()):
                    corrected_result = TestResult(
                        test_type=original_result.test_type,
                        statistic=original_result.statistic,
                        p_value=original_result.p_value,
                        effect_size=original_result.effect_size,
                        confidence_interval=original_result.confidence_interval,
                        degrees_of_freedom=original_result.degrees_of_freedom,
                        interpretation=original_result.interpretation,
                        sample_sizes=original_result.sample_sizes,
                        test_assumptions=original_result.test_assumptions,
                        power=original_result.power,
                        corrected_p_value=corrected_p_values[idx],
                    )
                    results[comparison_key] = corrected_result
            else:
                results.update(pairwise_results)

        return results

    def test_dimension_differences(
        self,
        evaluation_results: List[EvaluationResult],
        dimensions: List[str],
        confidence_level: float = 0.95,
    ) -> Dict[str, TestResult]:
        """Test for differences across evaluation dimensions."""

        if not dimensions:
            raise ValidationError("At least one dimension required")

        if not evaluation_results:
            raise InsufficientDataError("Evaluation results cannot be empty")

        results = {}

        # Compare each dimension pair
        for i in range(len(dimensions)):
            for j in range(i + 1, len(dimensions)):
                dim_a, dim_b = dimensions[i], dimensions[j]
                comparison_key = f"{dim_a}_vs_{dim_b}"

                try:
                    # Extract scores for each dimension
                    scores_a = []
                    scores_b = []

                    for result in evaluation_results:
                        if dim_a in result.dimension_scores and dim_b in result.dimension_scores:
                            scores_a.append(float(result.dimension_scores[dim_a]))
                            scores_b.append(float(result.dimension_scores[dim_b]))

                    if not scores_a or not scores_b:
                        continue  # Skip if no overlapping data

                    # Use paired t-test since scores are from same evaluations
                    data = {dim_a: scores_a, dim_b: scores_b}

                    statistical_test = StatisticalTest.create(
                        test_type=TestType.TTEST_PAIRED, confidence_level=confidence_level
                    )

                    results[comparison_key] = statistical_test.run_test(data)

                except Exception as e:
                    # Log error and continue with other comparisons
                    continue

        return results

    def test_difficulty_level_effects(
        self, evaluation_results: List[EvaluationResult], confidence_level: float = 0.95
    ) -> Optional[TestResult]:
        """Test if difficulty level affects model performance."""

        # Group results by difficulty level (from metadata)
        difficulty_groups = {}

        for result in evaluation_results:
            difficulty = result.metadata.get("difficulty_level", "unknown")
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(float(result.overall_score))

        # Remove unknown difficulty group if present
        if "unknown" in difficulty_groups:
            del difficulty_groups["unknown"]

        if len(difficulty_groups) < 2:
            return None  # Need at least 2 difficulty levels

        try:
            # Use one-way ANOVA for multiple difficulty levels
            if len(difficulty_groups) > 2:
                statistical_test = StatisticalTest.create(
                    test_type=TestType.ANOVA_ONEWAY, confidence_level=confidence_level
                )
                return statistical_test.run_test(difficulty_groups)
            else:
                # Use t-test for 2 difficulty levels
                difficulty_levels = list(difficulty_groups.keys())
                data = {
                    difficulty_levels[0]: difficulty_groups[difficulty_levels[0]],
                    difficulty_levels[1]: difficulty_groups[difficulty_levels[1]],
                }

                statistical_test = StatisticalTest.create(
                    test_type=TestType.TTEST_INDEPENDENT, confidence_level=confidence_level
                )
                return statistical_test.run_test(data)

        except Exception:
            return None  # Return None if test fails

    def calculate_required_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        test_type: TestType = TestType.TTEST_INDEPENDENT,
    ) -> int:
        """Calculate required sample size for desired power."""

        if not (0.0 < power < 1.0):
            raise ValidationError("Power must be between 0 and 1")

        if not (0.0 < alpha < 1.0):
            raise ValidationError("Alpha must be between 0 and 1")

        if effect_size <= 0:
            raise ValidationError("Effect size must be positive")

        try:
            from statsmodels.stats.power import tt_solve_power

            if test_type in [TestType.TTEST_INDEPENDENT, TestType.TTEST_WELCH]:
                # For independent t-test
                sample_size = tt_solve_power(
                    effect_size=effect_size, power=power, alpha=alpha, alternative="two-sided"
                )
                return max(self._minimum_sample_size, int(sample_size) + 1)

            elif test_type == TestType.TTEST_PAIRED:
                # For paired t-test (typically requires smaller samples)
                sample_size = tt_solve_power(
                    effect_size=effect_size, power=power, alpha=alpha, alternative="two-sided"
                )
                # Paired tests are more powerful, so adjust down slightly
                return max(self._minimum_sample_size, int(sample_size * 0.8) + 1)

        except ImportError:
            # Fallback calculation using approximation
            pass

        # Simple approximation for required sample size
        # Based on Cohen's formulas for power analysis
        if test_type == TestType.TTEST_INDEPENDENT:
            # Approximate formula: n ≈ 16 / (effect_size^2) for power=0.8, alpha=0.05
            base_n = 16 / (effect_size**2)

            # Adjust for different power and alpha levels
            if power != 0.8:
                power_adjustment = (1 - power) / 0.2  # Adjust based on power
                base_n *= 1 + power_adjustment

            if alpha != 0.05:
                alpha_adjustment = 0.05 / alpha  # Adjust based on alpha
                base_n *= alpha_adjustment

            return max(self._minimum_sample_size, int(base_n) + 1)

        # Conservative estimate for other test types
        return max(self._minimum_sample_size, int(20 / (effect_size**2)) + 1)

    def _validate_evaluation_results(
        self, results_a: List[EvaluationResult], results_b: List[EvaluationResult]
    ) -> None:
        """Validate evaluation results for comparison."""

        if not results_a:
            raise InsufficientDataError("Model A results cannot be empty")

        if not results_b:
            raise InsufficientDataError("Model B results cannot be empty")

        # Check if results are completed
        incomplete_a = [r for r in results_a if not r.is_completed()]
        incomplete_b = [r for r in results_b if not r.is_completed()]

        if incomplete_a:
            raise InvalidDataError(f"Model A has {len(incomplete_a)} incomplete evaluation results")

        if incomplete_b:
            raise InvalidDataError(f"Model B has {len(incomplete_b)} incomplete evaluation results")

        # Check for failed evaluations
        failed_a = [r for r in results_a if r.has_error()]
        failed_b = [r for r in results_b if r.has_error()]

        if failed_a:
            raise InvalidDataError(f"Model A has {len(failed_a)} failed evaluation results")

        if failed_b:
            raise InvalidDataError(f"Model B has {len(failed_b)} failed evaluation results")

    def _extract_scores(
        self, evaluation_results: List[EvaluationResult], dimension: Optional[str] = None
    ) -> List[float]:
        """Extract scores from evaluation results."""

        scores = []

        for result in evaluation_results:
            if dimension is None:
                # Use overall score
                scores.append(float(result.overall_score))
            else:
                # Use specific dimension score
                if dimension in result.dimension_scores:
                    scores.append(float(result.dimension_scores[dimension]))
                else:
                    raise InvalidDataError(
                        f"Dimension '{dimension}' not found in evaluation result"
                    )

        if not scores:
            raise InsufficientDataError("No scores extracted from evaluation results")

        return scores
