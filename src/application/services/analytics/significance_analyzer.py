"""Significance analyzer for advanced statistical analysis patterns."""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

from ....domain.analytics.entities.statistical_test import CorrectionMethod, TestType
from ....domain.analytics.exceptions import StatisticalError, ValidationError
from ....domain.analytics.value_objects.test_result import TestResult

logger = logging.getLogger(__name__)


@dataclass
class EffectSizeAnalysis:
    """Analysis of effect sizes with interpretations."""

    effect_size: float
    interpretation: str
    confidence_interval: Optional[Tuple[float, float]]
    practical_significance: bool
    statistical_power: Optional[float] = None


@dataclass
class MultipleComparisonResult:
    """Result of multiple comparison analysis."""

    original_p_values: List[float]
    corrected_p_values: List[float]
    significant_comparisons: List[str]
    correction_method: str
    family_wise_error_rate: float


class SignificanceAnalyzer:
    """Advanced analyzer for statistical significance patterns and corrections."""

    def __init__(self):
        self._logger = logger.getChild(self.__class__.__name__)
        self._effect_size_thresholds = {
            "negligible": 0.01,
            "small": 0.2,
            "medium": 0.5,
            "large": 0.8,
            "very_large": 1.2,
        }
        self._practical_significance_threshold = 0.1  # 10% improvement

    def analyze_effect_sizes(
        self, test_results: Dict[str, TestResult], include_power_analysis: bool = True
    ) -> Dict[str, EffectSizeAnalysis]:
        """
        Analyze effect sizes across multiple test results.

        Args:
            test_results: Dictionary of test results
            include_power_analysis: Whether to include statistical power analysis

        Returns:
            Dictionary of effect size analyses
        """
        analyses = {}

        for test_name, result in test_results.items():
            try:
                effect_size = float(result.effect_size) if result.effect_size else 0.0

                # Interpret effect size
                interpretation = self._interpret_effect_size(effect_size)

                # Calculate confidence interval for effect size if available
                ci = None
                if result.confidence_interval:
                    ci = (
                        float(result.confidence_interval.lower_bound),
                        float(result.confidence_interval.upper_bound),
                    )

                # Assess practical significance
                practical_significance = self._assess_practical_significance(effect_size)

                # Calculate statistical power if requested
                statistical_power = None
                if include_power_analysis and result.power:
                    statistical_power = float(result.power)

                analysis = EffectSizeAnalysis(
                    effect_size=effect_size,
                    interpretation=interpretation,
                    confidence_interval=ci,
                    practical_significance=practical_significance,
                    statistical_power=statistical_power,
                )

                analyses[test_name] = analysis

            except Exception as e:
                self._logger.warning(f"Effect size analysis failed for {test_name}: {str(e)}")
                continue

        return analyses

    def analyze_multiple_comparisons(
        self,
        test_results: Dict[str, TestResult],
        correction_method: CorrectionMethod = CorrectionMethod.BONFERRONI,
        family_wise_alpha: float = 0.05,
    ) -> MultipleComparisonResult:
        """
        Analyze multiple comparisons with correction methods.

        Args:
            test_results: Dictionary of test results
            correction_method: Method for multiple comparison correction
            family_wise_alpha: Family-wise error rate

        Returns:
            MultipleComparisonResult with corrected p-values
        """
        if not test_results:
            raise ValidationError("No test results provided for multiple comparison analysis")

        # Extract p-values
        p_values = []
        test_names = []

        for test_name, result in test_results.items():
            if result.p_value is not None:
                p_values.append(float(result.p_value))
                test_names.append(test_name)

        if not p_values:
            raise ValidationError("No valid p-values found in test results")

        # Apply correction method
        corrected_p_values = self._apply_multiple_comparison_correction(p_values, correction_method)

        # Identify significant comparisons
        significant_comparisons = []
        for i, (test_name, corrected_p) in enumerate(zip(test_names, corrected_p_values)):
            if corrected_p < family_wise_alpha:
                significant_comparisons.append(test_name)

        # Calculate family-wise error rate
        fwer = self._calculate_family_wise_error_rate(corrected_p_values, family_wise_alpha)

        return MultipleComparisonResult(
            original_p_values=p_values,
            corrected_p_values=corrected_p_values,
            significant_comparisons=significant_comparisons,
            correction_method=correction_method.value,
            family_wise_error_rate=fwer,
        )

    def detect_statistical_patterns(self, test_results: Dict[str, TestResult]) -> Dict[str, Any]:
        """
        Detect patterns in statistical test results.

        Args:
            test_results: Dictionary of test results

        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            "significance_patterns": {},
            "effect_size_patterns": {},
            "power_patterns": {},
            "outlier_analyses": {},
            "consistency_analysis": {},
        }

        if not test_results:
            return patterns

        # Analyze significance patterns
        patterns["significance_patterns"] = self._analyze_significance_patterns(test_results)

        # Analyze effect size patterns
        patterns["effect_size_patterns"] = self._analyze_effect_size_patterns(test_results)

        # Analyze statistical power patterns
        patterns["power_patterns"] = self._analyze_power_patterns(test_results)

        # Detect outlier analyses
        patterns["outlier_analyses"] = self._detect_outlier_analyses(test_results)

        # Analyze consistency across tests
        patterns["consistency_analysis"] = self._analyze_result_consistency(test_results)

        return patterns

    def calculate_meta_analysis_metrics(
        self, test_results: Dict[str, TestResult]
    ) -> Dict[str, Any]:
        """
        Calculate meta-analysis metrics across test results.

        Args:
            test_results: Dictionary of test results

        Returns:
            Dictionary of meta-analysis metrics
        """
        if not test_results:
            return {}

        # Extract effect sizes and weights
        effect_sizes = []
        weights = []

        for result in test_results.values():
            if result.effect_size is not None:
                effect_sizes.append(float(result.effect_size))
                # Use sample size as weight if available
                if result.sample_sizes:
                    total_n = sum(result.sample_sizes.values())
                    weights.append(total_n)
                else:
                    weights.append(1.0)  # Equal weight if no sample size info

        if not effect_sizes:
            return {}

        # Calculate weighted mean effect size
        if weights:
            weighted_mean = sum(es * w for es, w in zip(effect_sizes, weights)) / sum(weights)
        else:
            weighted_mean = sum(effect_sizes) / len(effect_sizes)

        # Calculate heterogeneity measures
        heterogeneity = self._calculate_heterogeneity(effect_sizes, weights)

        # Calculate overall confidence interval
        overall_ci = self._calculate_meta_analysis_ci(effect_sizes, weights)

        return {
            "weighted_mean_effect_size": weighted_mean,
            "effect_size_range": (min(effect_sizes), max(effect_sizes)),
            "heterogeneity": heterogeneity,
            "overall_confidence_interval": overall_ci,
            "number_of_studies": len(effect_sizes),
            "total_sample_size": sum(weights) if weights else len(effect_sizes),
        }

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(effect_size)

        if abs_effect < self._effect_size_thresholds["small"]:
            return "negligible"
        elif abs_effect < self._effect_size_thresholds["medium"]:
            return "small"
        elif abs_effect < self._effect_size_thresholds["large"]:
            return "medium"
        elif abs_effect < self._effect_size_thresholds["very_large"]:
            return "large"
        else:
            return "very_large"

    def _assess_practical_significance(self, effect_size: float) -> bool:
        """Assess whether effect size is practically significant."""
        return abs(effect_size) >= self._practical_significance_threshold

    def _apply_multiple_comparison_correction(
        self, p_values: List[float], correction_method: CorrectionMethod
    ) -> List[float]:
        """Apply multiple comparison correction to p-values."""

        if correction_method == CorrectionMethod.NONE:
            return p_values.copy()

        elif correction_method == CorrectionMethod.BONFERRONI:
            # Bonferroni correction: multiply each p-value by number of comparisons
            corrected = [min(1.0, p * len(p_values)) for p in p_values]
            return corrected

        elif correction_method == CorrectionMethod.HOLM:
            # Holm-Bonferroni correction (step-down method)
            sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
            corrected = [0.0] * len(p_values)

            for rank, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) - rank
                corrected[idx] = min(1.0, p_values[idx] * correction_factor)

                # Ensure monotonicity
                if rank > 0:
                    prev_idx = sorted_indices[rank - 1]
                    corrected[idx] = max(corrected[idx], corrected[prev_idx])

            return corrected

        elif correction_method == CorrectionMethod.BENJAMINI_HOCHBERG:
            # Benjamini-Hochberg (FDR) correction
            sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
            corrected = [0.0] * len(p_values)

            for rank, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) / (rank + 1)
                corrected[idx] = min(1.0, p_values[idx] * correction_factor)

            # Ensure monotonicity (reverse order)
            for i in range(len(sorted_indices) - 2, -1, -1):
                idx = sorted_indices[i]
                next_idx = sorted_indices[i + 1]
                corrected[idx] = min(corrected[idx], corrected[next_idx])

            return corrected

        else:
            # Default to Bonferroni for unknown methods
            self._logger.warning(f"Unknown correction method {correction_method}, using Bonferroni")
            return self._apply_multiple_comparison_correction(p_values, CorrectionMethod.BONFERRONI)

    def _calculate_family_wise_error_rate(
        self, corrected_p_values: List[float], alpha: float
    ) -> float:
        """Calculate family-wise error rate."""
        significant_count = sum(1 for p in corrected_p_values if p < alpha)
        return significant_count / len(corrected_p_values) if corrected_p_values else 0.0

    def _analyze_significance_patterns(self, test_results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Analyze patterns in statistical significance."""
        p_values = [float(r.p_value) for r in test_results.values() if r.p_value is not None]

        if not p_values:
            return {}

        significant_at_05 = sum(1 for p in p_values if p < 0.05)
        significant_at_01 = sum(1 for p in p_values if p < 0.01)

        return {
            "total_tests": len(p_values),
            "significant_at_05": significant_at_05,
            "significant_at_01": significant_at_01,
            "significance_rate_05": significant_at_05 / len(p_values),
            "significance_rate_01": significant_at_01 / len(p_values),
            "mean_p_value": sum(p_values) / len(p_values),
            "median_p_value": sorted(p_values)[len(p_values) // 2],
        }

    def _analyze_effect_size_patterns(self, test_results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Analyze patterns in effect sizes."""
        effect_sizes = [
            float(r.effect_size) for r in test_results.values() if r.effect_size is not None
        ]

        if not effect_sizes:
            return {}

        # Categorize effect sizes
        categories = {"negligible": 0, "small": 0, "medium": 0, "large": 0, "very_large": 0}

        for es in effect_sizes:
            interpretation = self._interpret_effect_size(es)
            categories[interpretation] += 1

        return {
            "total_effect_sizes": len(effect_sizes),
            "mean_effect_size": sum(effect_sizes) / len(effect_sizes),
            "effect_size_distribution": categories,
            "practically_significant_count": sum(
                1 for es in effect_sizes if self._assess_practical_significance(es)
            ),
        }

    def _analyze_power_patterns(self, test_results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Analyze patterns in statistical power."""
        power_values = [float(r.power) for r in test_results.values() if r.power is not None]

        if not power_values:
            return {}

        adequate_power = sum(1 for p in power_values if p >= 0.8)

        return {
            "total_power_analyses": len(power_values),
            "adequate_power_count": adequate_power,
            "adequate_power_rate": adequate_power / len(power_values),
            "mean_power": sum(power_values) / len(power_values),
            "underpowered_analyses": [i for i, p in enumerate(power_values) if p < 0.8],
        }

    def _detect_outlier_analyses(self, test_results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Detect outlier analyses based on unusual patterns."""
        outliers = {
            "extreme_p_values": [],
            "extreme_effect_sizes": [],
            "low_power_tests": [],
            "inconsistent_results": [],
        }

        # Extract values for outlier detection
        p_values = []
        effect_sizes = []
        power_values = []
        test_names = []

        for name, result in test_results.items():
            test_names.append(name)
            p_values.append(float(result.p_value) if result.p_value else None)
            effect_sizes.append(float(result.effect_size) if result.effect_size else None)
            power_values.append(float(result.power) if result.power else None)

        # Detect extreme p-values (very close to 0 or 1)
        for i, p in enumerate(p_values):
            if p is not None and (p < 0.001 or p > 0.99):
                outliers["extreme_p_values"].append({"test": test_names[i], "p_value": p})

        # Detect extreme effect sizes
        for i, es in enumerate(effect_sizes):
            if es is not None and abs(es) > 2.0:  # Very large effect
                outliers["extreme_effect_sizes"].append({"test": test_names[i], "effect_size": es})

        # Detect low power tests
        for i, power in enumerate(power_values):
            if power is not None and power < 0.5:  # Very low power
                outliers["low_power_tests"].append({"test": test_names[i], "power": power})

        return outliers

    def _analyze_result_consistency(self, test_results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Analyze consistency across test results."""
        p_values = [float(r.p_value) for r in test_results.values() if r.p_value is not None]
        effect_sizes = [
            float(r.effect_size) for r in test_results.values() if r.effect_size is not None
        ]

        consistency = {}

        if len(p_values) > 1:
            # Calculate coefficient of variation for p-values
            mean_p = sum(p_values) / len(p_values)
            var_p = sum((p - mean_p) ** 2 for p in p_values) / len(p_values)
            cv_p = (var_p**0.5) / mean_p if mean_p > 0 else 0

            consistency["p_value_consistency"] = {
                "coefficient_of_variation": cv_p,
                "interpretation": "high" if cv_p < 0.5 else "medium" if cv_p < 1.0 else "low",
            }

        if len(effect_sizes) > 1:
            # Calculate consistency of effect sizes
            mean_es = sum(effect_sizes) / len(effect_sizes)
            var_es = sum((es - mean_es) ** 2 for es in effect_sizes) / len(effect_sizes)

            # Check for direction consistency
            positive_effects = sum(1 for es in effect_sizes if es > 0)
            direction_consistency = max(
                positive_effects, len(effect_sizes) - positive_effects
            ) / len(effect_sizes)

            consistency["effect_size_consistency"] = {
                "variance": var_es,
                "direction_consistency": direction_consistency,
                "interpretation": (
                    "high"
                    if direction_consistency > 0.8
                    else "medium" if direction_consistency > 0.6 else "low"
                ),
            }

        return consistency

    def _calculate_heterogeneity(
        self, effect_sizes: List[float], weights: List[float]
    ) -> Dict[str, float]:
        """Calculate heterogeneity measures for meta-analysis."""
        if len(effect_sizes) < 2:
            return {"q_statistic": 0.0, "i_squared": 0.0, "tau_squared": 0.0}

        # Calculate weighted mean
        weighted_mean = sum(es * w for es, w in zip(effect_sizes, weights)) / sum(weights)

        # Calculate Q statistic (Cochran's Q)
        q_statistic = sum(w * (es - weighted_mean) ** 2 for es, w in zip(effect_sizes, weights))

        # Calculate I-squared
        df = len(effect_sizes) - 1
        i_squared = max(0, (q_statistic - df) / q_statistic) if q_statistic > 0 else 0

        # Calculate tau-squared (between-study variance)
        if q_statistic > df:
            c = sum(weights) - sum(w**2 for w in weights) / sum(weights)
            tau_squared = (q_statistic - df) / c if c > 0 else 0
        else:
            tau_squared = 0

        return {"q_statistic": q_statistic, "i_squared": i_squared, "tau_squared": tau_squared}

    def _calculate_meta_analysis_ci(
        self, effect_sizes: List[float], weights: List[float], confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for meta-analysis."""
        if not effect_sizes:
            return (0.0, 0.0)

        # Calculate weighted mean and standard error
        weighted_mean = sum(es * w for es, w in zip(effect_sizes, weights)) / sum(weights)
        se = (1 / sum(weights)) ** 0.5

        # Calculate critical value for confidence interval
        from scipy.stats import norm

        alpha = 1 - confidence_level
        z_critical = norm.ppf(1 - alpha / 2)

        # Calculate confidence interval
        lower = weighted_mean - z_critical * se
        upper = weighted_mean + z_critical * se

        return (lower, upper)
