"""Statistical test entity for A/B test analysis."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np
from scipy import stats

from ..exceptions import (
    InsufficientDataError,
    InvalidDataError,
    NumericalInstabilityError,
    ValidationError,
)
from ..value_objects.test_result import EffectMagnitude, TestInterpretation, TestResult


class TestType(Enum):
    """Statistical test types."""

    TTEST_PAIRED = "t_test_paired"
    TTEST_INDEPENDENT = "t_test_independent"
    TTEST_WELCH = "t_test_welch"  # Unequal variances
    ANOVA_ONEWAY = "anova_oneway"
    ANOVA_REPEATED = "anova_repeated_measures"
    CHI_SQUARE = "chi_square"
    WILCOXON = "wilcoxon_signed_rank"
    MANN_WHITNEY = "mann_whitney_u"
    BOOTSTRAP = "bootstrap_confidence"
    KRUSKAL_WALLIS = "kruskal_wallis"


class CorrectionMethod(Enum):
    """Multiple comparison correction methods."""

    NONE = "none"
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    SIDAK = "sidak"


@dataclass
class StatisticalTest:
    """Statistical test aggregate for A/B test analysis."""

    test_id: UUID
    test_type: TestType
    confidence_level: float
    alpha: float
    correction_method: CorrectionMethod
    minimum_sample_size: int
    _domain_events: List[object] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate statistical test parameters."""
        if not self.test_id:
            self.test_id = uuid4()

        self._validate_parameters()

    @classmethod
    def create(
        cls,
        test_type: TestType,
        confidence_level: float = 0.95,
        correction_method: CorrectionMethod = CorrectionMethod.NONE,
        minimum_sample_size: int = 30,
    ) -> "StatisticalTest":
        """Factory method to create statistical test."""
        alpha = 1.0 - confidence_level

        return cls(
            test_id=uuid4(),
            test_type=test_type,
            confidence_level=confidence_level,
            alpha=alpha,
            correction_method=correction_method,
            minimum_sample_size=minimum_sample_size,
        )

    def _validate_parameters(self) -> None:
        """Validate statistical test parameters."""
        if not (0.0 < self.confidence_level < 1.0):
            raise ValidationError("Confidence level must be between 0 and 1")

        if not (0.0 < self.alpha < 1.0):
            raise ValidationError("Alpha must be between 0 and 1")

        if abs((1.0 - self.alpha) - self.confidence_level) > 1e-10:
            raise ValidationError("Confidence level must equal 1 - alpha")

        if self.minimum_sample_size < 1:
            raise ValidationError("Minimum sample size must be at least 1")

    def run_test(self, data: Dict[str, List[float]]) -> TestResult:
        """Run statistical test on provided data."""

        # Validate data requirements
        self._validate_data(data)

        # Check sample size requirements
        self._check_sample_size_requirements(data)

        # Perform appropriate statistical test
        if self.test_type == TestType.TTEST_PAIRED:
            return self._run_paired_ttest(data)
        elif self.test_type == TestType.TTEST_INDEPENDENT:
            return self._run_independent_ttest(data)
        elif self.test_type == TestType.TTEST_WELCH:
            return self._run_welch_ttest(data)
        elif self.test_type == TestType.ANOVA_ONEWAY:
            return self._run_oneway_anova(data)
        elif self.test_type == TestType.CHI_SQUARE:
            return self._run_chi_square_test(data)
        elif self.test_type == TestType.WILCOXON:
            return self._run_wilcoxon_test(data)
        elif self.test_type == TestType.MANN_WHITNEY:
            return self._run_mann_whitney_test(data)
        elif self.test_type == TestType.BOOTSTRAP:
            return self._run_bootstrap_test(data)
        elif self.test_type == TestType.KRUSKAL_WALLIS:
            return self._run_kruskal_wallis_test(data)
        else:
            raise InvalidDataError(f"Test type {self.test_type} not implemented")

    def _validate_data(self, data: Dict[str, List[float]]) -> None:
        """Validate input data for statistical analysis."""
        if not data:
            raise InvalidDataError("Data cannot be empty")

        if len(data) < 1:
            raise InvalidDataError("At least one group is required")

        # Check for empty groups
        for group_name, values in data.items():
            if not values:
                raise InvalidDataError(f"Group '{group_name}' cannot be empty")

            if not all(isinstance(v, (int, float)) and not math.isnan(v) for v in values):
                raise InvalidDataError(f"Group '{group_name}' contains invalid values")

    def _check_sample_size_requirements(self, data: Dict[str, List[float]]) -> None:
        """Check if data meets minimum sample size requirements."""
        for group_name, values in data.items():
            if len(values) < self.minimum_sample_size:
                raise InsufficientDataError(
                    f"Group '{group_name}' has {len(values)} samples, "
                    f"minimum required is {self.minimum_sample_size}"
                )

    def _run_paired_ttest(self, data: Dict[str, List[float]]) -> TestResult:
        """Perform paired t-test."""
        if len(data) != 2:
            raise InvalidDataError("Paired t-test requires exactly 2 groups")

        groups = list(data.values())
        group_names = list(data.keys())
        group1, group2 = groups[0], groups[1]

        if len(group1) != len(group2):
            raise InvalidDataError("Paired t-test requires equal sample sizes")

        try:
            # Perform paired t-test
            statistic, p_value = stats.ttest_rel(group1, group2)

            # Calculate effect size (Cohen's d for paired samples)
            differences = np.array(group1) - np.array(group2)
            effect_size = np.mean(differences) / np.std(differences, ddof=1)

            # Calculate confidence interval for mean difference
            mean_diff = np.mean(differences)
            se_diff = stats.sem(differences)
            df = len(differences) - 1
            t_critical = stats.t.ppf(1 - self.alpha / 2, df)
            ci_half_width = t_critical * se_diff

            confidence_interval = (mean_diff - ci_half_width, mean_diff + ci_half_width)

            # Check test assumptions
            assumptions = self._check_ttest_assumptions(group1, group2, paired=True)

            # Calculate statistical power
            power = self._calculate_ttest_power(effect_size, len(differences), paired=True)

        except Exception as e:
            raise NumericalInstabilityError(f"Failed to compute paired t-test: {str(e)}")

        interpretation = self._interpret_result(p_value, effect_size)

        return TestResult(
            test_type=self.test_type.value,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=confidence_interval,
            degrees_of_freedom=df,
            interpretation=interpretation,
            sample_sizes={name: len(values) for name, values in data.items()},
            test_assumptions=assumptions,
            power=float(power) if power else None,
        )

    def _run_independent_ttest(self, data: Dict[str, List[float]]) -> TestResult:
        """Perform independent samples t-test."""
        if len(data) != 2:
            raise InvalidDataError("Independent t-test requires exactly 2 groups")

        groups = list(data.values())
        group_names = list(data.keys())
        group1, group2 = groups[0], groups[1]

        try:
            # Perform independent t-test (equal variances assumed)
            statistic, p_value = stats.ttest_ind(group1, group2)

            # Calculate pooled effect size (Cohen's d)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)

            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            effect_size = (mean1 - mean2) / pooled_std

            # Calculate confidence interval for mean difference
            mean_diff = mean1 - mean2
            se_diff = pooled_std * np.sqrt(1 / n1 + 1 / n2)
            df = n1 + n2 - 2
            t_critical = stats.t.ppf(1 - self.alpha / 2, df)
            ci_half_width = t_critical * se_diff

            confidence_interval = (mean_diff - ci_half_width, mean_diff + ci_half_width)

            # Check test assumptions
            assumptions = self._check_ttest_assumptions(group1, group2, paired=False)

            # Calculate statistical power
            power = self._calculate_ttest_power(effect_size, min(n1, n2), paired=False)

        except Exception as e:
            raise NumericalInstabilityError(f"Failed to compute independent t-test: {str(e)}")

        interpretation = self._interpret_result(p_value, effect_size)

        return TestResult(
            test_type=self.test_type.value,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=confidence_interval,
            degrees_of_freedom=df,
            interpretation=interpretation,
            sample_sizes={name: len(values) for name, values in data.items()},
            test_assumptions=assumptions,
            power=float(power) if power else None,
        )

    def _run_welch_ttest(self, data: Dict[str, List[float]]) -> TestResult:
        """Perform Welch's t-test (unequal variances)."""
        if len(data) != 2:
            raise InvalidDataError("Welch's t-test requires exactly 2 groups")

        groups = list(data.values())
        group1, group2 = groups[0], groups[1]

        try:
            # Perform Welch's t-test
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)

            # Calculate effect size using separate standard deviations
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)

            # Use Hedges' g for unequal variances
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            effect_size = (mean1 - mean2) / pooled_std

            # Calculate Welch-Satterthwaite degrees of freedom
            s1_sq, s2_sq = std1**2, std2**2
            df = (s1_sq / n1 + s2_sq / n2) ** 2 / (
                (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
            )

            # Calculate confidence interval
            mean_diff = mean1 - mean2
            se_diff = np.sqrt(s1_sq / n1 + s2_sq / n2)
            t_critical = stats.t.ppf(1 - self.alpha / 2, df)
            ci_half_width = t_critical * se_diff

            confidence_interval = (mean_diff - ci_half_width, mean_diff + ci_half_width)

            # Check assumptions (less restrictive for Welch's test)
            assumptions = {
                "normality": self._check_normality(group1) and self._check_normality(group2),
                "independence": True,  # Assumed for independent samples
                "equal_variances": False,  # Not required for Welch's test
            }

        except Exception as e:
            raise NumericalInstabilityError(f"Failed to compute Welch's t-test: {str(e)}")

        interpretation = self._interpret_result(p_value, effect_size)

        return TestResult(
            test_type=self.test_type.value,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=confidence_interval,
            degrees_of_freedom=int(df),
            interpretation=interpretation,
            sample_sizes={name: len(values) for name, values in data.items()},
            test_assumptions=assumptions,
        )

    def _run_bootstrap_test(self, data: Dict[str, List[float]]) -> TestResult:
        """Perform bootstrap confidence interval test."""
        if len(data) != 2:
            raise InvalidDataError("Bootstrap test requires exactly 2 groups")

        groups = list(data.values())
        group1, group2 = groups[0], groups[1]
        n_bootstrap = 10000

        try:
            # Bootstrap sampling for mean difference
            bootstrap_diffs = []
            mean1_orig, mean2_orig = np.mean(group1), np.mean(group2)

            np.random.seed(42)  # For reproducibility
            for _ in range(n_bootstrap):
                boot_sample1 = np.random.choice(group1, size=len(group1), replace=True)
                boot_sample2 = np.random.choice(group2, size=len(group2), replace=True)

                boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
                bootstrap_diffs.append(boot_diff)

            bootstrap_diffs = np.array(bootstrap_diffs)

            # Calculate confidence interval from bootstrap distribution
            alpha_level = self.alpha
            lower_percentile = 100 * alpha_level / 2
            upper_percentile = 100 * (1 - alpha_level / 2)

            ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
            ci_upper = np.percentile(bootstrap_diffs, upper_percentile)
            confidence_interval = (float(ci_lower), float(ci_upper))

            # Calculate p-value (proportion of bootstrap samples with opposite sign)
            observed_diff = mean1_orig - mean2_orig
            if observed_diff >= 0:
                p_value = 2 * np.mean(bootstrap_diffs <= 0)
            else:
                p_value = 2 * np.mean(bootstrap_diffs >= 0)

            p_value = min(p_value, 1.0)  # Cap at 1.0

            # Effect size using bootstrap standard error
            effect_size = observed_diff / np.std(bootstrap_diffs)

            # Assumptions (bootstrap is non-parametric)
            assumptions = {
                "independence": True,
                "representative_sampling": True,
                "sufficient_sample_size": len(group1) >= 30 and len(group2) >= 30,
            }

        except Exception as e:
            raise NumericalInstabilityError(f"Failed to compute bootstrap test: {str(e)}")

        interpretation = self._interpret_result(p_value, effect_size)

        return TestResult(
            test_type=self.test_type.value,
            statistic=float(observed_diff),  # Use mean difference as test statistic
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=confidence_interval,
            degrees_of_freedom=None,  # Not applicable for bootstrap
            interpretation=interpretation,
            sample_sizes={name: len(values) for name, values in data.items()},
            test_assumptions=assumptions,
        )

    def _check_ttest_assumptions(
        self, group1: List[float], group2: List[float], paired: bool = False
    ) -> Dict[str, bool]:
        """Check t-test assumptions."""
        assumptions = {
            "normality": self._check_normality(group1) and self._check_normality(group2),
            "independence": True,  # Assumed for experimental data
        }

        if not paired:
            assumptions["equal_variances"] = self._check_equal_variances(group1, group2)

        return assumptions

    def _check_normality(self, data: List[float], alpha: float = 0.05) -> bool:
        """Check normality using Shapiro-Wilk test."""
        if len(data) < 3:
            return False  # Cannot test normality with < 3 samples

        if len(data) > 5000:
            # Use Anderson-Darling for large samples
            result = stats.anderson(data, dist="norm")
            return result.statistic < result.critical_values[2]  # 5% level

        _, p_value = stats.shapiro(data)
        return p_value > alpha

    def _check_equal_variances(self, group1: List[float], group2: List[float]) -> bool:
        """Check equal variances using Levene's test."""
        try:
            _, p_value = stats.levene(group1, group2)
            return p_value > 0.05
        except:
            return False

    def _calculate_ttest_power(
        self, effect_size: float, sample_size: int, paired: bool = False, alpha: float = None
    ) -> Optional[float]:
        """Calculate statistical power for t-test."""
        if alpha is None:
            alpha = self.alpha

        if sample_size < 2:
            return None

        try:
            from statsmodels.stats.power import ttest_power

            if paired:
                return ttest_power(effect_size, sample_size, alpha, alternative="two-sided")
            else:
                return ttest_power(effect_size, sample_size, alpha, alternative="two-sided")
        except ImportError:
            # Fallback calculation if statsmodels not available
            return None

    def _interpret_result(self, p_value: float, effect_size: float) -> TestInterpretation:
        """Interpret statistical test results."""
        is_significant = p_value < self.alpha

        # Effect size interpretation (Cohen's conventions)
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            effect_magnitude = EffectMagnitude.NEGLIGIBLE
        elif abs_effect < 0.5:
            effect_magnitude = EffectMagnitude.SMALL
        elif abs_effect < 0.8:
            effect_magnitude = EffectMagnitude.MEDIUM
        else:
            effect_magnitude = EffectMagnitude.LARGE

        practical_significance = abs_effect >= 0.2  # Threshold for practical importance
        recommendation = self._generate_recommendation(
            is_significant, effect_size, effect_magnitude
        )

        return TestInterpretation(
            is_significant=is_significant,
            significance_level=self.alpha,
            effect_magnitude=effect_magnitude,
            practical_significance=practical_significance,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self, is_significant: bool, effect_size: float, effect_magnitude: EffectMagnitude
    ) -> str:
        """Generate recommendation based on test results."""
        direction = "positive" if effect_size > 0 else "negative" if effect_size < 0 else "no"

        if is_significant and effect_magnitude in [EffectMagnitude.MEDIUM, EffectMagnitude.LARGE]:
            return (
                f"Statistically significant {direction} difference with {effect_magnitude.value} "
                "effect size. Strong evidence for practical difference between groups."
            )
        elif is_significant and effect_magnitude == EffectMagnitude.SMALL:
            return (
                f"Statistically significant {direction} difference with small effect size. "
                "Consider practical significance and cost-benefit trade-offs."
            )
        elif is_significant and effect_magnitude == EffectMagnitude.NEGLIGIBLE:
            return (
                "Statistically significant but negligible effect size. "
                "Likely not practically meaningful despite statistical significance."
            )
        elif not is_significant and effect_magnitude in [
            EffectMagnitude.MEDIUM,
            EffectMagnitude.LARGE,
        ]:
            return (
                f"Not statistically significant but {effect_magnitude.value} effect size observed. "
                "Consider increasing sample size or check for methodological issues."
            )
        else:
            return (
                "No statistically significant difference detected. "
                "Insufficient evidence to conclude meaningful difference between groups."
            )

    def apply_multiple_comparisons_correction(self, p_values: List[float]) -> List[float]:
        """Apply multiple comparisons correction."""
        if self.correction_method == CorrectionMethod.NONE:
            return p_values

        try:
            from statsmodels.stats.multitest import multipletests

            method_map = {
                CorrectionMethod.BONFERRONI: "bonferroni",
                CorrectionMethod.HOLM: "holm",
                CorrectionMethod.BENJAMINI_HOCHBERG: "fdr_bh",
                CorrectionMethod.SIDAK: "sidak",
            }

            method = method_map.get(self.correction_method)
            if not method:
                return p_values

            _, corrected_p_values, _, _ = multipletests(p_values, alpha=self.alpha, method=method)

            return corrected_p_values.tolist()

        except ImportError:
            # Fallback to simple Bonferroni correction
            if self.correction_method == CorrectionMethod.BONFERRONI:
                return [min(1.0, p * len(p_values)) for p in p_values]
            return p_values

    # Additional test methods would be implemented here following the same pattern...

    def _run_oneway_anova(self, data: Dict[str, List[float]]) -> TestResult:
        """Placeholder for one-way ANOVA implementation."""
        # Implementation would follow similar pattern to t-tests
        raise NotImplementedError("ANOVA implementation coming in next iteration")

    def _run_chi_square_test(self, data: Dict[str, List[float]]) -> TestResult:
        """Placeholder for chi-square test implementation."""
        raise NotImplementedError("Chi-square test implementation coming in next iteration")

    def _run_wilcoxon_test(self, data: Dict[str, List[float]]) -> TestResult:
        """Placeholder for Wilcoxon test implementation."""
        raise NotImplementedError("Wilcoxon test implementation coming in next iteration")

    def _run_mann_whitney_test(self, data: Dict[str, List[float]]) -> TestResult:
        """Placeholder for Mann-Whitney test implementation."""
        raise NotImplementedError("Mann-Whitney test implementation coming in next iteration")

    def _run_kruskal_wallis_test(self, data: Dict[str, List[float]]) -> TestResult:
        """Placeholder for Kruskal-Wallis test implementation."""
        raise NotImplementedError("Kruskal-Wallis test implementation coming in next iteration")
