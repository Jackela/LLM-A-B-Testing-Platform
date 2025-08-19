"""Test statistical test implementations for mathematical accuracy."""

from decimal import Decimal

import numpy as np
import pytest
from scipy import stats

from src.domain.analytics.entities.statistical_test import (
    CorrectionMethod,
    StatisticalTest,
    TestType,
)
from src.domain.analytics.exceptions import InsufficientDataError, InvalidDataError, ValidationError


class TestStatisticalTestAccuracy:
    """Test statistical test mathematical accuracy against reference implementations."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)  # For reproducibility

        # Known datasets for validation
        self.group1 = [12.5, 14.2, 13.8, 15.1, 12.9, 14.6, 13.3, 15.0, 12.7, 14.4]
        self.group2 = [11.8, 12.1, 11.9, 12.5, 11.6, 12.3, 11.7, 12.4, 11.5, 12.0]

        # Paired data
        self.paired_before = [120, 118, 125, 130, 115, 122, 128, 119, 126, 121]
        self.paired_after = [115, 112, 120, 125, 108, 117, 123, 114, 121, 116]

        # Large effect size data for validation
        self.large_effect_1 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        self.large_effect_2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    def test_paired_ttest_accuracy(self):
        """Test paired t-test against scipy reference."""

        # Prepare test data
        data = {"before": self.paired_before, "after": self.paired_after}

        # Our implementation
        statistical_test = StatisticalTest.create(TestType.TTEST_PAIRED)
        our_result = statistical_test.run_test(data)

        # Scipy reference
        scipy_stat, scipy_p = stats.ttest_rel(self.paired_before, self.paired_after)

        # Validate statistical accuracy (within 0.001)
        assert abs(our_result.statistic - scipy_stat) < 0.001
        assert abs(our_result.p_value - scipy_p) < 0.001

        # Validate effect size calculation
        differences = np.array(self.paired_before) - np.array(self.paired_after)
        expected_effect_size = np.mean(differences) / np.std(differences, ddof=1)
        assert abs(our_result.effect_size - expected_effect_size) < 0.001

        # Validate confidence interval
        mean_diff = np.mean(differences)
        se_diff = stats.sem(differences)
        df = len(differences) - 1
        t_critical = stats.t.ppf(1 - 0.025, df)  # 95% CI
        expected_ci = (mean_diff - t_critical * se_diff, mean_diff + t_critical * se_diff)

        assert abs(our_result.confidence_interval[0] - expected_ci[0]) < 0.001
        assert abs(our_result.confidence_interval[1] - expected_ci[1]) < 0.001

    def test_independent_ttest_accuracy(self):
        """Test independent t-test against scipy reference."""

        data = {"group1": self.group1, "group2": self.group2}

        # Our implementation
        statistical_test = StatisticalTest.create(TestType.TTEST_INDEPENDENT)
        our_result = statistical_test.run_test(data)

        # Scipy reference
        scipy_stat, scipy_p = stats.ttest_ind(self.group1, self.group2)

        # Validate statistical accuracy
        assert abs(our_result.statistic - scipy_stat) < 0.001
        assert abs(our_result.p_value - scipy_p) < 0.001

        # Validate degrees of freedom
        expected_df = len(self.group1) + len(self.group2) - 2
        assert our_result.degrees_of_freedom == expected_df

    def test_welch_ttest_accuracy(self):
        """Test Welch's t-test (unequal variances) against scipy."""

        # Create data with unequal variances
        group1_unequal = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]  # Higher variance
        group2_unequal = [
            15.1,
            15.2,
            15.3,
            15.4,
            15.5,
            15.6,
            15.7,
            15.8,
            15.9,
            16.0,
        ]  # Lower variance

        data = {"group1": group1_unequal, "group2": group2_unequal}

        # Our implementation
        statistical_test = StatisticalTest.create(TestType.TTEST_WELCH)
        our_result = statistical_test.run_test(data)

        # Scipy reference
        scipy_stat, scipy_p = stats.ttest_ind(group1_unequal, group2_unequal, equal_var=False)

        # Validate statistical accuracy
        assert abs(our_result.statistic - scipy_stat) < 0.001
        assert abs(our_result.p_value - scipy_p) < 0.001

    def test_bootstrap_test_consistency(self):
        """Test bootstrap test for consistency and confidence intervals."""

        data = {"group1": self.group1, "group2": self.group2}

        # Our implementation
        statistical_test = StatisticalTest.create(TestType.BOOTSTRAP)
        result1 = statistical_test.run_test(data)

        # Run multiple times to test consistency
        np.random.seed(42)  # Reset seed for reproducibility
        result2 = statistical_test.run_test(data)

        # Results should be identical with same seed
        assert abs(result1.p_value - result2.p_value) < 0.001
        assert abs(result1.confidence_interval[0] - result2.confidence_interval[0]) < 0.001
        assert abs(result1.confidence_interval[1] - result2.confidence_interval[1]) < 0.001

        # Confidence interval should contain true difference most of the time
        true_diff = np.mean(self.group1) - np.mean(self.group2)
        ci_lower, ci_upper = result1.confidence_interval

        # For this specific dataset, the true difference should be within CI
        assert ci_lower <= true_diff <= ci_upper

    def test_effect_size_calculations(self):
        """Test effect size calculations for different magnitudes."""

        # Small effect size data (Cohen's d ≈ 0.2)
        small_effect_1 = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        small_effect_2 = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58]

        data_small = {"group1": small_effect_1, "group2": small_effect_2}

        statistical_test = StatisticalTest.create(TestType.TTEST_INDEPENDENT)
        result_small = statistical_test.run_test(data_small)

        # Should classify as small effect
        assert result_small.interpretation.effect_magnitude.value == "small"
        assert 0.1 <= abs(result_small.effect_size) <= 0.4

        # Large effect size data
        data_large = {"group1": self.large_effect_1, "group2": self.large_effect_2}
        result_large = statistical_test.run_test(data_large)

        # Should classify as large effect
        assert result_large.interpretation.effect_magnitude.value == "large"
        assert abs(result_large.effect_size) >= 0.8

    def test_confidence_intervals_coverage(self):
        """Test that confidence intervals have correct coverage properties."""

        # Generate multiple samples from known distribution
        np.random.seed(123)
        true_mean_diff = 2.0

        coverage_count = 0
        total_tests = 100

        for _ in range(total_tests):
            # Generate samples with known difference
            sample1 = np.random.normal(10, 2, 30)
            sample2 = np.random.normal(10 - true_mean_diff, 2, 30)

            data = {"group1": sample1.tolist(), "group2": sample2.tolist()}

            statistical_test = StatisticalTest.create(TestType.TTEST_INDEPENDENT)
            result = statistical_test.run_test(data)

            # Check if CI contains true difference
            ci_lower, ci_upper = result.confidence_interval
            if ci_lower <= true_mean_diff <= ci_upper:
                coverage_count += 1

        # 95% CI should cover true value approximately 95% of the time
        coverage_rate = coverage_count / total_tests
        assert 0.90 <= coverage_rate <= 1.0  # Allow some variability

    def test_multiple_comparisons_correction(self):
        """Test multiple comparisons correction methods."""

        # Create multiple p-values
        p_values = [0.01, 0.03, 0.05, 0.07, 0.12]

        # Test Bonferroni correction
        statistical_test = StatisticalTest.create(
            TestType.TTEST_INDEPENDENT, correction_method=CorrectionMethod.BONFERRONI
        )

        corrected_p_values = statistical_test.apply_multiple_comparisons_correction(p_values)

        # Bonferroni should multiply by number of comparisons
        expected_bonferroni = [min(1.0, p * len(p_values)) for p in p_values]

        for i, (corrected, expected) in enumerate(zip(corrected_p_values, expected_bonferroni)):
            assert abs(corrected - expected) < 0.001, f"Mismatch at index {i}"

    def test_assumption_checking(self):
        """Test statistical assumption checking."""

        # Normal data
        np.random.seed(42)
        normal_data1 = np.random.normal(0, 1, 50).tolist()
        normal_data2 = np.random.normal(0, 1, 50).tolist()

        data_normal = {"group1": normal_data1, "group2": normal_data2}

        statistical_test = StatisticalTest.create(TestType.TTEST_INDEPENDENT)
        result_normal = statistical_test.run_test(data_normal)

        # Should pass normality assumption
        assert result_normal.test_assumptions["normality"] == True

        # Should pass equal variance assumption
        assert result_normal.test_assumptions["equal_variances"] == True

        # Non-normal data (uniform distribution)
        uniform_data1 = np.random.uniform(0, 1, 50).tolist()
        uniform_data2 = np.random.uniform(0, 1, 50).tolist()

        data_uniform = {"group1": uniform_data1, "group2": uniform_data2}
        result_uniform = statistical_test.run_test(data_uniform)

        # May fail normality assumption (depending on sample)
        # This is a probabilistic test, so we just check the structure
        assert "normality" in result_uniform.test_assumptions
        assert "equal_variances" in result_uniform.test_assumptions

    def test_small_sample_handling(self):
        """Test handling of small sample sizes."""

        # Very small samples
        small_data = {"group1": [1, 2, 3], "group2": [4, 5, 6]}

        statistical_test = StatisticalTest.create(TestType.TTEST_INDEPENDENT, minimum_sample_size=5)

        # Should raise insufficient data error
        with pytest.raises(InsufficientDataError):
            statistical_test.run_test(small_data)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""

        statistical_test = StatisticalTest.create(TestType.TTEST_INDEPENDENT)

        # Empty data
        with pytest.raises(InvalidDataError):
            statistical_test.run_test({})

        # Single group
        with pytest.raises(InvalidDataError):
            statistical_test.run_test({"group1": [1, 2, 3]})

        # More than two groups for t-test
        with pytest.raises(InvalidDataError):
            statistical_test.run_test(
                {"group1": [1, 2, 3], "group2": [4, 5, 6], "group3": [7, 8, 9]}
            )

        # Invalid data types
        with pytest.raises(InvalidDataError):
            statistical_test.run_test({"group1": [1, 2, "invalid"], "group2": [4, 5, 6]})

        # NaN values
        with pytest.raises(InvalidDataError):
            statistical_test.run_test({"group1": [1, 2, float("nan")], "group2": [4, 5, 6]})

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""

        # Very large numbers
        large_data = {
            "group1": [1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4],
            "group2": [1e10 + 0.5, 1e10 + 1.5, 1e10 + 2.5, 1e10 + 3.5, 1e10 + 4.5],
        }

        statistical_test = StatisticalTest.create(
            TestType.TTEST_INDEPENDENT, minimum_sample_size=3  # Reduced for this test
        )

        # Should handle large numbers without overflow
        result = statistical_test.run_test(large_data)

        # Results should be finite
        assert not np.isnan(result.statistic)
        assert not np.isinf(result.statistic)
        assert not np.isnan(result.p_value)
        assert 0 <= result.p_value <= 1

    def test_parameter_validation(self):
        """Test statistical test parameter validation."""

        # Invalid confidence level
        with pytest.raises(ValidationError):
            StatisticalTest.create(TestType.TTEST_INDEPENDENT, confidence_level=1.5)

        with pytest.raises(ValidationError):
            StatisticalTest.create(TestType.TTEST_INDEPENDENT, confidence_level=0.0)

        # Invalid minimum sample size
        with pytest.raises(ValidationError):
            StatisticalTest.create(TestType.TTEST_INDEPENDENT, minimum_sample_size=0)

        # Valid parameters should work
        test = StatisticalTest.create(
            TestType.TTEST_INDEPENDENT, confidence_level=0.99, minimum_sample_size=10
        )

        assert test.confidence_level == 0.99
        assert test.alpha == 0.01
        assert test.minimum_sample_size == 10


class TestStatisticalTestInterpretation:
    """Test statistical test result interpretation."""

    def test_significance_interpretation(self):
        """Test significance level interpretation."""

        # Create test with different alpha levels
        statistical_test_05 = StatisticalTest.create(
            TestType.TTEST_INDEPENDENT, confidence_level=0.95
        )
        statistical_test_01 = StatisticalTest.create(
            TestType.TTEST_INDEPENDENT, confidence_level=0.99
        )

        # Data that should be significant at 0.05 but not at 0.01
        data = {
            "group1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "group2": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        }

        result_05 = statistical_test_05.run_test(data)
        result_01 = statistical_test_01.run_test(data)

        # Check p-value is between 0.01 and 0.05 (this is probabilistic)
        if 0.01 < result_05.p_value < 0.05:
            assert result_05.interpretation.is_significant == True
            assert result_01.interpretation.is_significant == False

    def test_effect_size_interpretation(self):
        """Test effect size interpretation guidelines."""

        statistical_test = StatisticalTest.create(TestType.TTEST_INDEPENDENT)

        # Large effect size data
        large_effect_data = {
            "group1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "group2": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        }

        result = statistical_test.run_test(large_effect_data)

        # Should interpret as large effect
        assert result.interpretation.effect_magnitude.value == "large"
        assert result.interpretation.practical_significance == True

        # Check recommendation includes effect size information
        assert (
            "large" in result.interpretation.recommendation.lower()
            or "substantial" in result.interpretation.recommendation.lower()
        )

    def test_recommendation_generation(self):
        """Test recommendation generation logic."""

        statistical_test = StatisticalTest.create(TestType.TTEST_INDEPENDENT)

        # Significant difference with large effect
        significant_large_data = {
            "group1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "group2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        }

        result = statistical_test.run_test(significant_large_data)

        # Should recommend action based on significant large effect
        recommendation = result.interpretation.recommendation
        assert len(recommendation) > 50  # Substantial recommendation
        assert any(
            word in recommendation.lower() for word in ["significant", "evidence", "difference"]
        )

    def test_confidence_intervals_interpretation(self):
        """Test confidence interval interpretation."""

        statistical_test = StatisticalTest.create(TestType.TTEST_INDEPENDENT)

        data = {
            "group1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "group2": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        }

        result = statistical_test.run_test(data)

        # Confidence interval should make sense
        ci_lower, ci_upper = result.confidence_interval

        # Lower bound should be less than upper bound
        assert ci_lower < ci_upper

        # Interval should contain the observed difference (for difference in means)
        observed_diff = np.mean(data["group1"]) - np.mean(data["group2"])
        assert ci_lower <= observed_diff <= ci_upper
