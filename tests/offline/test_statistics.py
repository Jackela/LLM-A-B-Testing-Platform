"""Offline numerical contracts; never load providers, credentials or datasets."""
import unittest

import numpy as np
from scipy import stats

from src.domain.analytics.entities.statistical_test import StatisticalTest, TestType
from src.domain.analytics.exceptions import InsufficientDataError, InvalidDataError


class StatisticalContracts(unittest.TestCase):
    def test_independent_and_welch_match_reference(self):
        a = list(range(60))
        b = [v * 1.2 + 2 for v in a]
        for kind, equal in [(TestType.TTEST_INDEPENDENT, True), (TestType.TTEST_WELCH, False)]:
            with self.subTest(kind=kind):
                result = StatisticalTest.create(kind).run_test({"a": a, "b": b})
                reference = stats.ttest_ind(a, b, equal_var=equal)
                self.assertAlmostEqual(result.statistic, reference.statistic, places=10)
                self.assertAlmostEqual(result.p_value, reference.pvalue, places=10)
                self.assertEqual(result.sample_sizes, {"a": 60, "b": 60})

    def test_paired_matches_reference(self):
        a = list(range(60))
        b = [v - 1 + float(np.sin(v)) for v in a]
        result = StatisticalTest.create(TestType.TTEST_PAIRED).run_test({"before": a, "after": b})
        reference = stats.ttest_rel(a, b)
        self.assertAlmostEqual(result.statistic, reference.statistic, places=10)
        self.assertAlmostEqual(result.p_value, reference.pvalue, places=10)

    def test_sample_size_and_pairing_failures_remain_failures(self):
        test = StatisticalTest.create(TestType.TTEST_PAIRED)
        with self.assertRaises(InsufficientDataError):
            test.run_test({"a": [1, 2], "b": [2, 3]})
        with self.assertRaisesRegex(InvalidDataError, "equal sample sizes"):
            test.run_test({"a": list(range(60)), "b": list(range(61))})

    def test_nonfinite_input_is_rejected_before_numerical_work(self):
        test = StatisticalTest.create(TestType.TTEST_INDEPENDENT)
        for value in [float("inf"), float("-inf"), float("nan")]:
            with self.subTest(value=value), self.assertRaisesRegex(
                InvalidDataError, "contains invalid values"
            ):
                test.run_test({"a": [value] * 30, "b": list(range(60))})


if __name__ == "__main__":
    unittest.main()
