"""Consensus algorithm service for multi-judge evaluation."""

import statistics
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from ..entities.evaluation_result import EvaluationResult
from ..exceptions import ConsensusCalculationError, InsufficientDataError, ValidationError
from ..value_objects.consensus_result import ConsensusResult


class ConsensusAlgorithm:
    """Service for calculating consensus from multiple judge results."""

    def __init__(self):
        """Initialize consensus algorithm."""
        self._outlier_detection_threshold = Decimal("2.0")  # Z-score threshold
        self._minimum_agreement_threshold = Decimal("0.6")
        self._confidence_weight_factor = Decimal("0.3")

    def calculate_consensus(
        self,
        results: List[EvaluationResult],
        method: str = "weighted_average",
        exclude_outliers: bool = True,
        confidence_weighting: bool = True,
    ) -> ConsensusResult:
        """Calculate consensus from multiple judge results."""
        if len(results) < 2:
            raise InsufficientDataError("Consensus requires at least 2 judge results")

        # Validate all results are successful
        successful_results = [r for r in results if r.is_successful()]
        if len(successful_results) < 2:
            raise InsufficientDataError("Consensus requires at least 2 successful evaluations")

        try:
            # Extract scores and judge information
            judge_scores = {}
            judge_confidences = {}

            for result in successful_results:
                judge_id = result.judge_id
                judge_scores[judge_id] = result.overall_score
                judge_confidences[judge_id] = result.confidence_score

            # Detect outliers if requested
            outlier_judges = []
            if exclude_outliers and len(judge_scores) >= 3:
                outlier_judges = self.detect_outliers(successful_results)

            # Calculate judge weights
            judge_weights = self._calculate_judge_weights(
                judge_scores, judge_confidences, confidence_weighting, outlier_judges
            )

            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(
                judge_scores, judge_weights, method, outlier_judges
            )

            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                judge_scores, judge_weights, outlier_judges
            )

            # Calculate agreement level
            agreement_level = self._calculate_agreement_level(judge_scores, outlier_judges)

            # Calculate statistical significance
            statistical_significance = self._calculate_statistical_significance(
                judge_scores, outlier_judges
            )

            # Create consensus result
            return ConsensusResult(
                consensus_score=consensus_score,
                confidence_interval=confidence_interval,
                agreement_level=agreement_level,
                judge_scores=judge_scores,
                judge_weights=judge_weights,
                outlier_judges=outlier_judges,
                statistical_significance=statistical_significance,
                consensus_method=method,
            )

        except Exception as e:
            raise ConsensusCalculationError(f"Failed to calculate consensus: {str(e)}")

    def detect_outliers(self, results: List[EvaluationResult]) -> List[str]:
        """Identify outlier judgments using statistical methods."""
        if len(results) < 3:
            return []  # Need at least 3 data points for outlier detection

        try:
            scores = [float(result.overall_score) for result in results]
            judge_ids = [result.judge_id for result in results]

            # Calculate mean and standard deviation
            mean_score = statistics.mean(scores)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0

            outliers = []

            if std_dev > 0:
                # Z-score based outlier detection
                for i, score in enumerate(scores):
                    z_score = abs(score - mean_score) / std_dev
                    if z_score > float(self._outlier_detection_threshold):
                        outliers.append(judge_ids[i])

            # IQR-based outlier detection as backup
            if not outliers and len(scores) >= 5:
                outliers.extend(self._detect_outliers_iqr(scores, judge_ids))

            return outliers

        except Exception as e:
            # If outlier detection fails, return empty list (no outliers)
            return []

    def _detect_outliers_iqr(self, scores: List[float], judge_ids: List[str]) -> List[str]:
        """Detect outliers using Interquartile Range method."""
        sorted_scores = sorted(scores)
        n = len(sorted_scores)

        # Calculate quartiles
        q1_index = n // 4
        q3_index = 3 * n // 4

        q1 = sorted_scores[q1_index]
        q3 = sorted_scores[q3_index]

        iqr = q3 - q1
        if iqr == 0:
            return []  # No variation in scores

        # Calculate outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify outliers
        outliers = []
        for i, score in enumerate(scores):
            if score < lower_bound or score > upper_bound:
                outliers.append(judge_ids[i])

        return outliers

    def calculate_agreement(
        self, results: List[EvaluationResult], exclude_outliers: bool = True
    ) -> Decimal:
        """Calculate inter-judge agreement score using Krippendorff's alpha or similar."""
        if len(results) < 2:
            raise InsufficientDataError("Agreement calculation requires at least 2 results")

        try:
            successful_results = [r for r in results if r.is_successful()]
            if len(successful_results) < 2:
                raise InsufficientDataError(
                    "Agreement calculation requires at least 2 successful results"
                )

            # Detect outliers if requested
            outlier_judges = []
            if exclude_outliers and len(successful_results) >= 3:
                outlier_judges = self.detect_outliers(successful_results)

            # Filter out outliers
            filtered_results = [r for r in successful_results if r.judge_id not in outlier_judges]

            if len(filtered_results) < 2:
                # If too many outliers, use original results
                filtered_results = successful_results

            # Calculate agreement using coefficient of variation approach
            scores = [float(result.overall_score) for result in filtered_results]

            if len(set(scores)) == 1:
                return Decimal("1.0")  # Perfect agreement

            mean_score = statistics.mean(scores)
            std_dev = statistics.stdev(scores)

            if mean_score == 0:
                return Decimal("0.0")

            # Coefficient of variation (lower is better agreement)
            cv = std_dev / mean_score

            # Convert to agreement score (higher is better)
            # Agreement = 1 - normalized_cv, capped at [0, 1]
            agreement = max(Decimal("0"), 1 - Decimal(str(cv)))

            return agreement.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        except Exception as e:
            raise ConsensusCalculationError(f"Failed to calculate agreement: {str(e)}")

    def _calculate_judge_weights(
        self,
        judge_scores: Dict[str, Decimal],
        judge_confidences: Dict[str, Decimal],
        confidence_weighting: bool,
        outlier_judges: List[str],
    ) -> Dict[str, Decimal]:
        """Calculate weights for each judge based on confidence and outlier status."""
        if not confidence_weighting:
            # Equal weights for non-outlier judges
            active_judges = [j for j in judge_scores.keys() if j not in outlier_judges]
            if not active_judges:
                active_judges = list(judge_scores.keys())

            equal_weight = Decimal("1.0") / len(active_judges)
            weights = {}

            for judge_id in judge_scores.keys():
                if judge_id in active_judges:
                    weights[judge_id] = equal_weight
                else:
                    weights[judge_id] = Decimal("0.0")  # Outliers get zero weight

            return weights

        # Confidence-weighted approach
        weights = {}
        total_confidence = Decimal("0")

        for judge_id in judge_scores.keys():
            if judge_id in outlier_judges:
                weights[judge_id] = Decimal("0.0")
            else:
                confidence = judge_confidences.get(judge_id, Decimal("0.5"))
                # Apply confidence weighting with factor
                weighted_confidence = (
                    Decimal("0.5") + (confidence - Decimal("0.5")) * self._confidence_weight_factor
                )
                weights[judge_id] = max(Decimal("0.1"), weighted_confidence)  # Minimum weight
                total_confidence += weights[judge_id]

        # Normalize weights to sum to 1
        if total_confidence > 0:
            for judge_id in weights:
                weights[judge_id] = (weights[judge_id] / total_confidence).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
        else:
            # Fallback to equal weights
            active_judges = [j for j in judge_scores.keys() if j not in outlier_judges]
            equal_weight = Decimal("1.0") / len(active_judges) if active_judges else Decimal("0.0")
            weights = {judge_id: equal_weight for judge_id in judge_scores.keys()}

        return weights

    def _calculate_consensus_score(
        self,
        judge_scores: Dict[str, Decimal],
        judge_weights: Dict[str, Decimal],
        method: str,
        outlier_judges: List[str],
    ) -> Decimal:
        """Calculate consensus score using specified method."""
        # Filter out outliers
        active_scores = {
            judge_id: score
            for judge_id, score in judge_scores.items()
            if judge_id not in outlier_judges
        }

        if not active_scores:
            # If all judges are outliers, use all scores
            active_scores = judge_scores

        if method == "weighted_average":
            total_weighted_score = Decimal("0")
            total_weight = Decimal("0")

            for judge_id, score in active_scores.items():
                weight = judge_weights.get(judge_id, Decimal("0"))
                total_weighted_score += score * weight
                total_weight += weight

            if total_weight > 0:
                return (total_weighted_score / total_weight).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
            else:
                return Decimal("0")

        elif method == "median":
            scores_list = sorted(active_scores.values())
            n = len(scores_list)

            if n == 0:
                return Decimal("0")
            elif n % 2 == 0:
                median = (scores_list[n // 2 - 1] + scores_list[n // 2]) / 2
            else:
                median = scores_list[n // 2]

            return median.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        elif method == "trimmed_mean":
            # Remove top and bottom 20% of scores
            scores_list = sorted(active_scores.values())
            n = len(scores_list)

            if n <= 4:
                # Not enough data for trimming, use regular mean
                mean_score = sum(scores_list) / n if n > 0 else Decimal("0")
            else:
                trim_count = max(1, n // 5)  # Remove 20%
                trimmed_scores = scores_list[trim_count:-trim_count]
                mean_score = sum(trimmed_scores) / len(trimmed_scores)

            return mean_score.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        elif method == "robust_average":
            # Weighted average with reduced influence from extreme scores
            scores_list = list(active_scores.values())

            if len(scores_list) <= 2:
                # Use simple average for small samples
                return (sum(scores_list) / len(scores_list)).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )

            median_score = statistics.median([float(s) for s in scores_list])

            # Weight scores based on distance from median
            total_weighted_score = Decimal("0")
            total_weight = Decimal("0")

            for judge_id, score in active_scores.items():
                distance = abs(float(score) - median_score)
                # Inverse weighting - closer to median gets higher weight
                weight = Decimal("1.0") / (Decimal("1.0") + Decimal(str(distance)))

                total_weighted_score += score * weight
                total_weight += weight

            if total_weight > 0:
                return (total_weighted_score / total_weight).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
            else:
                return Decimal("0")

        else:
            raise ValidationError(f"Unknown consensus method: {method}")

    def _calculate_confidence_interval(
        self,
        judge_scores: Dict[str, Decimal],
        judge_weights: Dict[str, Decimal],
        outlier_judges: List[str],
        confidence_level: float = 0.95,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate confidence interval for consensus score."""
        try:
            # Filter out outliers
            active_scores = [
                float(score)
                for judge_id, score in judge_scores.items()
                if judge_id not in outlier_judges
            ]

            if len(active_scores) < 2:
                # Not enough data for meaningful interval
                consensus_score = list(judge_scores.values())[0] if judge_scores else Decimal("0.5")
                margin = Decimal("0.1")  # Default margin
                return (
                    max(Decimal("0"), consensus_score - margin),
                    min(Decimal("1"), consensus_score + margin),
                )

            # Calculate mean and standard error
            mean_score = statistics.mean(active_scores)
            std_dev = statistics.stdev(active_scores) if len(active_scores) > 1 else 0.1
            std_err = std_dev / (len(active_scores) ** 0.5)

            # Calculate margin of error (using t-distribution approximation)
            # For 95% confidence, use 1.96 for large samples, adjust for small samples
            if len(active_scores) >= 30:
                t_value = 1.96
            elif len(active_scores) >= 10:
                t_value = 2.0
            else:
                t_value = 2.5  # Conservative for small samples

            margin = t_value * std_err

            lower_bound = max(0, mean_score - margin)
            upper_bound = min(1, mean_score + margin)

            return (
                Decimal(str(lower_bound)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                Decimal(str(upper_bound)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            )

        except Exception:
            # Fallback to default interval
            return (Decimal("0.4"), Decimal("0.6"))

    def _calculate_agreement_level(
        self, judge_scores: Dict[str, Decimal], outlier_judges: List[str]
    ) -> Decimal:
        """Calculate inter-judge agreement level."""
        try:
            # Filter out outliers
            active_scores = [
                float(score)
                for judge_id, score in judge_scores.items()
                if judge_id not in outlier_judges
            ]

            if len(active_scores) < 2:
                return Decimal("0.0")  # No agreement with single judge

            if len(set(active_scores)) == 1:
                return Decimal("1.0")  # Perfect agreement

            # Calculate coefficient of variation
            mean_score = statistics.mean(active_scores)
            std_dev = statistics.stdev(active_scores)

            if mean_score == 0:
                return Decimal("0.0")

            cv = std_dev / mean_score

            # Convert to agreement score (1 - normalized CV)
            # Use square root to make agreement score less sensitive to small variations
            agreement = max(0, 1 - cv**0.5)

            return Decimal(str(agreement)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        except Exception:
            return Decimal("0.5")  # Default moderate agreement

    def _calculate_statistical_significance(
        self, judge_scores: Dict[str, Decimal], outlier_judges: List[str]
    ) -> Decimal:
        """Calculate statistical significance of consensus."""
        try:
            # Filter out outliers
            active_scores = [
                float(score)
                for judge_id, score in judge_scores.items()
                if judge_id not in outlier_judges
            ]

            if len(active_scores) < 2:
                return Decimal("1.0")  # Not significant with single judge

            # Perform one-sample t-test against null hypothesis (score = 0.5)
            null_hypothesis_mean = 0.5
            sample_mean = statistics.mean(active_scores)

            if len(active_scores) == 2:
                # Simple difference test for two judges
                if abs(sample_mean - null_hypothesis_mean) > 0.2:
                    return Decimal("0.05")  # Significant
                else:
                    return Decimal("0.2")  # Not significant

            sample_std = statistics.stdev(active_scores)
            n = len(active_scores)

            if sample_std == 0:
                # Perfect agreement - highly significant if different from null
                if abs(sample_mean - null_hypothesis_mean) > 0.01:
                    return Decimal("0.001")
                else:
                    return Decimal("0.5")

            # Calculate t-statistic
            t_stat = abs(sample_mean - null_hypothesis_mean) / (sample_std / (n**0.5))

            # Convert to approximate p-value (simplified)
            if t_stat > 3.0:
                p_value = 0.001
            elif t_stat > 2.5:
                p_value = 0.01
            elif t_stat > 2.0:
                p_value = 0.05
            elif t_stat > 1.5:
                p_value = 0.1
            else:
                p_value = 0.2

            return Decimal(str(p_value))

        except Exception:
            return Decimal("0.1")  # Default moderate significance
