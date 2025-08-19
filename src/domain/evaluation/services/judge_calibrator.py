"""Judge calibration service for evaluation domain."""

import statistics
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ..entities.evaluation_result import EvaluationResult
from ..entities.judge import Judge
from ..exceptions import CalibrationError, InsufficientDataError, ValidationError
from ..value_objects.calibration_data import CalibrationData


class JudgeCalibrator:
    """Service for calibrating and monitoring judge performance."""

    def __init__(self):
        """Initialize judge calibrator with configuration."""
        # Calibration requirements
        self.minimum_calibration_samples = 10
        self.minimum_production_samples = 50
        self.recalibration_threshold_days = 30

        # Performance thresholds
        self.minimum_accuracy = Decimal("0.8")
        self.minimum_consistency = Decimal("0.75")
        self.maximum_bias = Decimal("0.2")
        self.minimum_confidence_calibration = Decimal("0.7")

        # Drift detection thresholds
        self.drift_detection_window = 100  # Recent samples to check for drift
        self.accuracy_drift_threshold = Decimal("0.1")  # 10% drop triggers drift alert
        self.consistency_drift_threshold = Decimal("0.1")
        self.bias_drift_threshold = Decimal("0.15")

    async def calibrate_judge(
        self,
        judge: Judge,
        golden_standard_results: List[Tuple[EvaluationResult, Decimal]],
        recalibration: bool = False,
    ) -> CalibrationData:
        """Calibrate judge against golden standard evaluations."""
        if len(golden_standard_results) < self.minimum_calibration_samples:
            raise InsufficientDataError(
                f"Calibration requires at least {self.minimum_calibration_samples} samples, "
                f"got {len(golden_standard_results)}"
            )

        try:
            # Validate inputs
            judge_results = []
            golden_scores = []

            for result, golden_score in golden_standard_results:
                if not result.is_successful():
                    continue

                if not (0 <= golden_score <= 1):
                    raise ValidationError(
                        f"Golden standard score must be between 0 and 1, got {golden_score}"
                    )

                judge_results.append(result)
                golden_scores.append(golden_score)

            if len(judge_results) < self.minimum_calibration_samples:
                raise InsufficientDataError(
                    f"Only {len(judge_results)} successful results available for calibration"
                )

            # Calculate calibration metrics
            accuracy = self._calculate_accuracy(judge_results, golden_scores)
            consistency = self._calculate_consistency(judge_results)
            bias_score = self._calculate_bias(judge_results, golden_scores)
            confidence_calibration = self._calculate_confidence_calibration(
                judge_results, golden_scores
            )

            # Create golden standard scores dictionary
            golden_standard_scores = {}
            for i, result in enumerate(judge_results):
                golden_standard_scores[str(result.result_id)] = golden_scores[i]

            # Create performance metrics
            performance_metrics = {
                "calibration_samples": len(judge_results),
                "recalibration": recalibration,
                "evaluation_time_avg": statistics.mean(
                    [r.evaluation_time_ms for r in judge_results]
                ),
                "confidence_avg": statistics.mean(
                    [float(r.confidence_score) for r in judge_results]
                ),
                "score_variance": statistics.variance(
                    [float(r.overall_score) for r in judge_results]
                ),
                "golden_score_variance": statistics.variance(
                    [float(score) for score in golden_scores]
                ),
            }

            # Create calibration data
            calibration_data = CalibrationData(
                accuracy=accuracy,
                consistency=consistency,
                bias_score=bias_score,
                confidence_calibration=confidence_calibration,
                sample_size=len(judge_results),
                calibrated_at=datetime.utcnow(),
                golden_standard_scores=golden_standard_scores,
                performance_metrics=performance_metrics,
            )

            # Update judge with calibration data
            judge.calibrate(calibration_data)

            return calibration_data

        except Exception as e:
            if isinstance(e, (InsufficientDataError, ValidationError)):
                raise
            raise CalibrationError(f"Judge calibration failed: {str(e)}")

    def check_calibration_drift(
        self, judge: Judge, recent_results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Check for calibration drift in recent judge performance."""
        if not judge.calibration_data:
            return {"has_drift": False, "reason": "Judge not calibrated"}

        if len(recent_results) < 10:
            return {"has_drift": False, "reason": "Insufficient recent data"}

        try:
            # Filter successful results
            successful_results = [r for r in recent_results if r.is_successful()]
            if len(successful_results) < 10:
                return {"has_drift": False, "reason": "Insufficient successful results"}

            # Calculate recent performance metrics
            recent_metrics = self._calculate_recent_performance(successful_results)

            # Compare with baseline calibration
            drift_indicators = []

            # Check accuracy drift (if golden standard available)
            if "accuracy_estimate" in recent_metrics:
                accuracy_drop = (
                    judge.calibration_data.accuracy - recent_metrics["accuracy_estimate"]
                )
                if accuracy_drop > self.accuracy_drift_threshold:
                    drift_indicators.append(
                        {
                            "type": "accuracy_drift",
                            "baseline": str(judge.calibration_data.accuracy),
                            "recent": str(recent_metrics["accuracy_estimate"]),
                            "drop": str(accuracy_drop),
                        }
                    )

            # Check consistency drift
            consistency_drop = judge.calibration_data.consistency - recent_metrics["consistency"]
            if consistency_drop > self.consistency_drift_threshold:
                drift_indicators.append(
                    {
                        "type": "consistency_drift",
                        "baseline": str(judge.calibration_data.consistency),
                        "recent": str(recent_metrics["consistency"]),
                        "drop": str(consistency_drop),
                    }
                )

            # Check bias drift
            bias_change = abs(judge.calibration_data.bias_score - recent_metrics["bias_estimate"])
            if bias_change > self.bias_drift_threshold:
                drift_indicators.append(
                    {
                        "type": "bias_drift",
                        "baseline": str(judge.calibration_data.bias_score),
                        "recent": str(recent_metrics["bias_estimate"]),
                        "change": str(bias_change),
                    }
                )

            # Check calibration age
            calibration_age_days = (datetime.utcnow() - judge.calibration_data.calibrated_at).days
            if calibration_age_days > self.recalibration_threshold_days:
                drift_indicators.append(
                    {
                        "type": "stale_calibration",
                        "age_days": calibration_age_days,
                        "threshold_days": self.recalibration_threshold_days,
                    }
                )

            return {
                "has_drift": len(drift_indicators) > 0,
                "drift_indicators": drift_indicators,
                "recent_metrics": recent_metrics,
                "calibration_age_days": calibration_age_days,
                "needs_recalibration": len(drift_indicators) > 1,  # Multiple drift indicators
            }

        except Exception as e:
            return {"has_drift": False, "error": str(e)}

    def recommend_recalibration(
        self, judge: Judge, recent_results: Optional[List[EvaluationResult]] = None
    ) -> Dict[str, Any]:
        """Recommend whether judge needs recalibration."""
        recommendations = {
            "needs_recalibration": False,
            "reasons": [],
            "priority": "low",
            "recommended_samples": self.minimum_calibration_samples,
        }

        # Check if judge is calibrated at all
        if not judge.calibration_data:
            recommendations.update(
                {
                    "needs_recalibration": True,
                    "reasons": ["Judge has never been calibrated"],
                    "priority": "high",
                    "recommended_samples": self.minimum_production_samples,
                }
            )
            return recommendations

        # Check calibration quality
        if not judge.calibration_data.is_production_ready():
            recommendations["needs_recalibration"] = True
            recommendations["reasons"].append("Current calibration below production standards")
            recommendations["priority"] = "high"
            recommendations["recommended_samples"] = self.minimum_production_samples

        # Check calibration age
        if judge.calibration_data.needs_recalibration():
            recommendations["needs_recalibration"] = True
            recommendations["reasons"].append("Calibration data is stale")
            if recommendations["priority"] == "low":
                recommendations["priority"] = "medium"

        # Check for drift if recent results provided
        if recent_results:
            drift_check = self.check_calibration_drift(judge, recent_results)
            if drift_check.get("has_drift", False):
                recommendations["needs_recalibration"] = True
                recommendations["reasons"].append("Performance drift detected")
                recommendations["priority"] = "high"

                # Add drift details
                recommendations["drift_details"] = drift_check.get("drift_indicators", [])

        # Check sample size adequacy
        if judge.calibration_data.sample_size < self.minimum_production_samples:
            recommendations["reasons"].append("Insufficient calibration sample size")
            if not recommendations["needs_recalibration"]:
                recommendations["needs_recalibration"] = True
                recommendations["priority"] = "medium"
            recommendations["recommended_samples"] = self.minimum_production_samples

        return recommendations

    def _calculate_accuracy(
        self, judge_results: List[EvaluationResult], golden_scores: List[Decimal]
    ) -> Decimal:
        """Calculate judge accuracy against golden standard."""
        if len(judge_results) != len(golden_scores):
            raise ValidationError("Mismatched result and golden score counts")

        # Calculate Mean Absolute Error (MAE)
        absolute_errors = []
        for i, result in enumerate(judge_results):
            error = abs(result.overall_score - golden_scores[i])
            absolute_errors.append(float(error))

        mae = statistics.mean(absolute_errors)

        # Convert MAE to accuracy score (1 - normalized_mae)
        # Assuming maximum possible error is 1.0 (full scale)
        accuracy = 1.0 - mae

        return Decimal(str(max(0.0, accuracy))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _calculate_consistency(self, judge_results: List[EvaluationResult]) -> Decimal:
        """Calculate judge consistency (inverse of score variance)."""
        if len(judge_results) < 2:
            return Decimal("0.5")  # Default for insufficient data

        scores = [float(result.overall_score) for result in judge_results]

        # Calculate coefficient of variation
        if statistics.mean(scores) == 0:
            return Decimal("1.0")  # Perfect consistency if all scores are 0

        cv = statistics.stdev(scores) / statistics.mean(scores)

        # Convert to consistency score (lower variance = higher consistency)
        # Use exponential decay to map CV to consistency score
        consistency = max(0.0, 1.0 - cv)

        return Decimal(str(consistency)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _calculate_bias(
        self, judge_results: List[EvaluationResult], golden_scores: List[Decimal]
    ) -> Decimal:
        """Calculate systematic bias in judge scoring."""
        if len(judge_results) != len(golden_scores):
            raise ValidationError("Mismatched result and golden score counts")

        # Calculate signed differences
        differences = []
        for i, result in enumerate(judge_results):
            diff = float(result.overall_score - golden_scores[i])
            differences.append(diff)

        # Calculate mean bias
        mean_bias = statistics.mean(differences)

        return Decimal(str(mean_bias)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _calculate_confidence_calibration(
        self, judge_results: List[EvaluationResult], golden_scores: List[Decimal]
    ) -> Decimal:
        """Calculate how well judge confidence matches actual accuracy."""
        if len(judge_results) != len(golden_scores):
            raise ValidationError("Mismatched result and golden score counts")

        # Group predictions by confidence bins
        confidence_bins = {}
        for i, result in enumerate(judge_results):
            confidence = result.confidence_score
            accuracy = 1 - abs(result.overall_score - golden_scores[i])

            # Round confidence to nearest 0.1 for binning
            conf_bin = round(float(confidence) * 10) / 10

            if conf_bin not in confidence_bins:
                confidence_bins[conf_bin] = []
            confidence_bins[conf_bin].append(float(accuracy))

        # Calculate calibration error
        calibration_errors = []
        for conf_level, accuracies in confidence_bins.items():
            if len(accuracies) >= 2:  # Need multiple samples for reliable estimate
                actual_accuracy = statistics.mean(accuracies)
                calibration_error = abs(conf_level - actual_accuracy)
                calibration_errors.append(calibration_error)

        if not calibration_errors:
            return Decimal("0.5")  # Default for insufficient data

        # Calculate mean calibration error
        mean_calibration_error = statistics.mean(calibration_errors)

        # Convert to calibration quality score
        calibration_quality = max(0.0, 1.0 - mean_calibration_error)

        return Decimal(str(calibration_quality)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _calculate_recent_performance(
        self, recent_results: List[EvaluationResult]
    ) -> Dict[str, Decimal]:
        """Calculate performance metrics for recent results."""
        if not recent_results:
            return {}

        metrics = {}

        # Calculate consistency
        scores = [float(result.overall_score) for result in recent_results]
        if len(scores) > 1:
            mean_score = statistics.mean(scores)
            if mean_score > 0:
                cv = statistics.stdev(scores) / mean_score
                consistency = max(0.0, 1.0 - cv)
                metrics["consistency"] = Decimal(str(consistency)).quantize(Decimal("0.001"))

        # Estimate bias (using assumption that mean score should be around 0.5)
        if scores:
            mean_score = statistics.mean(scores)
            estimated_bias = mean_score - 0.5  # Neutral expectation
            metrics["bias_estimate"] = Decimal(str(estimated_bias)).quantize(Decimal("0.001"))

        # Calculate average confidence
        confidences = [float(result.confidence_score) for result in recent_results]
        if confidences:
            avg_confidence = statistics.mean(confidences)
            metrics["average_confidence"] = Decimal(str(avg_confidence)).quantize(Decimal("0.001"))

        # Calculate evaluation time statistics
        eval_times = [result.evaluation_time_ms for result in recent_results]
        if eval_times:
            metrics["average_eval_time_ms"] = Decimal(str(statistics.mean(eval_times)))
            metrics["eval_time_variance"] = Decimal(str(statistics.variance(eval_times)))

        return metrics

    def get_calibration_summary(self, judge: Judge) -> Dict[str, Any]:
        """Get comprehensive calibration summary for judge."""
        if not judge.calibration_data:
            return {
                "is_calibrated": False,
                "status": "uncalibrated",
                "message": "Judge has never been calibrated",
            }

        calibration = judge.calibration_data

        summary = {
            "is_calibrated": judge.is_calibrated(),
            "is_production_ready": calibration.is_production_ready(),
            "quality_grade": calibration.get_quality_grade(),
            "reliability_score": str(calibration.get_reliability_score()),
            "calibrated_at": calibration.calibrated_at.isoformat(),
            "sample_size": calibration.sample_size,
            "metrics": {
                "accuracy": str(calibration.accuracy),
                "consistency": str(calibration.consistency),
                "bias_score": str(calibration.bias_score),
                "confidence_calibration": str(calibration.confidence_calibration),
            },
            "drift_indicators": calibration.get_drift_indicators(),
            "needs_recalibration": calibration.needs_recalibration(),
            "performance_metrics": calibration.performance_metrics,
        }

        # Add recommendations
        recommendations = self.recommend_recalibration(judge)
        summary["recalibration_recommendation"] = recommendations

        return summary
