"""Quality control service for evaluation domain."""

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..entities.evaluation_result import EvaluationResult
from ..entities.evaluation_template import EvaluationTemplate
from ..exceptions import QualityControlError, ValidationError
from ..value_objects.consensus_result import ConsensusResult
from ..value_objects.quality_report import (
    QualityIssue,
    QualityIssueType,
    QualityLevel,
    QualityReport,
)


class QualityController:
    """Service for quality assurance in evaluations."""

    def __init__(self):
        """Initialize quality controller with configurable thresholds."""
        # Quality thresholds
        self.min_reasoning_length = 50
        self.min_confidence_threshold = Decimal("0.3")
        self.max_confidence_threshold = Decimal("0.95")
        self.score_consistency_threshold = Decimal("0.3")  # Max deviation from reasoning sentiment
        self.outlier_detection_threshold = Decimal("2.0")  # Z-score threshold

        # Pattern detection
        self.suspicious_patterns = [
            r"I don't know",
            r"cannot evaluate",
            r"unable to assess",
            r"not enough information",
            r"unclear prompt",
            r"ambiguous request",
        ]

        # Bias detection keywords
        self.bias_indicators = {
            "length_bias": [r"too short", r"too long", r"brief", r"lengthy"],
            "complexity_bias": [r"too simple", r"too complex", r"basic", r"advanced"],
            "style_bias": [r"informal", r"formal", r"casual", r"professional"],
            "sentiment_bias": [r"negative", r"positive", r"pessimistic", r"optimistic"],
        }

    def validate_evaluation_quality(
        self,
        result: EvaluationResult,
        template: EvaluationTemplate,
        evaluator_id: str = "quality_controller",
    ) -> QualityReport:
        """Validate individual evaluation quality."""
        if not result.is_successful():
            return QualityReport.create_failed(
                quality_score=Decimal("0.0"),
                issues=[
                    QualityIssue(
                        issue_type=QualityIssueType.TEMPLATE_ERROR,
                        severity="high",
                        description="Evaluation failed or incomplete",
                        affected_component="evaluation_result",
                    )
                ],
                evaluator_id=evaluator_id,
            )

        try:
            issues = []
            metrics = {}

            # Check reasoning quality
            reasoning_issues, reasoning_metrics = self._check_reasoning_quality(result)
            issues.extend(reasoning_issues)
            metrics.update(reasoning_metrics)

            # Check score consistency
            consistency_issues, consistency_metrics = self._check_score_consistency(
                result, template
            )
            issues.extend(consistency_issues)
            metrics.update(consistency_metrics)

            # Check confidence levels
            confidence_issues, confidence_metrics = self._check_confidence_levels(result)
            issues.extend(confidence_issues)
            metrics.update(confidence_metrics)

            # Check for suspicious patterns
            pattern_issues = self._check_suspicious_patterns(result)
            issues.extend(pattern_issues)

            # Check for potential bias
            bias_issues = self._check_bias_indicators(result)
            issues.extend(bias_issues)

            # Calculate overall quality score
            quality_score = self._calculate_quality_score(issues, metrics)

            # Create quality report
            if quality_score >= Decimal("0.7") and not any(
                issue.severity == "high" for issue in issues
            ):
                return QualityReport.create_passed(
                    quality_score=quality_score, evaluator_id=evaluator_id, metrics=metrics
                )
            else:
                return QualityReport.create_failed(
                    quality_score=quality_score,
                    issues=issues,
                    evaluator_id=evaluator_id,
                    metrics=metrics,
                )

        except Exception as e:
            raise QualityControlError(f"Quality validation failed: {str(e)}")

    def check_consensus_quality(
        self, consensus: ConsensusResult, evaluator_id: str = "quality_controller"
    ) -> QualityReport:
        """Validate consensus quality."""
        try:
            issues = []
            metrics = {}

            # Check agreement levels
            if consensus.agreement_level < Decimal("0.6"):
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.STATISTICAL_ANOMALY,
                        severity="medium",
                        description=f"Low inter-judge agreement: {consensus.agreement_level}",
                        affected_component="consensus",
                        suggested_action="Review individual evaluations for consistency",
                    )
                )

            # Check statistical significance
            if not consensus.is_statistically_significant():
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.STATISTICAL_ANOMALY,
                        severity="medium",
                        description=f"Consensus not statistically significant: p={consensus.statistical_significance}",
                        affected_component="consensus",
                        suggested_action="Increase number of judges or review evaluation criteria",
                    )
                )

            # Check for outliers
            if consensus.has_outliers():
                outlier_count = len(consensus.outlier_judges)
                total_judges = len(consensus.judge_scores)
                outlier_ratio = outlier_count / total_judges

                if outlier_ratio > 0.3:  # More than 30% outliers
                    issues.append(
                        QualityIssue(
                            issue_type=QualityIssueType.OUTLIER_SCORE,
                            severity="high",
                            description=f"High outlier ratio: {outlier_ratio:.2%} ({outlier_count}/{total_judges})",
                            affected_component="consensus",
                            suggested_action="Review outlier judges for calibration issues",
                        )
                    )
                else:
                    issues.append(
                        QualityIssue(
                            issue_type=QualityIssueType.OUTLIER_SCORE,
                            severity="low",
                            description=f"Outlier judges detected: {outlier_count}",
                            affected_component="consensus",
                            suggested_action="Monitor outlier judges for patterns",
                        )
                    )

            # Check confidence interval width
            ci_width = consensus.get_confidence_width()
            if ci_width > Decimal("0.3"):
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.LOW_CONFIDENCE,
                        severity="medium",
                        description=f"Wide confidence interval: {ci_width}",
                        affected_component="consensus",
                        suggested_action="Increase number of judges for more precise consensus",
                    )
                )

            # Add metrics
            metrics.update(
                {
                    "agreement_level": consensus.agreement_level,
                    "statistical_significance": consensus.statistical_significance,
                    "confidence_interval_width": ci_width,
                    "outlier_ratio": Decimal(
                        str(len(consensus.outlier_judges) / len(consensus.judge_scores))
                    ),
                    "effective_judges_count": Decimal(str(consensus.get_effective_judges_count())),
                }
            )

            # Calculate quality score
            quality_score = self._calculate_consensus_quality_score(consensus, issues)

            # Create quality report
            if quality_score >= Decimal("0.7") and not any(
                issue.severity == "high" for issue in issues
            ):
                return QualityReport.create_passed(
                    quality_score=quality_score, evaluator_id=evaluator_id, metrics=metrics
                )
            else:
                return QualityReport.create_failed(
                    quality_score=quality_score,
                    issues=issues,
                    evaluator_id=evaluator_id,
                    metrics=metrics,
                )

        except Exception as e:
            raise QualityControlError(f"Consensus quality check failed: {str(e)}")

    def _check_reasoning_quality(
        self, result: EvaluationResult
    ) -> tuple[List[QualityIssue], Dict[str, Decimal]]:
        """Check quality of reasoning provided."""
        issues = []
        metrics = {}

        reasoning_length = len(result.reasoning.strip())
        metrics["reasoning_length"] = Decimal(str(reasoning_length))

        # Check minimum reasoning length
        if reasoning_length < self.min_reasoning_length:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.REASONING_INCOMPLETE,
                    severity="medium",
                    description=f"Reasoning too brief: {reasoning_length} characters",
                    affected_component="reasoning",
                    suggested_action="Request more detailed explanations from judge",
                )
            )

        # Check for empty or generic reasoning
        generic_phrases = ["good response", "bad response", "okay", "fine", "not bad", "decent"]

        reasoning_lower = result.reasoning.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in reasoning_lower)

        if generic_count >= 2:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.REASONING_INCOMPLETE,
                    severity="medium",
                    description="Reasoning contains generic or vague language",
                    affected_component="reasoning",
                    suggested_action="Encourage more specific and detailed evaluations",
                )
            )

        # Check for reasoning structure
        has_structured_reasoning = any(
            [
                "because" in reasoning_lower,
                "however" in reasoning_lower,
                "therefore" in reasoning_lower,
                "although" in reasoning_lower,
                "specifically" in reasoning_lower,
                "for example" in reasoning_lower,
            ]
        )

        metrics["has_structured_reasoning"] = Decimal("1.0" if has_structured_reasoning else "0.0")

        if not has_structured_reasoning and reasoning_length > self.min_reasoning_length:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.REASONING_INCOMPLETE,
                    severity="low",
                    description="Reasoning lacks structured argumentation",
                    affected_component="reasoning",
                    suggested_action="Encourage use of logical connectors and examples",
                )
            )

        return issues, metrics

    def _check_score_consistency(
        self, result: EvaluationResult, template: EvaluationTemplate
    ) -> tuple[List[QualityIssue], Dict[str, Decimal]]:
        """Check consistency between scores and reasoning."""
        issues = []
        metrics = {}

        # Simple sentiment analysis of reasoning
        positive_words = ["good", "excellent", "clear", "accurate", "helpful", "useful", "well"]
        negative_words = ["poor", "bad", "unclear", "inaccurate", "unhelpful", "useless", "poorly"]

        reasoning_lower = result.reasoning.lower()
        positive_count = sum(1 for word in positive_words if word in reasoning_lower)
        negative_count = sum(1 for word in negative_words if word in reasoning_lower)

        # Calculate sentiment score (-1 to +1)
        if positive_count + negative_count > 0:
            sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            sentiment_score = 0.0

        # Normalize overall score to -1 to +1 range for comparison
        normalized_overall_score = (float(result.overall_score) - 0.5) * 2

        # Check consistency
        consistency_deviation = abs(sentiment_score - normalized_overall_score)
        metrics["sentiment_score"] = Decimal(str(sentiment_score))
        metrics["consistency_deviation"] = Decimal(str(consistency_deviation))

        if consistency_deviation > float(self.score_consistency_threshold):
            severity = "high" if consistency_deviation > 0.6 else "medium"
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.SCORE_INCONSISTENT,
                    severity=severity,
                    description=f"Score-reasoning inconsistency: deviation {consistency_deviation:.2f}",
                    affected_component="scoring",
                    suggested_action="Review alignment between scores and written reasoning",
                )
            )

        # Check dimension score spread
        if result.dimension_scores:
            scores = list(result.dimension_scores.values())
            if len(set(scores)) == 1 and len(scores) > 2:
                # All dimension scores are identical - potentially lazy evaluation
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.SUSPICIOUS_PATTERN,
                        severity="medium",
                        description="All dimension scores are identical",
                        affected_component="dimension_scoring",
                        suggested_action="Encourage differentiated scoring across dimensions",
                    )
                )

        return issues, metrics

    def _check_confidence_levels(
        self, result: EvaluationResult
    ) -> tuple[List[QualityIssue], Dict[str, Decimal]]:
        """Check appropriateness of confidence levels."""
        issues = []
        metrics = {}

        confidence = result.confidence_score
        metrics["confidence_score"] = confidence

        # Check for unrealistically low confidence
        if confidence < self.min_confidence_threshold:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.LOW_CONFIDENCE,
                    severity="medium",
                    description=f"Extremely low confidence: {confidence}",
                    affected_component="confidence",
                    suggested_action="Investigate evaluation difficulty or judge calibration",
                )
            )

        # Check for unrealistically high confidence
        if confidence > self.max_confidence_threshold:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.SUSPICIOUS_PATTERN,
                    severity="low",
                    description=f"Unrealistically high confidence: {confidence}",
                    affected_component="confidence",
                    suggested_action="Monitor for overconfidence patterns",
                )
            )

        # Check confidence-reasoning alignment
        reasoning_length = len(result.reasoning.strip())
        expected_confidence_from_length = min(
            Decimal("0.9"), Decimal("0.3") + (Decimal(str(reasoning_length)) / Decimal("200"))
        )

        confidence_length_deviation = abs(confidence - expected_confidence_from_length)
        metrics["confidence_length_deviation"] = confidence_length_deviation

        if confidence_length_deviation > Decimal("0.4"):
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.LOW_CONFIDENCE,
                    severity="low",
                    description="Confidence level inconsistent with reasoning detail",
                    affected_component="confidence",
                    suggested_action="Review confidence calibration",
                )
            )

        return issues, metrics

    def _check_suspicious_patterns(self, result: EvaluationResult) -> List[QualityIssue]:
        """Check for suspicious patterns in evaluation."""
        issues = []

        reasoning_lower = result.reasoning.lower()

        for pattern in self.suspicious_patterns:
            if re.search(pattern, reasoning_lower):
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.SUSPICIOUS_PATTERN,
                        severity="medium",
                        description=f"Suspicious pattern detected: '{pattern}'",
                        affected_component="reasoning",
                        suggested_action="Review evaluation prompt clarity and judge instructions",
                    )
                )

        return issues

    def _check_bias_indicators(self, result: EvaluationResult) -> List[QualityIssue]:
        """Check for potential bias indicators."""
        issues = []

        reasoning_lower = result.reasoning.lower()

        for bias_type, indicators in self.bias_indicators.items():
            bias_count = sum(1 for indicator in indicators if re.search(indicator, reasoning_lower))

            if bias_count >= 2:  # Multiple indicators of same bias type
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.BIAS_DETECTED,
                        severity="low",
                        description=f"Potential {bias_type} detected",
                        affected_component="reasoning",
                        suggested_action="Monitor judge for systematic bias patterns",
                        metadata={"bias_type": bias_type, "indicator_count": bias_count},
                    )
                )

        return issues

    def _calculate_quality_score(
        self, issues: List[QualityIssue], metrics: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate overall quality score from issues and metrics."""
        # Start with perfect score
        base_score = Decimal("1.0")

        # Deduct points for issues
        for issue in issues:
            if issue.severity == "high":
                base_score -= Decimal("0.3")
            elif issue.severity == "medium":
                base_score -= Decimal("0.15")
            elif issue.severity == "low":
                base_score -= Decimal("0.05")

        # Bonus for good metrics
        if "reasoning_length" in metrics:
            length = metrics["reasoning_length"]
            if length >= 100:  # Good detail
                base_score += Decimal("0.1")
            elif length >= 200:  # Excellent detail
                base_score += Decimal("0.2")

        if "has_structured_reasoning" in metrics:
            if metrics["has_structured_reasoning"] == Decimal("1.0"):
                base_score += Decimal("0.1")

        # Ensure score is within bounds
        final_score = max(Decimal("0.0"), min(Decimal("1.0"), base_score))

        return final_score.quantize(Decimal("0.001"))

    def _calculate_consensus_quality_score(
        self, consensus: ConsensusResult, issues: List[QualityIssue]
    ) -> Decimal:
        """Calculate quality score for consensus result."""
        # Start with agreement level as base score
        base_score = consensus.agreement_level

        # Add statistical significance bonus
        if consensus.is_statistically_significant():
            base_score += Decimal("0.2")

        # Add precision bonus for narrow confidence interval
        if consensus.is_consensus_precise():
            base_score += Decimal("0.1")

        # Deduct for issues
        for issue in issues:
            if issue.severity == "high":
                base_score -= Decimal("0.3")
            elif issue.severity == "medium":
                base_score -= Decimal("0.15")
            elif issue.severity == "low":
                base_score -= Decimal("0.05")

        # Ensure score is within bounds
        final_score = max(Decimal("0.0"), min(Decimal("1.0"), base_score))

        return final_score.quantize(Decimal("0.001"))
