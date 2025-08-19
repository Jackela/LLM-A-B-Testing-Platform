"""Tests for evaluation domain services."""

import statistics
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.domain.evaluation.entities.dimension import Dimension
from src.domain.evaluation.entities.evaluation_result import EvaluationResult
from src.domain.evaluation.entities.evaluation_template import EvaluationTemplate
from src.domain.evaluation.entities.judge import Judge
from src.domain.evaluation.exceptions import (
    CalibrationError,
    ConsensusCalculationError,
    InsufficientDataError,
    QualityControlError,
)
from src.domain.evaluation.services.consensus_algorithm import ConsensusAlgorithm
from src.domain.evaluation.services.judge_calibrator import JudgeCalibrator
from src.domain.evaluation.services.quality_controller import QualityController
from src.domain.evaluation.value_objects.calibration_data import CalibrationData
from src.domain.evaluation.value_objects.consensus_result import ConsensusResult
from src.domain.evaluation.value_objects.quality_report import QualityIssueType, QualityLevel
from src.domain.evaluation.value_objects.scoring_scale import ScoringScale


class TestConsensusAlgorithm:
    """Test cases for ConsensusAlgorithm service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.algorithm = ConsensusAlgorithm()

        # Create mock evaluation results
        self.results = []
        for i, (judge_id, score, confidence) in enumerate(
            [
                ("judge_1", Decimal("0.8"), Decimal("0.9")),
                ("judge_2", Decimal("0.7"), Decimal("0.8")),
                ("judge_3", Decimal("0.9"), Decimal("0.85")),
            ]
        ):
            result = EvaluationResult.create_pending(
                judge_id=judge_id,
                template_id=uuid4(),
                prompt="Test prompt",
                response="Test response",
            )

            # Complete the result
            result.overall_score = score
            result.confidence_score = confidence
            result.completed_at = datetime.utcnow()
            result.dimension_scores = {"test": 4}
            result.reasoning = f"Test reasoning from {judge_id}"
            result.evaluation_time_ms = 1000

            self.results.append(result)

    def test_calculate_consensus_basic(self):
        """Test basic consensus calculation."""
        consensus = self.algorithm.calculate_consensus(self.results)

        assert isinstance(consensus, ConsensusResult)
        assert len(consensus.judge_scores) == 3
        assert Decimal("0.7") <= consensus.consensus_score <= Decimal("0.9")
        assert consensus.consensus_method == "weighted_average"

    def test_calculate_consensus_different_methods(self):
        """Test consensus calculation with different methods."""
        # Weighted average (default)
        consensus_weighted = self.algorithm.calculate_consensus(
            self.results, method="weighted_average"
        )

        # Median method
        consensus_median = self.algorithm.calculate_consensus(self.results, method="median")

        # Trimmed mean method
        consensus_trimmed = self.algorithm.calculate_consensus(self.results, method="trimmed_mean")

        assert consensus_weighted.consensus_method == "weighted_average"
        assert consensus_median.consensus_method == "median"
        assert consensus_trimmed.consensus_method == "trimmed_mean"

        # Median should be 0.8 (middle of [0.7, 0.8, 0.9])
        assert consensus_median.consensus_score == Decimal("0.8")

    def test_insufficient_data_error(self):
        """Test error when insufficient evaluation results provided."""
        with pytest.raises(InsufficientDataError):
            self.algorithm.calculate_consensus([self.results[0]])  # Only one result

    def test_detect_outliers(self):
        """Test outlier detection."""
        # Add an outlier result
        outlier_result = EvaluationResult.create_pending(
            judge_id="outlier_judge",
            template_id=uuid4(),
            prompt="Test prompt",
            response="Test response",
        )
        outlier_result.overall_score = Decimal("0.1")  # Far from others
        outlier_result.confidence_score = Decimal("0.9")
        outlier_result.completed_at = datetime.utcnow()
        outlier_result.dimension_scores = {"test": 1}
        outlier_result.reasoning = "Outlier reasoning"
        outlier_result.evaluation_time_ms = 1000

        results_with_outlier = self.results + [outlier_result]

        outliers = self.algorithm.detect_outliers(results_with_outlier)

        # Should detect the outlier
        assert "outlier_judge" in outliers

    def test_calculate_agreement(self):
        """Test agreement calculation."""
        agreement = self.algorithm.calculate_agreement(self.results)

        assert 0 <= agreement <= 1
        # Should have decent agreement since scores are close (0.7, 0.8, 0.9)
        assert agreement > Decimal("0.5")

    def test_consensus_with_outliers_excluded(self):
        """Test consensus calculation excluding outliers."""
        # Add outlier
        outlier_result = EvaluationResult.create_pending(
            judge_id="outlier_judge",
            template_id=uuid4(),
            prompt="Test prompt",
            response="Test response",
        )
        outlier_result.overall_score = Decimal("0.1")
        outlier_result.confidence_score = Decimal("0.5")
        outlier_result.completed_at = datetime.utcnow()
        outlier_result.dimension_scores = {"test": 1}
        outlier_result.reasoning = "Outlier reasoning"
        outlier_result.evaluation_time_ms = 1000

        results_with_outlier = self.results + [outlier_result]

        # Calculate consensus excluding outliers
        consensus = self.algorithm.calculate_consensus(results_with_outlier, exclude_outliers=True)

        assert "outlier_judge" in consensus.outlier_judges
        assert consensus.get_effective_judges_count() == 3  # Original judges

    def test_consensus_confidence_weighting(self):
        """Test consensus with confidence weighting."""
        # Test with confidence weighting
        consensus_weighted = self.algorithm.calculate_consensus(
            self.results, confidence_weighting=True
        )

        # Test without confidence weighting
        consensus_unweighted = self.algorithm.calculate_consensus(
            self.results, confidence_weighting=False
        )

        # Both should be valid but potentially different
        assert isinstance(consensus_weighted, ConsensusResult)
        assert isinstance(consensus_unweighted, ConsensusResult)

        # Judge weights should be different
        assert consensus_weighted.judge_weights != consensus_unweighted.judge_weights


class TestQualityController:
    """Test cases for QualityController service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.controller = QualityController()

        # Create template for testing
        dimensions = [
            Dimension.create("accuracy", "Test accuracy", Decimal("0.6"), {1: "Poor", 5: "Great"}),
            Dimension.create("clarity", "Test clarity", Decimal("0.4"), {1: "Poor", 5: "Great"}),
        ]

        self.template = EvaluationTemplate.create(
            name="Test Template",
            description="Test template",
            dimensions=dimensions,
            prompt_template="Test {prompt} {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

    def test_validate_good_evaluation_quality(self):
        """Test validation of high-quality evaluation."""
        result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=self.template.template_id,
            prompt="What is 2+2?",
            response="2+2 equals 4, which is a basic arithmetic calculation.",
        )

        # Complete with good evaluation
        result.complete_evaluation(
            template=self.template,
            dimension_scores={"accuracy": 5, "clarity": 4},
            confidence_score=Decimal("0.85"),
            reasoning="The response is mathematically accurate and clearly explains the calculation process. The answer is correct and the explanation shows understanding of basic arithmetic.",
            evaluation_time_ms=1500,
        )

        quality_report = self.controller.validate_evaluation_quality(
            result, self.template, "test_evaluator"
        )

        assert quality_report.is_passing()
        assert quality_report.overall_quality in [QualityLevel.GOOD, QualityLevel.EXCELLENT]
        assert not quality_report.has_critical_issues()

    def test_validate_poor_evaluation_quality(self):
        """Test validation of poor-quality evaluation."""
        result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=self.template.template_id,
            prompt="What is 2+2?",
            response="Four.",
        )

        # Complete with poor evaluation
        result.complete_evaluation(
            template=self.template,
            dimension_scores={"accuracy": 3, "clarity": 2},
            confidence_score=Decimal("0.2"),  # Very low confidence
            reasoning="ok",  # Very brief reasoning
            evaluation_time_ms=100,
        )

        quality_report = self.controller.validate_evaluation_quality(
            result, self.template, "test_evaluator"
        )

        assert not quality_report.is_passing()
        assert len(quality_report.issues) > 0

        # Should detect brief reasoning
        reasoning_issues = quality_report.get_issues_by_type(QualityIssueType.REASONING_INCOMPLETE)
        assert len(reasoning_issues) > 0

        # Should detect low confidence
        confidence_issues = quality_report.get_issues_by_type(QualityIssueType.LOW_CONFIDENCE)
        assert len(confidence_issues) > 0

    def test_validate_failed_evaluation(self):
        """Test validation of failed evaluation."""
        result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=self.template.template_id,
            prompt="Test prompt",
            response="Test response",
        )

        result.fail_evaluation("Evaluation failed for testing")

        quality_report = self.controller.validate_evaluation_quality(
            result, self.template, "test_evaluator"
        )

        assert not quality_report.is_passing()
        assert quality_report.has_critical_issues()

    def test_check_consensus_quality(self):
        """Test consensus quality validation."""
        # High quality consensus
        high_quality_consensus = ConsensusResult.create_simple_consensus(
            {"judge_1": Decimal("0.8"), "judge_2": Decimal("0.82"), "judge_3": Decimal("0.78")}
        )

        quality_report = self.controller.check_consensus_quality(
            high_quality_consensus, "test_evaluator"
        )

        assert quality_report.is_passing()

        # Low quality consensus
        low_quality_consensus = ConsensusResult.create_simple_consensus(
            {"judge_1": Decimal("0.2"), "judge_2": Decimal("0.8"), "judge_3": Decimal("0.5")}
        )

        low_quality_report = self.controller.check_consensus_quality(
            low_quality_consensus, "test_evaluator"
        )

        # Should have lower quality due to poor agreement
        assert low_quality_report.quality_score < quality_report.quality_score

    def test_suspicious_pattern_detection(self):
        """Test detection of suspicious patterns in reasoning."""
        result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=self.template.template_id,
            prompt="Explain quantum physics",
            response="Quantum physics is complex.",
        )

        # Complete with suspicious reasoning
        result.complete_evaluation(
            template=self.template,
            dimension_scores={"accuracy": 3, "clarity": 3},
            confidence_score=Decimal("0.5"),
            reasoning="I don't know enough about this topic to evaluate properly. Cannot assess the accuracy.",
            evaluation_time_ms=500,
        )

        quality_report = self.controller.validate_evaluation_quality(
            result, self.template, "test_evaluator"
        )

        # Should detect suspicious patterns
        suspicious_issues = quality_report.get_issues_by_type(QualityIssueType.SUSPICIOUS_PATTERN)
        assert len(suspicious_issues) > 0

    def test_score_consistency_check(self):
        """Test score-reasoning consistency validation."""
        result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=self.template.template_id,
            prompt="Test prompt",
            response="Test response",
        )

        # Inconsistent: high scores but negative reasoning
        result.complete_evaluation(
            template=self.template,
            dimension_scores={"accuracy": 5, "clarity": 5},  # High scores
            confidence_score=Decimal("0.9"),
            reasoning="This response is terrible, poorly written, inaccurate, and completely unhelpful. Very bad quality.",  # Negative reasoning
            evaluation_time_ms=1000,
        )

        quality_report = self.controller.validate_evaluation_quality(
            result, self.template, "test_evaluator"
        )

        # Should detect score-reasoning inconsistency
        consistency_issues = quality_report.get_issues_by_type(QualityIssueType.SCORE_INCONSISTENT)
        assert len(consistency_issues) > 0


class TestJudgeCalibrator:
    """Test cases for JudgeCalibrator service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calibrator = JudgeCalibrator()

        # Create judge for testing
        self.judge = Judge.create(
            judge_id="test_judge",
            name="Test Judge",
            description="Test judge for calibration",
            model_provider_id="test_provider",
        )

        # Create template
        dimensions = [
            Dimension.create("accuracy", "Test accuracy", Decimal("1.0"), {1: "Poor", 5: "Great"})
        ]

        self.template = EvaluationTemplate.create(
            name="Test Template",
            description="Test template",
            dimensions=dimensions,
            prompt_template="Test {prompt} {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

    @pytest.mark.asyncio
    async def test_calibrate_judge_success(self):
        """Test successful judge calibration."""
        # Create golden standard results
        golden_standard_results = []

        for i, (judge_score, golden_score) in enumerate(
            [
                (Decimal("0.8"), Decimal("0.75")),  # Close to golden standard
                (Decimal("0.6"), Decimal("0.65")),  # Close to golden standard
                (Decimal("0.9"), Decimal("0.85")),  # Close to golden standard
                (Decimal("0.7"), Decimal("0.7")),  # Exact match
                (Decimal("0.5"), Decimal("0.55")),  # Close to golden standard
                (Decimal("0.4"), Decimal("0.4")),  # Exact match
                (Decimal("0.8"), Decimal("0.8")),  # Exact match
                (Decimal("0.3"), Decimal("0.35")),  # Close to golden standard
                (Decimal("0.9"), Decimal("0.9")),  # Exact match
                (Decimal("0.6"), Decimal("0.6")),  # Exact match
            ]
        ):
            # Create evaluation result
            result = EvaluationResult.create_pending(
                judge_id=self.judge.judge_id,
                template_id=self.template.template_id,
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
            )

            result.complete_evaluation(
                template=self.template,
                dimension_scores={"accuracy": int(judge_score * 5)},  # Convert to 1-5 scale
                confidence_score=Decimal("0.8"),
                reasoning=f"Test reasoning {i}",
                evaluation_time_ms=1000,
            )

            golden_standard_results.append((result, golden_score))

        # Calibrate judge
        calibration_data = await self.calibrator.calibrate_judge(
            self.judge, golden_standard_results
        )

        assert isinstance(calibration_data, CalibrationData)
        assert calibration_data.sample_size == 10
        assert calibration_data.accuracy > Decimal("0.8")  # Good accuracy since scores are close
        assert self.judge.is_calibrated()

    @pytest.mark.asyncio
    async def test_calibrate_judge_insufficient_data(self):
        """Test calibration failure with insufficient data."""
        # Only provide a few samples (less than minimum)
        golden_standard_results = [
            (
                EvaluationResult.create_pending(
                    judge_id=self.judge.judge_id,
                    template_id=self.template.template_id,
                    prompt="Test",
                    response="Test",
                ),
                Decimal("0.8"),
            )
        ]

        with pytest.raises(InsufficientDataError):
            await self.calibrator.calibrate_judge(self.judge, golden_standard_results)

    def test_check_calibration_drift(self):
        """Test calibration drift detection."""
        # Add initial calibration
        calibration_data = CalibrationData(
            accuracy=Decimal("0.9"),
            consistency=Decimal("0.85"),
            bias_score=Decimal("0.05"),
            confidence_calibration=Decimal("0.8"),
            sample_size=100,
            calibrated_at=datetime.utcnow() - timedelta(days=45),  # Old calibration
        )
        self.judge.calibrate(calibration_data)

        # Create recent results showing drift
        recent_results = []
        for i in range(15):
            result = EvaluationResult.create_pending(
                judge_id=self.judge.judge_id,
                template_id=self.template.template_id,
                prompt=f"Recent test {i}",
                response=f"Recent response {i}",
            )

            # Results showing inconsistency (drift in consistency)
            score = Decimal("0.8") if i % 2 == 0 else Decimal("0.3")  # High variance
            result.complete_evaluation(
                template=self.template,
                dimension_scores={"accuracy": int(score * 5)},
                confidence_score=Decimal("0.7"),
                reasoning=f"Recent reasoning {i}",
                evaluation_time_ms=1000,
            )
            recent_results.append(result)

        drift_check = self.calibrator.check_calibration_drift(self.judge, recent_results)

        assert drift_check["has_drift"] is True
        assert "stale_calibration" in [
            indicator["type"] for indicator in drift_check["drift_indicators"]
        ]

    def test_recommend_recalibration(self):
        """Test recalibration recommendations."""
        # Uncalibrated judge should need calibration
        recommendation = self.calibrator.recommend_recalibration(self.judge)

        assert recommendation["needs_recalibration"] is True
        assert recommendation["priority"] == "high"
        assert "never been calibrated" in recommendation["reasons"][0]

        # Add poor calibration
        poor_calibration = CalibrationData(
            accuracy=Decimal("0.6"),  # Below production threshold
            consistency=Decimal("0.5"),  # Below production threshold
            bias_score=Decimal("0.4"),  # High bias
            confidence_calibration=Decimal("0.4"),  # Poor confidence calibration
            sample_size=20,  # Small sample
            calibrated_at=datetime.utcnow() - timedelta(days=60),  # Stale
        )
        self.judge.calibrate(poor_calibration)

        recommendation = self.calibrator.recommend_recalibration(self.judge)

        assert recommendation["needs_recalibration"] is True
        assert recommendation["priority"] == "high"
        assert any("production standards" in reason for reason in recommendation["reasons"])

    def test_get_calibration_summary(self):
        """Test calibration summary generation."""
        # Uncalibrated judge
        summary = self.calibrator.get_calibration_summary(self.judge)

        assert summary["is_calibrated"] is False
        assert summary["status"] == "uncalibrated"

        # Add good calibration
        good_calibration = CalibrationData(
            accuracy=Decimal("0.9"),
            consistency=Decimal("0.87"),
            bias_score=Decimal("0.08"),
            confidence_calibration=Decimal("0.82"),
            sample_size=150,
            calibrated_at=datetime.utcnow(),
        )
        self.judge.calibrate(good_calibration)

        summary = self.calibrator.get_calibration_summary(self.judge)

        assert summary["is_calibrated"] is True
        assert summary["is_production_ready"] is True
        assert summary["quality_grade"] in ["GOOD", "EXCELLENT"]
        assert "metrics" in summary
        assert "recalibration_recommendation" in summary
