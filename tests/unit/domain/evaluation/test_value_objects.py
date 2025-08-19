"""Tests for evaluation domain value objects."""

from datetime import datetime
from decimal import Decimal
from typing import Dict

import pytest

from src.domain.evaluation.exceptions import InsufficientDataError, ValidationError
from src.domain.evaluation.value_objects.calibration_data import CalibrationData
from src.domain.evaluation.value_objects.consensus_result import ConsensusResult
from src.domain.evaluation.value_objects.quality_report import (
    QualityIssue,
    QualityIssueType,
    QualityLevel,
    QualityReport,
)
from src.domain.evaluation.value_objects.scoring_scale import ScoringScale


class TestScoringScale:
    """Test cases for ScoringScale value object."""

    def test_create_five_point_likert(self):
        """Test factory method for 5-point Likert scale."""
        scale = ScoringScale.create_five_point_likert()

        assert scale.min_score == Decimal("1")
        assert scale.max_score == Decimal("5")
        assert scale.scale_type == "likert"
        assert scale.scale_steps == 5

    def test_create_continuous_zero_to_one(self):
        """Test factory method for continuous 0-1 scale."""
        scale = ScoringScale.create_continuous_zero_to_one()

        assert scale.min_score == Decimal("0.0")
        assert scale.max_score == Decimal("1.0")
        assert scale.scale_type == "continuous"

    def test_is_valid_score_continuous(self):
        """Test score validation for continuous scale."""
        scale = ScoringScale.create_continuous_zero_to_one()

        assert scale.is_valid_score(0.5)
        assert scale.is_valid_score(0.0)
        assert scale.is_valid_score(1.0)
        assert not scale.is_valid_score(-0.1)
        assert not scale.is_valid_score(1.1)

    def test_is_valid_score_likert(self):
        """Test score validation for Likert scale."""
        scale = ScoringScale.create_five_point_likert()

        assert scale.is_valid_score(1)
        assert scale.is_valid_score(3)
        assert scale.is_valid_score(5)
        assert not scale.is_valid_score(0)
        assert not scale.is_valid_score(6)
        assert not scale.is_valid_score(2.5)  # Not discrete

    def test_normalize_score(self):
        """Test score normalization."""
        scale = ScoringScale.create_five_point_likert()

        assert scale.normalize_score(1) == Decimal("0")
        assert scale.normalize_score(3) == Decimal("0.5")
        assert scale.normalize_score(5) == Decimal("1")

    def test_denormalize_score(self):
        """Test score denormalization."""
        scale = ScoringScale.create_five_point_likert()

        assert scale.denormalize_score(Decimal("0")) == Decimal("1")
        assert scale.denormalize_score(Decimal("0.5")) == Decimal("3")
        assert scale.denormalize_score(Decimal("1")) == Decimal("5")

    def test_get_discrete_values(self):
        """Test getting discrete values for Likert scale."""
        scale = ScoringScale.create_five_point_likert()
        values = scale.get_discrete_values()

        assert len(values) == 5
        assert values == [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]

    def test_invalid_scale_parameters(self):
        """Test validation of scale parameters."""
        with pytest.raises(ValidationError):
            ScoringScale(
                min_score=Decimal("5"),
                max_score=Decimal("1"),  # Invalid: min > max
                scale_type="continuous",
                scale_steps=0,
                description="Invalid scale",
            )

        with pytest.raises(ValidationError):
            ScoringScale(
                min_score=Decimal("1"),
                max_score=Decimal("5"),
                scale_type="invalid_type",  # Invalid scale type
                scale_steps=5,
                description="Invalid scale",
            )


class TestCalibrationData:
    """Test cases for CalibrationData value object."""

    def test_create_uncalibrated(self):
        """Test factory method for uncalibrated data."""
        data = CalibrationData.create_uncalibrated()

        assert data.accuracy == Decimal("0.0")
        assert data.consistency == Decimal("0.0")
        assert data.sample_size == 10  # Minimum required
        assert not data.is_production_ready()  # Still not production ready due to low metrics

    def test_is_production_ready(self):
        """Test production readiness check."""
        # Production ready calibration
        data = CalibrationData(
            accuracy=Decimal("0.85"),
            consistency=Decimal("0.8"),
            bias_score=Decimal("0.1"),
            confidence_calibration=Decimal("0.75"),
            sample_size=100,
            calibrated_at=datetime.utcnow(),
        )

        assert data.is_production_ready()

        # Not production ready - low accuracy
        data_low_accuracy = CalibrationData(
            accuracy=Decimal("0.7"),  # Below threshold
            consistency=Decimal("0.8"),
            bias_score=Decimal("0.1"),
            confidence_calibration=Decimal("0.75"),
            sample_size=100,
            calibrated_at=datetime.utcnow(),
        )

        assert not data_low_accuracy.is_production_ready()

    def test_has_significant_bias(self):
        """Test bias detection."""
        # No significant bias
        data_no_bias = CalibrationData(
            accuracy=Decimal("0.85"),
            consistency=Decimal("0.8"),
            bias_score=Decimal("0.1"),
            confidence_calibration=Decimal("0.75"),
            sample_size=100,
            calibrated_at=datetime.utcnow(),
        )

        assert not data_no_bias.has_significant_bias()

        # Significant positive bias
        data_bias = CalibrationData(
            accuracy=Decimal("0.85"),
            consistency=Decimal("0.8"),
            bias_score=Decimal("0.4"),  # Above threshold
            confidence_calibration=Decimal("0.75"),
            sample_size=100,
            calibrated_at=datetime.utcnow(),
        )

        assert data_bias.has_significant_bias()

    def test_get_quality_grade(self):
        """Test quality grade calculation."""
        # Excellent grade
        excellent_data = CalibrationData(
            accuracy=Decimal("0.96"),
            consistency=Decimal("0.92"),
            bias_score=Decimal("0.05"),
            confidence_calibration=Decimal("0.8"),
            sample_size=100,
            calibrated_at=datetime.utcnow(),
        )

        assert excellent_data.get_quality_grade() == "EXCELLENT"

        # Good grade
        good_data = CalibrationData(
            accuracy=Decimal("0.92"),
            consistency=Decimal("0.87"),
            bias_score=Decimal("0.12"),
            confidence_calibration=Decimal("0.75"),
            sample_size=100,
            calibrated_at=datetime.utcnow(),
        )

        assert good_data.get_quality_grade() == "GOOD"

        # Needs calibration
        poor_data = CalibrationData(
            accuracy=Decimal("0.7"),
            consistency=Decimal("0.6"),
            bias_score=Decimal("0.3"),
            confidence_calibration=Decimal("0.5"),
            sample_size=20,
            calibrated_at=datetime.utcnow(),
        )

        assert poor_data.get_quality_grade() == "NEEDS_CALIBRATION"

    def test_get_reliability_score(self):
        """Test composite reliability score calculation."""
        data = CalibrationData(
            accuracy=Decimal("0.9"),
            consistency=Decimal("0.85"),
            bias_score=Decimal("0.1"),
            confidence_calibration=Decimal("0.8"),
            sample_size=100,
            calibrated_at=datetime.utcnow(),
        )

        reliability = data.get_reliability_score()

        # Should be high reliability score
        assert reliability > Decimal("0.8")
        assert reliability <= Decimal("1.0")

    def test_invalid_calibration_parameters(self):
        """Test validation of calibration parameters."""
        with pytest.raises(ValidationError):
            CalibrationData(
                accuracy=Decimal("1.5"),  # Invalid: > 1
                consistency=Decimal("0.8"),
                bias_score=Decimal("0.1"),
                confidence_calibration=Decimal("0.75"),
                sample_size=100,
                calibrated_at=datetime.utcnow(),
            )

        with pytest.raises(ValidationError):
            CalibrationData(
                accuracy=Decimal("0.8"),
                consistency=Decimal("0.8"),
                bias_score=Decimal("2.0"),  # Invalid: > 1
                confidence_calibration=Decimal("0.75"),
                sample_size=100,
                calibrated_at=datetime.utcnow(),
            )

        with pytest.raises(ValidationError):
            CalibrationData(
                accuracy=Decimal("0.8"),
                consistency=Decimal("0.8"),
                bias_score=Decimal("0.1"),
                confidence_calibration=Decimal("0.75"),
                sample_size=5,  # Invalid: < 10
                calibrated_at=datetime.utcnow(),
            )


class TestConsensusResult:
    """Test cases for ConsensusResult value object."""

    def test_create_simple_consensus(self):
        """Test simple consensus creation with equal weights."""
        judge_scores = {
            "judge_1": Decimal("0.8"),
            "judge_2": Decimal("0.7"),
            "judge_3": Decimal("0.9"),
        }

        consensus = ConsensusResult.create_simple_consensus(judge_scores)

        assert len(consensus.judge_scores) == 3
        assert consensus.consensus_method == "weighted_average"
        assert Decimal("0.7") <= consensus.consensus_score <= Decimal("0.9")

    def test_create_simple_consensus_median(self):
        """Test simple consensus creation using median method."""
        judge_scores = {
            "judge_1": Decimal("0.8"),
            "judge_2": Decimal("0.7"),
            "judge_3": Decimal("0.9"),
        }

        consensus = ConsensusResult.create_simple_consensus(judge_scores, method="median")

        assert consensus.consensus_score == Decimal("0.8")  # Median of [0.7, 0.8, 0.9]
        assert consensus.consensus_method == "median"

    def test_insufficient_judges_error(self):
        """Test error when insufficient judges provided."""
        with pytest.raises(InsufficientDataError):
            ConsensusResult.create_simple_consensus({"judge_1": Decimal("0.8")})

    def test_is_high_agreement(self):
        """Test high agreement detection."""
        # High agreement consensus
        high_agreement_scores = {
            "judge_1": Decimal("0.8"),
            "judge_2": Decimal("0.82"),
            "judge_3": Decimal("0.78"),
        }

        consensus = ConsensusResult.create_simple_consensus(high_agreement_scores)
        assert consensus.is_high_agreement()

        # Low agreement consensus
        low_agreement_scores = {
            "judge_1": Decimal("0.2"),
            "judge_2": Decimal("0.8"),
            "judge_3": Decimal("0.5"),
        }

        consensus_low = ConsensusResult.create_simple_consensus(low_agreement_scores)
        assert not consensus_low.is_high_agreement()

    def test_get_consensus_strength(self):
        """Test consensus strength assessment."""
        # Very strong consensus
        strong_scores = {
            "judge_1": Decimal("0.8"),
            "judge_2": Decimal("0.8"),
            "judge_3": Decimal("0.8"),
        }

        consensus = ConsensusResult.create_simple_consensus(strong_scores)
        strength = consensus.get_consensus_strength()
        assert strength in ["VERY_STRONG", "STRONG"]

    def test_get_judge_deviations(self):
        """Test judge deviation calculation."""
        judge_scores = {
            "judge_1": Decimal("0.8"),
            "judge_2": Decimal("0.6"),
            "judge_3": Decimal("0.9"),
        }

        consensus = ConsensusResult.create_simple_consensus(judge_scores)
        deviations = consensus.get_judge_deviations()

        assert len(deviations) == 3
        assert all(dev >= 0 for dev in deviations.values())

    def test_invalid_consensus_parameters(self):
        """Test validation of consensus parameters."""
        with pytest.raises(ValidationError):
            ConsensusResult(
                consensus_score=Decimal("1.5"),  # Invalid: > 1
                confidence_interval=(Decimal("0.4"), Decimal("0.6")),
                agreement_level=Decimal("0.8"),
                judge_scores={"judge_1": Decimal("0.8")},
                judge_weights={"judge_1": Decimal("1.0")},
                outlier_judges=[],
                statistical_significance=Decimal("0.05"),
                consensus_method="weighted_average",
            )


class TestQualityReport:
    """Test cases for QualityReport value object."""

    def test_create_passed_report(self):
        """Test creation of passing quality report."""
        report = QualityReport.create_passed(
            quality_score=Decimal("0.85"), evaluator_id="test_evaluator"
        )

        assert report.is_passing()
        assert report.overall_quality == QualityLevel.GOOD
        assert len(report.issues) == 0
        assert not report.has_critical_issues()

    def test_create_failed_report(self):
        """Test creation of failing quality report."""
        issues = [
            QualityIssue(
                issue_type=QualityIssueType.REASONING_INCOMPLETE,
                severity="high",
                description="Reasoning is too brief",
            )
        ]

        report = QualityReport.create_failed(
            quality_score=Decimal("0.4"), issues=issues, evaluator_id="test_evaluator"
        )

        assert not report.is_passing()
        assert report.overall_quality in [QualityLevel.POOR, QualityLevel.UNACCEPTABLE]
        assert len(report.issues) == 1
        assert report.has_critical_issues()

    def test_get_issues_by_severity(self):
        """Test filtering issues by severity."""
        issues = [
            QualityIssue(
                issue_type=QualityIssueType.REASONING_INCOMPLETE,
                severity="high",
                description="High severity issue",
            ),
            QualityIssue(
                issue_type=QualityIssueType.LOW_CONFIDENCE,
                severity="medium",
                description="Medium severity issue",
            ),
            QualityIssue(
                issue_type=QualityIssueType.BIAS_DETECTED,
                severity="low",
                description="Low severity issue",
            ),
        ]

        report = QualityReport.create_failed(
            quality_score=Decimal("0.3"), issues=issues, evaluator_id="test_evaluator"
        )

        high_issues = report.get_issues_by_severity("high")
        medium_issues = report.get_issues_by_severity("medium")
        low_issues = report.get_issues_by_severity("low")

        assert len(high_issues) == 1
        assert len(medium_issues) == 1
        assert len(low_issues) == 1

    def test_get_issues_by_type(self):
        """Test filtering issues by type."""
        issues = [
            QualityIssue(
                issue_type=QualityIssueType.REASONING_INCOMPLETE,
                severity="high",
                description="Reasoning issue 1",
            ),
            QualityIssue(
                issue_type=QualityIssueType.REASONING_INCOMPLETE,
                severity="medium",
                description="Reasoning issue 2",
            ),
            QualityIssue(
                issue_type=QualityIssueType.LOW_CONFIDENCE,
                severity="low",
                description="Confidence issue",
            ),
        ]

        report = QualityReport.create_failed(
            quality_score=Decimal("0.3"), issues=issues, evaluator_id="test_evaluator"
        )

        reasoning_issues = report.get_issues_by_type(QualityIssueType.REASONING_INCOMPLETE)
        confidence_issues = report.get_issues_by_type(QualityIssueType.LOW_CONFIDENCE)

        assert len(reasoning_issues) == 2
        assert len(confidence_issues) == 1

    def test_add_metric(self):
        """Test adding metrics to quality report."""
        report = QualityReport.create_passed(
            quality_score=Decimal("0.85"), evaluator_id="test_evaluator"
        )

        updated_report = report.add_metric("test_metric", Decimal("0.9"))

        assert updated_report.get_metric("test_metric") == Decimal("0.9")
        assert report.get_metric("test_metric") is None  # Original unchanged

    def test_invalid_quality_parameters(self):
        """Test validation of quality report parameters."""
        with pytest.raises(ValidationError):
            QualityReport(
                overall_quality=QualityLevel.GOOD,
                quality_score=Decimal("1.5"),  # Invalid: > 1
                issues=[],
                metrics={},
                recommendations=[],
                assessed_at=datetime.utcnow(),
                evaluator_id="test",
            )

        with pytest.raises(ValidationError):
            QualityReport(
                overall_quality=QualityLevel.GOOD,
                quality_score=Decimal("0.8"),
                issues=[],
                metrics={},
                recommendations=[],
                assessed_at=datetime.utcnow(),
                evaluator_id="",  # Invalid: empty
            )
