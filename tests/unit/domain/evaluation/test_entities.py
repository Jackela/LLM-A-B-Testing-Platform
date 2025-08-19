"""Tests for evaluation domain entities."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from src.domain.evaluation.entities.dimension import STANDARD_DIMENSIONS, Dimension
from src.domain.evaluation.entities.evaluation_result import EvaluationResult
from src.domain.evaluation.entities.evaluation_template import EvaluationTemplate
from src.domain.evaluation.entities.judge import Judge
from src.domain.evaluation.exceptions import (
    BusinessRuleViolation,
    InvalidScore,
    JudgeNotCalibratedError,
    MissingDimensionScore,
    TemplateRenderError,
    ValidationError,
)
from src.domain.evaluation.value_objects.calibration_data import CalibrationData
from src.domain.evaluation.value_objects.quality_report import QualityReport
from src.domain.evaluation.value_objects.scoring_scale import ScoringScale


class TestDimension:
    """Test cases for Dimension entity."""

    def test_create_dimension(self):
        """Test dimension creation."""
        dimension = Dimension.create(
            name="accuracy",
            description="Test accuracy dimension",
            weight=Decimal("0.3"),
            scoring_criteria={
                1: "Poor accuracy",
                2: "Fair accuracy",
                3: "Good accuracy",
                4: "Very good accuracy",
                5: "Excellent accuracy",
            },
        )

        assert dimension.name == "accuracy"
        assert dimension.description == "Test accuracy dimension"
        assert dimension.weight == Decimal("0.3")
        assert len(dimension.scoring_criteria) == 5
        assert dimension.is_required is True

    def test_dimension_validation(self):
        """Test dimension validation rules."""
        # Empty name should fail
        with pytest.raises(ValidationError, match="Dimension name cannot be empty"):
            Dimension.create(
                name="",
                description="Test",
                weight=Decimal("0.3"),
                scoring_criteria={1: "Poor", 5: "Excellent"},
            )

        # Invalid weight should fail
        with pytest.raises(ValidationError, match="Dimension weight must be between 0 and 1"):
            Dimension.create(
                name="test",
                description="Test",
                weight=Decimal("1.5"),  # Invalid
                scoring_criteria={1: "Poor", 5: "Excellent"},
            )

        # Empty scoring criteria should fail
        with pytest.raises(ValidationError, match="Dimension must have scoring criteria"):
            Dimension.create(
                name="test", description="Test", weight=Decimal("0.3"), scoring_criteria={}  # Empty
            )

    def test_score_validation(self):
        """Test score validation methods."""
        dimension = Dimension.create(
            name="test",
            description="Test dimension",
            weight=Decimal("0.3"),
            scoring_criteria={1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"},
        )

        assert dimension.is_valid_score(3)
        assert not dimension.is_valid_score(0)
        assert not dimension.is_valid_score(6)

        assert dimension.get_score_range() == (1, 5)

    def test_score_normalization(self):
        """Test score normalization and denormalization."""
        dimension = Dimension.create(
            name="test",
            description="Test dimension",
            weight=Decimal("0.3"),
            scoring_criteria={1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"},
        )

        # Test normalization
        assert dimension.normalize_score(1) == Decimal("0")
        assert dimension.normalize_score(3) == Decimal("0.5")
        assert dimension.normalize_score(5) == Decimal("1")

        # Test denormalization
        assert dimension.denormalize_score(Decimal("0")) == 1
        assert dimension.denormalize_score(Decimal("0.5")) == 3
        assert dimension.denormalize_score(Decimal("1")) == 5

    def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        dimension = Dimension.create(
            name="test",
            description="Test dimension",
            weight=Decimal("0.4"),
            scoring_criteria={1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"},
        )

        # Score of 5 (normalized to 1.0) with weight 0.4 should give 0.4
        weighted_score = dimension.calculate_weighted_score(5)
        assert weighted_score == Decimal("0.4")

        # Score of 3 (normalized to 0.5) with weight 0.4 should give 0.2
        weighted_score = dimension.calculate_weighted_score(3)
        assert weighted_score == Decimal("0.2")

    def test_standard_dimensions(self):
        """Test predefined standard dimensions."""
        accuracy_dim = STANDARD_DIMENSIONS["accuracy"]

        assert accuracy_dim.name == "accuracy"
        assert accuracy_dim.weight == Decimal("0.3")
        assert accuracy_dim.is_valid_score(3)
        assert len(accuracy_dim.scoring_criteria) == 5

        # Check all standard dimensions exist
        expected_dimensions = ["accuracy", "relevance", "clarity", "usefulness"]
        for dim_name in expected_dimensions:
            assert dim_name in STANDARD_DIMENSIONS
            dim = STANDARD_DIMENSIONS[dim_name]
            assert isinstance(dim, Dimension)
            assert dim.is_valid_score(3)


class TestEvaluationTemplate:
    """Test cases for EvaluationTemplate entity."""

    def test_create_template(self):
        """Test evaluation template creation."""
        dimensions = [STANDARD_DIMENSIONS["accuracy"], STANDARD_DIMENSIONS["clarity"]]

        # Adjust weights to sum to 1.0
        dimensions[0].weight = Decimal("0.6")
        dimensions[1].weight = Decimal("0.4")

        template = EvaluationTemplate.create(
            name="Test Template",
            description="A test evaluation template",
            dimensions=dimensions,
            prompt_template="Evaluate the following:\nPrompt: {prompt}\nResponse: {response}\n\nDimensions:\n{dimensions}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
            model_parameters={"temperature": 0.1},
        )

        assert template.name == "Test Template"
        assert len(template.dimensions) == 2
        assert template.is_active is True
        assert template.version == 1

    def test_template_validation(self):
        """Test template validation rules."""
        dimensions = [STANDARD_DIMENSIONS["accuracy"]]

        # Empty name should fail
        with pytest.raises(ValidationError, match="Template name cannot be empty"):
            EvaluationTemplate.create(
                name="",
                description="Test",
                dimensions=dimensions,
                prompt_template="Test {prompt} {response}",
                scoring_scale=ScoringScale.create_five_point_likert(),
                judge_model_id="gpt-4",
            )

        # No dimensions should fail
        with pytest.raises(ValidationError, match="Template must have at least one dimension"):
            EvaluationTemplate.create(
                name="Test",
                description="Test",
                dimensions=[],  # Empty
                prompt_template="Test {prompt} {response}",
                scoring_scale=ScoringScale.create_five_point_likert(),
                judge_model_id="gpt-4",
            )

    def test_dimension_weight_validation(self):
        """Test dimension weight sum validation."""
        # Weights that don't sum to 1.0 should fail
        dimensions = [
            Dimension.create(
                "dim1",
                "Test 1",
                Decimal("0.3"),
                {1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Great"},
            ),
            Dimension.create(
                "dim2",
                "Test 2",
                Decimal("0.4"),
                {1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Great"},
            ),  # Sum = 0.7
        ]

        with pytest.raises(ValidationError, match="Dimension weights must sum to 1.0"):
            EvaluationTemplate.create(
                name="Test",
                description="Test",
                dimensions=dimensions,
                prompt_template="Test {prompt} {response}",
                scoring_scale=ScoringScale.create_five_point_likert(),
                judge_model_id="gpt-4",
            )

    def test_prompt_template_rendering(self):
        """Test prompt template rendering."""
        dimensions = [STANDARD_DIMENSIONS["accuracy"]]

        template = EvaluationTemplate.create(
            name="Test Template",
            description="Test",
            dimensions=dimensions,
            prompt_template="Evaluate: {prompt} -> {response}\nExtra: {extra_context}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        rendered = template.render(
            prompt="What is 2+2?", response="4", extra_context="Math problem"
        )

        assert "What is 2+2?" in rendered
        assert "4" in rendered
        assert "Math problem" in rendered

    def test_missing_template_placeholders(self):
        """Test error when required placeholders are missing."""
        dimensions = [STANDARD_DIMENSIONS["accuracy"]]

        with pytest.raises(ValidationError, match="Template missing required placeholder"):
            EvaluationTemplate.create(
                name="Test",
                description="Test",
                dimensions=dimensions,
                prompt_template="No placeholders here",  # Missing {prompt} and {response}
                scoring_scale=ScoringScale.create_five_point_likert(),
                judge_model_id="gpt-4",
            )

    def test_calculate_weighted_score(self):
        """Test weighted score calculation across dimensions."""
        dimensions = [
            Dimension.create(
                "dim1",
                "Test 1",
                Decimal("0.6"),
                {1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"},
            ),
            Dimension.create(
                "dim2",
                "Test 2",
                Decimal("0.4"),
                {1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"},
            ),
        ]

        template = EvaluationTemplate.create(
            name="Test",
            description="Test",
            dimensions=dimensions,
            prompt_template="Test {prompt} {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        # dim1: score 5 (normalized 1.0) * weight 0.6 = 0.6
        # dim2: score 3 (normalized 0.5) * weight 0.4 = 0.2
        # Total: 0.8
        total_score = template.calculate_weighted_score({"dim1": 5, "dim2": 3})
        assert total_score == Decimal("0.8")

    def test_add_remove_dimensions(self):
        """Test adding and removing dimensions."""
        dimensions = [STANDARD_DIMENSIONS["accuracy"]]

        template = EvaluationTemplate.create(
            name="Test",
            description="Test",
            dimensions=dimensions,
            prompt_template="Test {prompt} {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        # Add dimension
        new_dimension = Dimension.create(
            "clarity", "Test clarity", Decimal("0.5"), {1: "Poor", 5: "Great"}
        )
        template.add_dimension(new_dimension)

        assert len(template.dimensions) == 2
        assert template.has_dimension("clarity")

        # Remove dimension
        template.remove_dimension("clarity")
        assert len(template.dimensions) == 1
        assert not template.has_dimension("clarity")


class TestEvaluationResult:
    """Test cases for EvaluationResult entity."""

    def test_create_pending_result(self):
        """Test creating pending evaluation result."""
        result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=uuid4(),
            prompt="Test prompt",
            response="Test response",
        )

        assert result.judge_id == "test_judge"
        assert result.prompt == "Test prompt"
        assert result.response == "Test response"
        assert not result.is_completed()
        assert not result.is_successful()

    def test_complete_evaluation(self):
        """Test completing an evaluation."""
        template_id = uuid4()
        result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=template_id,
            prompt="Test prompt",
            response="Test response",
        )

        # Create mock template
        dimensions = [
            Dimension.create("dim1", "Test 1", Decimal("0.6"), {1: "Poor", 5: "Great"}),
            Dimension.create("dim2", "Test 2", Decimal("0.4"), {1: "Poor", 5: "Great"}),
        ]

        template = EvaluationTemplate.create(
            name="Test",
            description="Test",
            dimensions=dimensions,
            prompt_template="Test {prompt} {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        # Complete evaluation
        result.complete_evaluation(
            template=template,
            dimension_scores={"dim1": 4, "dim2": 3},
            confidence_score=Decimal("0.85"),
            reasoning="Good evaluation with clear reasoning",
            evaluation_time_ms=1500,
        )

        assert result.is_completed()
        assert result.is_successful()
        assert result.overall_score == Decimal("0.7")  # (4*0.6 + 3*0.4) normalized
        assert result.confidence_score == Decimal("0.85")
        assert result.evaluation_time_ms == 1500

    def test_fail_evaluation(self):
        """Test failing an evaluation."""
        result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=uuid4(),
            prompt="Test prompt",
            response="Test response",
        )

        result.fail_evaluation("Test error message")

        assert result.is_completed()
        assert not result.is_successful()
        assert result.has_error()
        assert result.metadata["error_message"] == "Test error message"

    def test_confidence_level_checks(self):
        """Test confidence level utility methods."""
        template_id = uuid4()
        result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=template_id,
            prompt="Test prompt",
            response="Test response",
        )

        # Mock template with single dimension for simplicity
        dimensions = [Dimension.create("test", "Test", Decimal("1.0"), {1: "Poor", 5: "Great"})]
        template = EvaluationTemplate.create(
            name="Test",
            description="Test",
            dimensions=dimensions,
            prompt_template="Test {prompt} {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        # High confidence
        result.complete_evaluation(
            template=template,
            dimension_scores={"test": 4},
            confidence_score=Decimal("0.9"),
            reasoning="High confidence evaluation",
            evaluation_time_ms=1000,
        )

        assert result.is_high_confidence()
        assert not result.is_low_confidence()

    def test_quality_report_integration(self):
        """Test integration with quality reports."""
        result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=uuid4(),
            prompt="Test prompt",
            response="Test response",
        )

        # Add quality report
        quality_report = QualityReport.create_passed(
            quality_score=Decimal("0.85"), evaluator_id="quality_controller"
        )

        result.add_quality_report(quality_report)

        assert result.quality_report == quality_report
        assert not result.has_quality_issues()
        assert result.get_quality_score() == Decimal("0.85")


class TestJudge:
    """Test cases for Judge aggregate root."""

    def test_create_judge(self):
        """Test judge creation."""
        judge = Judge.create(
            judge_id="test_judge_1",
            name="Test Judge",
            description="A test judge for evaluation",
            model_provider_id="openai_provider",
        )

        assert judge.judge_id == "test_judge_1"
        assert judge.name == "Test Judge"
        assert judge.is_active is True
        assert len(judge.templates) == 0
        assert not judge.is_calibrated()
        assert not judge.is_production_ready()

    def test_judge_validation(self):
        """Test judge validation rules."""
        # Empty judge_id should fail
        with pytest.raises(ValidationError, match="Judge ID cannot be empty"):
            Judge.create(
                judge_id="", name="Test Judge", description="Test", model_provider_id="provider"
            )

        # Empty name should fail
        with pytest.raises(ValidationError, match="Judge name cannot be empty"):
            Judge.create(judge_id="test", name="", description="Test", model_provider_id="provider")

    def test_judge_calibration(self):
        """Test judge calibration."""
        judge = Judge.create(
            judge_id="test_judge",
            name="Test Judge",
            description="Test",
            model_provider_id="provider",
        )

        # Initially not calibrated
        assert not judge.is_calibrated()

        # Add calibration data
        calibration_data = CalibrationData(
            accuracy=Decimal("0.85"),
            consistency=Decimal("0.8"),
            bias_score=Decimal("0.1"),
            confidence_calibration=Decimal("0.75"),
            sample_size=100,
            calibrated_at=datetime.utcnow(),
        )

        judge.calibrate(calibration_data)

        assert judge.is_calibrated()
        assert judge.is_production_ready()  # Has calibration + templates will be added

    @pytest.mark.asyncio
    async def test_evaluate_uncalibrated_judge(self):
        """Test that uncalibrated judge throws error on evaluation."""
        judge = Judge.create(
            judge_id="test_judge",
            name="Test Judge",
            description="Test",
            model_provider_id="provider",
        )

        template = EvaluationTemplate.create(
            name="Test Template",
            description="Test",
            dimensions=[STANDARD_DIMENSIONS["accuracy"]],
            prompt_template="Test {prompt} {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        with pytest.raises(JudgeNotCalibratedError):
            await judge.evaluate("Test prompt", "Test response", template)

    def test_template_management(self):
        """Test adding and removing templates."""
        judge = Judge.create(
            judge_id="test_judge",
            name="Test Judge",
            description="Test",
            model_provider_id="provider",
        )

        template = EvaluationTemplate.create(
            name="Test Template",
            description="Test",
            dimensions=[STANDARD_DIMENSIONS["accuracy"]],
            prompt_template="Test {prompt} {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        # Add template
        judge.add_template(template)
        assert len(judge.templates) == 1
        assert judge.has_template(template.template_id)

        # Remove template
        judge.remove_template(template.template_id)
        assert len(judge.templates) == 0
        assert not judge.has_template(template.template_id)

    def test_judge_activation_deactivation(self):
        """Test judge activation and deactivation."""
        judge = Judge.create(
            judge_id="test_judge",
            name="Test Judge",
            description="Test",
            model_provider_id="provider",
        )

        # Initially active
        assert judge.is_active

        # Deactivate
        judge.deactivate("Testing deactivation")
        assert not judge.is_active
        assert judge.metadata["deactivation_reason"] == "Testing deactivation"

        # Cannot reactivate without calibration
        with pytest.raises(JudgeNotCalibratedError):
            judge.reactivate()

        # Add calibration and reactivate
        calibration_data = CalibrationData(
            accuracy=Decimal("0.85"),
            consistency=Decimal("0.8"),
            bias_score=Decimal("0.1"),
            confidence_calibration=Decimal("0.75"),
            sample_size=100,
            calibrated_at=datetime.utcnow(),
        )
        judge.calibrate(calibration_data)

        judge.reactivate()
        assert judge.is_active
        assert "deactivation_reason" not in judge.metadata

    def test_performance_tracking(self):
        """Test performance history tracking."""
        judge = Judge.create(
            judge_id="test_judge",
            name="Test Judge",
            description="Test",
            model_provider_id="provider",
        )

        # Initially empty performance history
        assert len(judge.performance_history) == 0

        # Performance stats should handle empty history
        stats = judge.get_performance_stats()
        assert stats["total_evaluations"] == 0
        assert stats["success_rate"] == 0.0
