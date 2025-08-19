"""Integration tests for evaluation domain."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.domain.evaluation.entities.dimension import STANDARD_DIMENSIONS, Dimension
from src.domain.evaluation.entities.evaluation_result import EvaluationResult
from src.domain.evaluation.entities.evaluation_template import EvaluationTemplate
from src.domain.evaluation.entities.judge import Judge
from src.domain.evaluation.services.consensus_algorithm import ConsensusAlgorithm
from src.domain.evaluation.services.judge_calibrator import JudgeCalibrator
from src.domain.evaluation.services.quality_controller import QualityController
from src.domain.evaluation.value_objects.calibration_data import CalibrationData
from src.domain.evaluation.value_objects.scoring_scale import ScoringScale


class TestEvaluationWorkflow:
    """Test complete evaluation workflow integration."""

    def setup_method(self):
        """Set up test fixtures for integration tests."""
        # Create evaluation template
        dimensions = [
            STANDARD_DIMENSIONS["accuracy"],
            STANDARD_DIMENSIONS["relevance"],
            STANDARD_DIMENSIONS["clarity"],
        ]

        # Adjust weights to sum to 1.0
        dimensions[0].weight = Decimal("0.4")  # accuracy
        dimensions[1].weight = Decimal("0.35")  # relevance
        dimensions[2].weight = Decimal("0.25")  # clarity

        self.template = EvaluationTemplate.create(
            name="Integration Test Template",
            description="Template for integration testing",
            dimensions=dimensions,
            prompt_template="""
            Evaluate the following response:
            
            Prompt: {prompt}
            Response: {response}
            
            Evaluation Criteria:
            {dimensions}
            
            Please provide scores for each dimension and detailed reasoning.
            """,
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
            model_parameters={"temperature": 0.1, "max_tokens": 1000},
        )

        # Create judges
        self.judges = []
        for i in range(3):
            judge = Judge.create(
                judge_id=f"judge_{i+1}",
                name=f"Test Judge {i+1}",
                description=f"Judge {i+1} for integration testing",
                model_provider_id="openai_provider",
            )

            # Add template to judge
            judge.add_template(self.template)

            # Add calibration data
            calibration_data = CalibrationData(
                accuracy=Decimal("0.85") + (Decimal("0.05") * i),  # Vary accuracy slightly
                consistency=Decimal("0.8") + (Decimal("0.03") * i),
                bias_score=Decimal("0.05") + (Decimal("0.02") * i),
                confidence_calibration=Decimal("0.75") + (Decimal("0.05") * i),
                sample_size=100 + (10 * i),
                calibrated_at=datetime.utcnow(),
            )
            judge.calibrate(calibration_data)

            self.judges.append(judge)

        # Initialize services
        self.consensus_algorithm = ConsensusAlgorithm()
        self.quality_controller = QualityController()
        self.judge_calibrator = JudgeCalibrator()

    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(self):
        """Test complete evaluation workflow from prompt to consensus."""
        prompt = "What are the main benefits of renewable energy?"
        response = "Renewable energy sources like solar and wind power offer several key benefits: they reduce greenhouse gas emissions, provide energy independence, create jobs in emerging industries, and have lower long-term operational costs compared to fossil fuels."

        # Step 1: Create evaluation results from multiple judges
        evaluation_results = []

        for i, judge in enumerate(self.judges):
            # Mock the model provider call
            judge._call_model_provider = AsyncMock(
                return_value=f"""
            Accuracy: {4 + i % 2} - The response provides factual information about renewable energy benefits.
            Relevance: {4 + i % 2} - Directly addresses the question about benefits.
            Clarity: {4 if i != 1 else 3} - Well-structured and easy to understand.
            
            Overall, this is a solid response that covers the main benefits of renewable energy.
            Confidence: 0.{85 + i*5}
            """
            )

            # Perform evaluation
            result = await judge.evaluate(prompt, response, self.template)

            assert result.is_successful()
            assert result.overall_score > Decimal("0.5")
            assert result.confidence_score > Decimal("0.8")

            evaluation_results.append(result)

        # Step 2: Quality control for each evaluation
        quality_reports = []
        for result in evaluation_results:
            quality_report = self.quality_controller.validate_evaluation_quality(
                result, self.template, "test_controller"
            )
            result.add_quality_report(quality_report)
            quality_reports.append(quality_report)

        # All evaluations should pass quality control
        assert all(report.is_passing() for report in quality_reports)

        # Step 3: Calculate consensus
        consensus = self.consensus_algorithm.calculate_consensus(
            evaluation_results, method="weighted_average", confidence_weighting=True
        )

        assert consensus.is_high_agreement()
        assert consensus.is_statistically_significant()
        assert not consensus.has_outliers()

        # Step 4: Quality control for consensus
        consensus_quality = self.quality_controller.check_consensus_quality(
            consensus, "test_controller"
        )

        assert consensus_quality.is_passing()

        # Verify complete workflow produces valid results
        assert Decimal("0.6") <= consensus.consensus_score <= Decimal("1.0")
        assert consensus.agreement_level > Decimal("0.7")
        assert len(consensus.judge_scores) == 3

    def test_multi_judge_calibration_comparison(self):
        """Test comparing calibration across multiple judges."""
        # Get calibration summaries for all judges
        calibration_summaries = []
        for judge in self.judges:
            summary = self.judge_calibrator.get_calibration_summary(judge)
            calibration_summaries.append(summary)

        # All judges should be calibrated and production ready
        assert all(summary["is_calibrated"] for summary in calibration_summaries)
        assert all(summary["is_production_ready"] for summary in calibration_summaries)

        # Compare judge quality grades
        quality_grades = [summary["quality_grade"] for summary in calibration_summaries]
        assert all(grade in ["GOOD", "EXCELLENT"] for grade in quality_grades)

        # Judge 3 should have highest reliability (set up with best metrics)
        reliability_scores = [
            Decimal(summary["reliability_score"]) for summary in calibration_summaries
        ]
        assert reliability_scores[2] >= reliability_scores[1] >= reliability_scores[0]

    def test_consensus_with_quality_filtering(self):
        """Test consensus calculation with quality-based filtering."""
        # Create evaluation results with varying quality
        evaluation_results = []

        # Good quality result
        good_result = EvaluationResult.create_pending(
            judge_id="good_judge",
            template_id=self.template.template_id,
            prompt="Test prompt",
            response="Test response",
        )
        good_result.complete_evaluation(
            template=self.template,
            dimension_scores={"accuracy": 4, "relevance": 4, "clarity": 4},
            confidence_score=Decimal("0.9"),
            reasoning="This is a comprehensive and well-structured response that directly addresses the question with accurate information. The language is clear and easy to understand.",
            evaluation_time_ms=2000,
        )
        evaluation_results.append(good_result)

        # Medium quality result
        medium_result = EvaluationResult.create_pending(
            judge_id="medium_judge",
            template_id=self.template.template_id,
            prompt="Test prompt",
            response="Test response",
        )
        medium_result.complete_evaluation(
            template=self.template,
            dimension_scores={"accuracy": 3, "relevance": 3, "clarity": 3},
            confidence_score=Decimal("0.7"),
            reasoning="The response is okay and provides some relevant information, though it could be more detailed.",
            evaluation_time_ms=1000,
        )
        evaluation_results.append(medium_result)

        # Poor quality result
        poor_result = EvaluationResult.create_pending(
            judge_id="poor_judge",
            template_id=self.template.template_id,
            prompt="Test prompt",
            response="Test response",
        )
        poor_result.complete_evaluation(
            template=self.template,
            dimension_scores={"accuracy": 2, "relevance": 2, "clarity": 2},
            confidence_score=Decimal("0.3"),  # Low confidence
            reasoning="ok",  # Very brief reasoning
            evaluation_time_ms=200,
        )
        evaluation_results.append(poor_result)

        # Add quality reports
        for result in evaluation_results:
            quality_report = self.quality_controller.validate_evaluation_quality(
                result, self.template, "test_controller"
            )
            result.add_quality_report(quality_report)

        # Filter results based on quality
        high_quality_results = [
            result
            for result in evaluation_results
            if result.quality_report and result.quality_report.is_passing()
        ]

        # Should filter out the poor quality result
        assert len(high_quality_results) == 2  # good and medium results

        # Calculate consensus with filtered results
        consensus = self.consensus_algorithm.calculate_consensus(high_quality_results)

        assert consensus.get_effective_judges_count() == 2
        assert consensus.is_statistically_significant()

    def test_judge_performance_tracking_integration(self):
        """Test integration of judge performance tracking across evaluations."""
        judge = self.judges[0]

        # Simulate multiple evaluations over time
        for i in range(10):
            result = EvaluationResult.create_pending(
                judge_id=judge.judge_id,
                template_id=self.template.template_id,
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
            )

            result.complete_evaluation(
                template=self.template,
                dimension_scores={"accuracy": 4, "relevance": 4, "clarity": 3 + (i % 2)},
                confidence_score=Decimal("0.8") + (Decimal("0.1") * (i % 2)),
                reasoning=f"Detailed reasoning for evaluation {i} with comprehensive analysis.",
                evaluation_time_ms=1000 + (i * 100),
            )

            # Record performance in judge
            judge._record_evaluation_performance(result, self.template)

        # Check performance statistics
        stats = judge.get_performance_stats(days=1)  # Recent performance

        assert stats["total_evaluations"] == 10
        assert stats["successful_evaluations"] == 10
        assert stats["success_rate"] == 1.0
        assert stats["average_confidence"] > 0.7
        assert stats["unique_templates"] == 1

    def test_template_dimension_weight_balancing(self):
        """Test template dimension weight balancing affects consensus."""
        # Create two templates with different weight distributions

        # Template 1: Accuracy heavily weighted
        accuracy_focused_dims = [
            Dimension.create("accuracy", "Accuracy test", Decimal("0.8"), {1: "Poor", 5: "Great"}),
            Dimension.create("clarity", "Clarity test", Decimal("0.2"), {1: "Poor", 5: "Great"}),
        ]

        accuracy_template = EvaluationTemplate.create(
            name="Accuracy Focused Template",
            description="Template prioritizing accuracy",
            dimensions=accuracy_focused_dims,
            prompt_template="Test {prompt} {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        # Template 2: Clarity heavily weighted
        clarity_focused_dims = [
            Dimension.create("accuracy", "Accuracy test", Decimal("0.2"), {1: "Poor", 5: "Great"}),
            Dimension.create("clarity", "Clarity test", Decimal("0.8"), {1: "Poor", 5: "Great"}),
        ]

        clarity_template = EvaluationTemplate.create(
            name="Clarity Focused Template",
            description="Template prioritizing clarity",
            dimensions=clarity_focused_dims,
            prompt_template="Test {prompt} {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        # Create evaluation results with same dimension scores
        dimension_scores = {"accuracy": 5, "clarity": 3}  # High accuracy, medium clarity

        # Result with accuracy-focused template
        accuracy_result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=accuracy_template.template_id,
            prompt="Test",
            response="Test",
        )
        accuracy_result.complete_evaluation(
            template=accuracy_template,
            dimension_scores=dimension_scores,
            confidence_score=Decimal("0.8"),
            reasoning="Test reasoning",
            evaluation_time_ms=1000,
        )

        # Result with clarity-focused template
        clarity_result = EvaluationResult.create_pending(
            judge_id="test_judge",
            template_id=clarity_template.template_id,
            prompt="Test",
            response="Test",
        )
        clarity_result.complete_evaluation(
            template=clarity_template,
            dimension_scores=dimension_scores,
            confidence_score=Decimal("0.8"),
            reasoning="Test reasoning",
            evaluation_time_ms=1000,
        )

        # Accuracy-focused template should yield higher overall score
        # accuracy: 5 (norm 1.0) * 0.8 + clarity: 3 (norm 0.5) * 0.2 = 0.9
        assert accuracy_result.overall_score == Decimal("0.9")

        # Clarity-focused template should yield lower overall score
        # accuracy: 5 (norm 1.0) * 0.2 + clarity: 3 (norm 0.5) * 0.8 = 0.6
        assert clarity_result.overall_score == Decimal("0.6")

    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test consensus calculation with failed evaluations
        failed_result = EvaluationResult.create_pending(
            judge_id="failing_judge",
            template_id=self.template.template_id,
            prompt="Test prompt",
            response="Test response",
        )
        failed_result.fail_evaluation("Simulated evaluation failure")

        successful_result = EvaluationResult.create_pending(
            judge_id="working_judge",
            template_id=self.template.template_id,
            prompt="Test prompt",
            response="Test response",
        )
        successful_result.complete_evaluation(
            template=self.template,
            dimension_scores={"accuracy": 4, "relevance": 4, "clarity": 4},
            confidence_score=Decimal("0.8"),
            reasoning="Good evaluation",
            evaluation_time_ms=1000,
        )

        # Consensus algorithm should handle mixed results gracefully
        mixed_results = [failed_result, successful_result]

        # Should raise error due to insufficient successful results
        with pytest.raises(Exception):  # InsufficientDataError from consensus algorithm
            self.consensus_algorithm.calculate_consensus(mixed_results)

        # Quality controller should properly handle failed evaluations
        failed_quality_report = self.quality_controller.validate_evaluation_quality(
            failed_result, self.template, "test_controller"
        )

        assert not failed_quality_report.is_passing()
        assert failed_quality_report.has_critical_issues()
