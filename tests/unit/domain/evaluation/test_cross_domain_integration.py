"""Cross-domain integration tests for evaluation domain."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.domain.evaluation.entities.dimension import STANDARD_DIMENSIONS
from src.domain.evaluation.entities.evaluation_result import EvaluationResult
from src.domain.evaluation.entities.evaluation_template import EvaluationTemplate

# Evaluation domain imports
from src.domain.evaluation.entities.judge import Judge
from src.domain.evaluation.services.consensus_algorithm import ConsensusAlgorithm
from src.domain.evaluation.value_objects.calibration_data import CalibrationData
from src.domain.evaluation.value_objects.scoring_scale import ScoringScale
from src.domain.model_provider.entities.model_config import ModelConfig

# Model Provider domain imports (integrated with Evaluation)
from src.domain.model_provider.entities.model_response import ModelResponse
from src.domain.model_provider.value_objects.money import Money
from src.domain.test_management.entities.test_configuration import TestConfiguration

# Test Management domain imports (integrated with Evaluation)
from src.domain.test_management.entities.test_sample import TestSample
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel


class TestEvaluationTestManagementIntegration:
    """Test integration between Evaluation and Test Management domains."""

    def setup_method(self):
        """Set up test fixtures for cross-domain integration."""
        # Create evaluation template
        dimensions = [STANDARD_DIMENSIONS["accuracy"], STANDARD_DIMENSIONS["relevance"]]
        dimensions[0].weight = Decimal("0.6")
        dimensions[1].weight = Decimal("0.4")

        self.evaluation_template = EvaluationTemplate.create(
            name="Test Integration Template",
            description="Template for testing cross-domain integration",
            dimensions=dimensions,
            prompt_template="Evaluate the model's response to: {prompt}\nResponse: {response}\n{dimensions}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        # Create calibrated judge
        self.judge = Judge.create(
            judge_id="integration_judge",
            name="Integration Test Judge",
            description="Judge for cross-domain integration testing",
            model_provider_id="openai_provider",
        )

        calibration_data = CalibrationData(
            accuracy=Decimal("0.88"),
            consistency=Decimal("0.85"),
            bias_score=Decimal("0.05"),
            confidence_calibration=Decimal("0.8"),
            sample_size=100,
            calibrated_at=datetime.utcnow(),
        )
        self.judge.calibrate(calibration_data)
        self.judge.add_template(self.evaluation_template)

    def test_evaluate_test_sample_integration(self):
        """Test evaluation of test management samples."""
        # Create test sample from Test Management domain
        test_sample = TestSample(
            prompt="What is the capital of France?",
            difficulty=DifficultyLevel.EASY,
            expected_output="Paris is the capital of France.",
            tags=["geography", "factual"],
            metadata={"category": "world_capitals", "source": "integration_test"},
        )

        # Simulate model response evaluation
        model_response = "Paris is the capital of France. It is located in the north-central part of the country."

        # Create evaluation result using test sample data
        evaluation_result = EvaluationResult.create_pending(
            judge_id=self.judge.judge_id,
            template_id=self.evaluation_template.template_id,
            prompt=test_sample.prompt,
            response=model_response,
        )

        # Complete evaluation with high scores for accurate response
        evaluation_result.complete_evaluation(
            template=self.evaluation_template,
            dimension_scores={"accuracy": 5, "relevance": 5},
            confidence_score=Decimal("0.95"),
            reasoning="The response correctly identifies Paris as the capital of France and provides additional relevant geographical context.",
            evaluation_time_ms=1200,
        )

        # Verify evaluation integrates with test sample
        assert evaluation_result.prompt == test_sample.prompt
        assert evaluation_result.is_successful()
        assert evaluation_result.overall_score == Decimal("1.0")  # Perfect score

        # Test sample can store evaluation results
        test_sample.add_evaluation_result(
            self.judge.judge_id,
            {
                "score": float(evaluation_result.overall_score),
                "confidence": float(evaluation_result.confidence_score),
                "reasoning": evaluation_result.reasoning,
                "dimension_scores": evaluation_result.dimension_scores,
                "evaluation_time_ms": evaluation_result.evaluation_time_ms,
            },
        )

        # Verify test sample integration
        assert test_sample.has_evaluation_for_model(self.judge.judge_id)
        stored_result = test_sample.get_evaluation_result(self.judge.judge_id)
        assert stored_result["score"] == 1.0
        assert stored_result["confidence"] == 0.95

    def test_difficulty_level_evaluation_correlation(self):
        """Test correlation between test difficulty and evaluation complexity."""
        # Create test samples with different difficulties
        test_samples = [
            TestSample(
                prompt="What is 2 + 2?", difficulty=DifficultyLevel.EASY, expected_output="4"
            ),
            TestSample(
                prompt="Explain the principles of quantum mechanics and their applications in modern technology.",
                difficulty=DifficultyLevel.HARD,
                expected_output="Quantum mechanics is based on principles like superposition and entanglement...",
            ),
            TestSample(
                prompt="Derive the Schrödinger equation from first principles and explain its physical interpretation in the context of wave-particle duality.",
                difficulty=DifficultyLevel.EXPERT,
                expected_output="The Schrödinger equation can be derived by considering the de Broglie wavelength...",
            ),
        ]

        # Simulate model responses with varying quality
        model_responses = [
            "2 + 2 equals 4",  # Simple, accurate
            "Quantum mechanics involves particles behaving as waves, with applications in computing and cryptography.",  # Good but incomplete
            "The equation involves wave functions and energy operators, relating to particle-wave behavior.",  # Incomplete for expert level
        ]

        evaluation_results = []

        for i, (sample, response) in enumerate(zip(test_samples, model_responses)):
            result = EvaluationResult.create_pending(
                judge_id=self.judge.judge_id,
                template_id=self.evaluation_template.template_id,
                prompt=sample.prompt,
                response=response,
            )

            # Scores should generally decrease with difficulty relative to response quality
            if sample.difficulty == DifficultyLevel.EASY:
                scores = {"accuracy": 5, "relevance": 5}  # Perfect for simple math
                confidence = Decimal("0.95")
            elif sample.difficulty == DifficultyLevel.HARD:
                scores = {"accuracy": 3, "relevance": 4}  # Partial credit
                confidence = Decimal("0.75")
            else:  # EXPERT
                scores = {"accuracy": 2, "relevance": 3}  # Insufficient for expert level
                confidence = Decimal("0.6")

            result.complete_evaluation(
                template=self.evaluation_template,
                dimension_scores=scores,
                confidence_score=confidence,
                reasoning=f"Evaluation for {sample.difficulty.value} difficulty question",
                evaluation_time_ms=1000 + (i * 500),  # More time for harder questions
            )

            evaluation_results.append((sample, result))

        # Verify correlation between difficulty and scores
        easy_result = evaluation_results[0][1]
        hard_result = evaluation_results[1][1]
        expert_result = evaluation_results[2][1]

        # Easy question should have highest score
        assert easy_result.overall_score > hard_result.overall_score
        assert hard_result.overall_score > expert_result.overall_score

        # Confidence should also correlate with difficulty/quality match
        assert easy_result.confidence_score > hard_result.confidence_score
        assert hard_result.confidence_score > expert_result.confidence_score

    def test_test_configuration_evaluation_settings(self):
        """Test integration with test configuration settings."""
        # Create test configuration from Test Management domain
        test_config = TestConfiguration(
            name="Integration Test Configuration",
            description="Configuration for cross-domain integration testing",
            sample_size=100,
            difficulty_distribution={
                DifficultyLevel.EASY: 0.3,
                DifficultyLevel.MEDIUM: 0.4,
                DifficultyLevel.HARD: 0.2,
                DifficultyLevel.EXPERT: 0.1,
            },
            tags=["integration", "cross_domain"],
            metadata={
                "evaluation_template_id": str(self.evaluation_template.template_id),
                "judge_requirements": {
                    "min_accuracy": 0.85,
                    "min_consistency": 0.8,
                    "max_bias": 0.1,
                },
                "consensus_settings": {
                    "min_judges": 3,
                    "agreement_threshold": 0.75,
                    "outlier_detection": True,
                },
            },
        )

        # Verify judge meets configuration requirements
        judge_requirements = test_config.metadata["judge_requirements"]

        assert self.judge.calibration_data.accuracy >= Decimal(
            str(judge_requirements["min_accuracy"])
        )
        assert self.judge.calibration_data.consistency >= Decimal(
            str(judge_requirements["min_consistency"])
        )
        assert abs(self.judge.calibration_data.bias_score) <= Decimal(
            str(judge_requirements["max_bias"])
        )

        # Test that evaluation template matches configuration
        configured_template_id = test_config.metadata["evaluation_template_id"]
        assert str(self.evaluation_template.template_id) == configured_template_id

        # Verify consensus settings can be applied
        consensus_settings = test_config.metadata["consensus_settings"]
        assert consensus_settings["min_judges"] <= 3  # We have 1 judge, would need more in practice
        assert 0 <= consensus_settings["agreement_threshold"] <= 1
        assert isinstance(consensus_settings["outlier_detection"], bool)


class TestEvaluationModelProviderIntegration:
    """Test integration between Evaluation and Model Provider domains."""

    def setup_method(self):
        """Set up test fixtures for model provider integration."""
        self.judge = Judge.create(
            judge_id="model_integration_judge",
            name="Model Integration Judge",
            description="Judge for model provider integration testing",
            model_provider_id="test_provider",
        )

        # Add calibration
        calibration_data = CalibrationData(
            accuracy=Decimal("0.9"),
            consistency=Decimal("0.85"),
            bias_score=Decimal("0.02"),
            confidence_calibration=Decimal("0.85"),
            sample_size=150,
            calibrated_at=datetime.utcnow(),
        )
        self.judge.calibrate(calibration_data)

    def test_model_response_evaluation_integration(self):
        """Test evaluation of ModelResponse entities."""
        # Create model configuration from Model Provider domain
        model_config = ModelConfig(
            model_id="gpt-4",
            provider_type="openai",
            parameters={"temperature": 0.1, "max_tokens": 1000, "top_p": 0.9},
            cost_per_input_token=Money(Decimal("0.00003"), "USD"),
            cost_per_output_token=Money(Decimal("0.00006"), "USD"),
        )

        # Create model response
        model_response = ModelResponse.create_pending(
            model_config=model_config, prompt="Explain the concept of machine learning."
        )

        # Complete the model response
        response_text = "Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn patterns and make predictions."

        model_response.complete_response(
            response_text=response_text,
            input_tokens=50,
            output_tokens=150,
            latency_ms=2500,
            metadata={"model_version": "gpt-4-0613"},
        )

        # Create evaluation template for model response evaluation
        dimensions = [STANDARD_DIMENSIONS["accuracy"], STANDARD_DIMENSIONS["clarity"]]
        dimensions[0].weight = Decimal("0.7")
        dimensions[1].weight = Decimal("0.3")

        template = EvaluationTemplate.create(
            name="Model Response Evaluation",
            description="Template for evaluating model responses",
            dimensions=dimensions,
            prompt_template="Evaluate this AI model response:\nPrompt: {prompt}\nResponse: {response}\n{dimensions}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )

        self.judge.add_template(template)

        # Create evaluation result using model response data
        evaluation_result = EvaluationResult.create_pending(
            judge_id=self.judge.judge_id,
            template_id=template.template_id,
            prompt=model_response.prompt,
            response=model_response.response_text,
        )

        # Complete evaluation
        evaluation_result.complete_evaluation(
            template=template,
            dimension_scores={"accuracy": 4, "clarity": 5},
            confidence_score=Decimal("0.88"),
            reasoning="The response provides an accurate definition of machine learning and explains key concepts clearly. The explanation is well-structured and accessible.",
            evaluation_time_ms=1800,
        )

        # Verify integration between domains
        assert evaluation_result.prompt == model_response.prompt
        assert evaluation_result.response == model_response.response_text
        assert evaluation_result.is_successful()

        # Verify performance correlation
        assert (
            evaluation_result.evaluation_time_ms < model_response.latency_ms
        )  # Evaluation faster than generation

        # Calculate cost-quality ratio
        response_cost = model_response.calculate_cost()
        quality_score = evaluation_result.overall_score
        cost_quality_ratio = float(response_cost.amount) / float(quality_score)

        # Should have reasonable cost per quality point
        assert cost_quality_ratio > 0  # Positive cost
        assert quality_score > Decimal("0.7")  # Good quality

    def test_model_performance_evaluation_correlation(self):
        """Test correlation between model performance metrics and evaluation scores."""
        # Create model responses with different performance characteristics
        model_configs = [
            # Fast, cheap model
            ModelConfig(
                model_id="gpt-3.5-turbo",
                provider_type="openai",
                parameters={"temperature": 0.1},
                cost_per_input_token=Money(Decimal("0.000001"), "USD"),
                cost_per_output_token=Money(Decimal("0.000002"), "USD"),
            ),
            # Slower, more expensive model
            ModelConfig(
                model_id="gpt-4",
                provider_type="openai",
                parameters={"temperature": 0.1},
                cost_per_input_token=Money(Decimal("0.00003"), "USD"),
                cost_per_output_token=Money(Decimal("0.00006"), "USD"),
            ),
        ]

        model_responses = []
        evaluation_results = []

        prompt = "Analyze the economic implications of renewable energy adoption."

        for i, config in enumerate(model_configs):
            # Create model response
            response = ModelResponse.create_pending(config, prompt)

            if config.model_id == "gpt-3.5-turbo":
                # Simulate faster but less detailed response
                response.complete_response(
                    response_text="Renewable energy can reduce costs and create jobs, but requires initial investment.",
                    input_tokens=40,
                    output_tokens=80,
                    latency_ms=800,
                    metadata={"quality": "basic"},
                )
                eval_scores = {"accuracy": 3, "clarity": 3}
                eval_confidence = Decimal("0.7")
            else:  # gpt-4
                # Simulate slower but more comprehensive response
                response.complete_response(
                    response_text="The adoption of renewable energy sources presents significant economic implications across multiple sectors. Initial capital investments are substantial but lead to reduced operational costs, job creation in emerging industries, energy independence benefits, and long-term economic sustainability through reduced environmental externalities.",
                    input_tokens=40,
                    output_tokens=200,
                    latency_ms=3000,
                    metadata={"quality": "comprehensive"},
                )
                eval_scores = {"accuracy": 5, "clarity": 4}
                eval_confidence = Decimal("0.9")

            model_responses.append(response)

            # Create corresponding evaluation
            template = EvaluationTemplate.create(
                name=f"Eval Template {i}",
                description="Test template",
                dimensions=[STANDARD_DIMENSIONS["accuracy"], STANDARD_DIMENSIONS["clarity"]],
                prompt_template="Evaluate: {prompt}\nResponse: {response}",
                scoring_scale=ScoringScale.create_five_point_likert(),
                judge_model_id="gpt-4",
            )
            template.dimensions[0].weight = Decimal("0.6")
            template.dimensions[1].weight = Decimal("0.4")

            eval_result = EvaluationResult.create_pending(
                judge_id=self.judge.judge_id,
                template_id=template.template_id,
                prompt=prompt,
                response=response.response_text,
            )

            eval_result.complete_evaluation(
                template=template,
                dimension_scores=eval_scores,
                confidence_score=eval_confidence,
                reasoning=f"Evaluation of {config.model_id} response quality",
                evaluation_time_ms=1000,
            )

            evaluation_results.append(eval_result)

        # Verify correlations
        gpt35_response, gpt4_response = model_responses
        gpt35_eval, gpt4_eval = evaluation_results

        # GPT-4 should have higher evaluation scores
        assert gpt4_eval.overall_score > gpt35_eval.overall_score
        assert gpt4_eval.confidence_score > gpt35_eval.confidence_score

        # GPT-4 should be slower and more expensive
        assert gpt4_response.latency_ms > gpt35_response.latency_ms
        assert gpt4_response.calculate_cost().amount > gpt35_response.calculate_cost().amount

        # But GPT-4 should have better cost-effectiveness relative to quality
        gpt35_cost_per_quality = float(gpt35_response.calculate_cost().amount) / float(
            gpt35_eval.overall_score
        )
        gpt4_cost_per_quality = float(gpt4_response.calculate_cost().amount) / float(
            gpt4_eval.overall_score
        )

        # This ratio could go either way depending on the specific use case
        # The important thing is that we can calculate and compare these metrics
        assert gpt35_cost_per_quality > 0
        assert gpt4_cost_per_quality > 0

    def test_judge_model_provider_consistency(self):
        """Test consistency between judge's model provider and actual evaluations."""
        # Judge should use consistent model provider
        assert self.judge.model_provider_id == "test_provider"

        # Create template that references same provider
        template = EvaluationTemplate.create(
            name="Provider Consistency Test",
            description="Test provider consistency",
            dimensions=[STANDARD_DIMENSIONS["accuracy"]],
            prompt_template="Evaluate: {prompt} -> {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",  # Should match judge's provider capabilities
            model_parameters={"temperature": 0.1, "max_tokens": 500},
        )

        self.judge.add_template(template)

        # Verify template is compatible with judge
        assert template in self.judge.templates
        assert self.judge.has_template(template.template_id)

        # Mock evaluation process
        self.judge._call_model_provider = AsyncMock(return_value="Mocked evaluation response")

        # The async evaluation would call the model provider
        # This tests the integration contract without actual API calls
        assert hasattr(self.judge, "_call_model_provider")

        # Verify template parameters are preserved for provider calls
        assert template.model_parameters["temperature"] == 0.1
        assert template.model_parameters["max_tokens"] == 500
        assert template.judge_model_id == "gpt-4"


class TestMultiDomainConsensusWorkflow:
    """Test consensus calculation across multiple domain integrations."""

    def test_cross_domain_consensus_calculation(self):
        """Test consensus calculation using data from multiple domains."""
        # Create test samples from Test Management domain
        test_samples = [
            TestSample(
                prompt="What is the speed of light?",
                difficulty=DifficultyLevel.EASY,
                expected_output="The speed of light is approximately 299,792,458 meters per second.",
            )
        ]

        # Create model responses from Model Provider domain
        model_configs = [
            ModelConfig("gpt-3.5-turbo", "openai", {"temperature": 0.1}),
            ModelConfig("gpt-4", "openai", {"temperature": 0.1}),
            ModelConfig("claude-3", "anthropic", {"temperature": 0.1}),
        ]

        model_responses = []
        for config in model_configs:
            response = ModelResponse.create_pending(config, test_samples[0].prompt)
            response.complete_response(
                response_text="The speed of light in vacuum is 299,792,458 meters per second (approximately 300,000 km/s).",
                input_tokens=25,
                output_tokens=60,
                latency_ms=1200,
            )
            model_responses.append(response)

        # Create judges from Evaluation domain
        judges = []
        for i, config in enumerate(model_configs):
            judge = Judge.create(
                judge_id=f"judge_{config.model_id}",
                name=f"Judge using {config.model_id}",
                description=f"Judge backed by {config.model_id}",
                model_provider_id=config.provider_type,
            )

            # Calibrate each judge
            calibration = CalibrationData(
                accuracy=Decimal("0.85") + (Decimal("0.02") * i),
                consistency=Decimal("0.8") + (Decimal("0.01") * i),
                bias_score=Decimal("0.05") - (Decimal("0.01") * i),
                confidence_calibration=Decimal("0.75") + (Decimal("0.03") * i),
                sample_size=100,
                calibrated_at=datetime.utcnow(),
            )
            judge.calibrate(calibration)
            judges.append(judge)

        # Create evaluation template
        template = EvaluationTemplate.create(
            name="Multi-Domain Integration Template",
            description="Template for multi-domain consensus testing",
            dimensions=[STANDARD_DIMENSIONS["accuracy"], STANDARD_DIMENSIONS["relevance"]],
            prompt_template="Evaluate response to: {prompt}\nResponse: {response}",
            scoring_scale=ScoringScale.create_five_point_likert(),
            judge_model_id="gpt-4",
        )
        template.dimensions[0].weight = Decimal("0.7")
        template.dimensions[1].weight = Decimal("0.3")

        # Create evaluation results
        evaluation_results = []
        for i, (judge, model_response) in enumerate(zip(judges, model_responses)):
            result = EvaluationResult.create_pending(
                judge_id=judge.judge_id,
                template_id=template.template_id,
                prompt=model_response.prompt,
                response=model_response.response_text,
            )

            # All should score well on this factual question
            result.complete_evaluation(
                template=template,
                dimension_scores={"accuracy": 5, "relevance": 4 + (i % 2)},
                confidence_score=Decimal("0.85") + (Decimal("0.05") * i),
                reasoning=f"Accurate scientific fact provided by {judge.name}",
                evaluation_time_ms=1000 + (i * 200),
            )
            evaluation_results.append(result)

        # Calculate consensus across all domains
        consensus_algorithm = ConsensusAlgorithm()
        consensus = consensus_algorithm.calculate_consensus(
            evaluation_results, method="weighted_average", confidence_weighting=True
        )

        # Verify cross-domain consensus
        assert len(consensus.judge_scores) == 3
        assert consensus.is_high_agreement()
        assert consensus.consensus_score > Decimal("0.8")  # High score for accurate responses
        assert not consensus.has_outliers()  # All judges should agree on factual content

        # Verify integration preserved domain-specific data
        for result in evaluation_results:
            assert result.prompt == test_samples[0].prompt  # From Test Management
            assert "299,792,458" in result.response  # From Model Provider
            assert result.judge_id.startswith("judge_")  # From Evaluation domain
            assert result.overall_score > Decimal("0.7")  # Good evaluation score

        # Verify consensus metadata includes cross-domain information
        consensus_metadata = {
            "test_sample_difficulty": test_samples[0].difficulty.value,
            "model_providers": [config.provider_type for config in model_configs],
            "judge_calibration_quality": [
                judge.calibration_data.get_quality_grade() for judge in judges
            ],
            "average_model_latency": sum(resp.latency_ms for resp in model_responses)
            / len(model_responses),
        }

        # All integration points should be functional
        assert consensus_metadata["test_sample_difficulty"] == "easy"
        assert set(consensus_metadata["model_providers"]) == {"openai", "anthropic"}
        assert all(
            grade in ["GOOD", "EXCELLENT"]
            for grade in consensus_metadata["judge_calibration_quality"]
        )
        assert consensus_metadata["average_model_latency"] > 0
