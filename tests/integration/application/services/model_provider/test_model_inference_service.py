"""Integration tests for ModelInferenceService and Agent 2.1 integration."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.dto.model_response_dto import ModelResponseDTO, ResponseStatus
from src.application.services.model_provider.model_inference_service import (
    ModelInferenceService,
    TestContext,
)
from src.application.services.model_provider.model_provider_service import ModelProviderService
from src.application.services.model_provider.provider_selector import SelectionCriteria
from src.domain.test_management.entities.test_configuration import TestConfiguration
from src.domain.test_management.entities.test_sample import TestSample


class TestModelInferenceServiceIntegration:
    """Test ModelInferenceService integration with Agent 2.1 workflow."""

    @pytest.fixture
    def mock_model_provider_service(self):
        """Create mock model provider service."""
        service = Mock(spec=ModelProviderService)
        service.process_test_sample = AsyncMock()
        service.call_model_with_intelligent_routing = AsyncMock()
        return service

    @pytest.fixture
    def model_inference_service(self, mock_model_provider_service):
        """Create ModelInferenceService instance."""
        return ModelInferenceService(mock_model_provider_service)

    @pytest.fixture
    def test_context(self):
        """Create test context."""
        return TestContext(
            test_id="test-123",
            test_name="Integration Test",
            sample_id="sample-456",
            model_configurations={
                "gpt-4": {
                    "provider_id": "openai-provider",
                    "model_id": "gpt-4",
                    "parameters": {"temperature": 0.7, "max_tokens": 150},
                }
            },
            evaluation_config={"judge_count": 3, "consensus_threshold": 0.7},
        )

    @pytest.fixture
    def test_sample(self):
        """Create test sample."""
        return TestSample(
            prompt="What is the capital of France?",
            expected_output="Paris",
            metadata={"category": "geography", "difficulty": "easy"},
        )

    @pytest.mark.asyncio
    async def test_single_sample_processing(
        self, model_inference_service, mock_model_provider_service, test_sample, test_context
    ):
        """Test processing a single test sample."""
        # Setup mock response
        mock_response = ModelResponseDTO.create_success_response(
            provider_id="openai-provider",
            model_id="gpt-4",
            text="Paris is the capital of France.",
            input_tokens=20,
            output_tokens=15,
            latency_ms=1500,
        )
        mock_model_provider_service.process_test_sample.return_value = mock_response

        # Create model configuration
        model_config = {
            "provider_id": "openai-provider",
            "model_id": "gpt-4",
            "parameters": {"temperature": 0.7, "max_tokens": 150},
        }

        # Execute
        response = await model_inference_service.process_test_sample(
            sample=test_sample, model_config=model_config, test_context=test_context
        )

        # Verify
        assert response is not None
        assert response.provider_id == "openai-provider"
        assert response.model_id == "gpt-4"
        assert response.is_successful()
        assert "Paris" in response.text

        # Verify service was called correctly
        mock_model_provider_service.process_test_sample.assert_called_once()
        call_args = mock_model_provider_service.process_test_sample.call_args
        assert call_args[1]["sample"] == test_sample
        assert call_args[1]["provider_id"] == "openai-provider"
        assert call_args[1]["model_id"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_invalid_model_configuration(
        self, model_inference_service, test_sample, test_context
    ):
        """Test handling of invalid model configuration."""
        # Create invalid model config (missing required fields)
        invalid_config = {
            "provider_id": "openai-provider",
            # Missing model_id
            "parameters": {"temperature": 0.7},
        }

        # Execute
        response = await model_inference_service.process_test_sample(
            sample=test_sample, model_config=invalid_config, test_context=test_context
        )

        # Verify error handling
        assert response is not None
        assert response.status == ResponseStatus.ERROR
        assert "missing" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_provider_service_error_handling(
        self, model_inference_service, mock_model_provider_service, test_sample, test_context
    ):
        """Test handling of provider service errors."""
        # Setup mock to raise exception
        mock_model_provider_service.process_test_sample.side_effect = Exception("Provider error")

        model_config = {
            "provider_id": "openai-provider",
            "model_id": "gpt-4",
            "parameters": {"temperature": 0.7},
        }

        # Execute
        response = await model_inference_service.process_test_sample(
            sample=test_sample, model_config=model_config, test_context=test_context
        )

        # Verify error response
        assert response is not None
        assert response.status == ResponseStatus.ERROR
        assert "processing failed" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_batch_processing_multiple_models(
        self, model_inference_service, mock_model_provider_service, test_context
    ):
        """Test batch processing with multiple models."""
        # Setup test samples
        samples = [
            TestSample(
                prompt=f"Test question {i}?", expected_output=f"Answer {i}", metadata={"index": i}
            )
            for i in range(3)
        ]

        # Setup model configurations
        model_configs = {
            "gpt-4": {
                "provider_id": "openai-provider",
                "model_id": "gpt-4",
                "parameters": {"temperature": 0.7},
            },
            "claude-3": {
                "provider_id": "anthropic-provider",
                "model_id": "claude-3-sonnet",
                "parameters": {"temperature": 0.7},
            },
        }

        # Setup mock responses
        def mock_process_sample(sample, provider_id, model_id, **kwargs):
            return ModelResponseDTO.create_success_response(
                provider_id=provider_id,
                model_id=model_id,
                text=f"Response from {model_id} for {sample.prompt}",
                input_tokens=20,
                output_tokens=15,
                latency_ms=1200,
            )

        mock_model_provider_service.process_test_sample.side_effect = mock_process_sample

        # Execute
        results = await model_inference_service.process_test_samples_batch(
            samples=samples, model_configs=model_configs, test_context=test_context, batch_size=2
        )

        # Verify results structure
        assert isinstance(results, dict)
        assert "gpt-4" in results
        assert "claude-3" in results
        assert len(results["gpt-4"]) == 3
        assert len(results["claude-3"]) == 3

        # Verify all responses are successful
        for model_responses in results.values():
            for response in model_responses:
                assert response.is_successful()

    @pytest.mark.asyncio
    async def test_intelligent_routing_integration(
        self, model_inference_service, mock_model_provider_service, test_context
    ):
        """Test intelligent provider routing integration."""
        # Setup test samples
        samples = [
            TestSample(
                prompt="What is machine learning?",
                expected_output="Machine learning explanation",
                metadata={"topic": "AI"},
            )
        ]

        # Setup model requirements
        model_requirements = [
            {"model_id": "gpt-4", "parameters": {"temperature": 0.7, "max_tokens": 200}}
        ]

        # Setup mock response
        mock_response = ModelResponseDTO.create_success_response(
            provider_id="intelligent_routing",
            model_id="gpt-4",
            text="Machine learning is a subset of artificial intelligence...",
            input_tokens=25,
            output_tokens=40,
            latency_ms=1800,
        )
        mock_model_provider_service.call_model_with_intelligent_routing.return_value = mock_response

        # Execute
        results = await model_inference_service.process_test_with_intelligent_routing(
            samples=samples,
            model_requirements=model_requirements,
            test_context=test_context,
            selection_criteria=SelectionCriteria.PERFORMANCE_OPTIMIZATION,
        )

        # Verify results
        assert isinstance(results, dict)
        assert "gpt-4" in results
        assert len(results["gpt-4"]) == 1
        assert results["gpt-4"][0].is_successful()

        # Verify intelligent routing was called correctly
        mock_model_provider_service.call_model_with_intelligent_routing.assert_called()

    def test_metrics_calculation(self, model_inference_service):
        """Test comprehensive metrics calculation."""
        # Create mock responses for multiple models
        responses = {
            "gpt-4": [
                ModelResponseDTO.create_success_response(
                    provider_id="openai-provider",
                    model_id="gpt-4",
                    text="Response 1",
                    input_tokens=20,
                    output_tokens=15,
                    latency_ms=1500,
                ),
                ModelResponseDTO.create_success_response(
                    provider_id="openai-provider",
                    model_id="gpt-4",
                    text="Response 2",
                    input_tokens=25,
                    output_tokens=18,
                    latency_ms=1200,
                ),
            ],
            "claude-3": [
                ModelResponseDTO.create_error_response(
                    provider_id="anthropic-provider",
                    model_id="claude-3-sonnet",
                    error_message="Rate limited",
                )
            ],
        }

        # Add cost information
        from decimal import Decimal

        responses["gpt-4"][0].total_cost = Decimal("0.0015")
        responses["gpt-4"][1].total_cost = Decimal("0.0018")

        # Calculate metrics
        metrics = model_inference_service.calculate_test_metrics(responses)

        # Verify overall metrics
        assert metrics["total_requests"] == 3
        assert metrics["successful_requests"] == 2
        assert metrics["overall_success_rate"] == 2 / 3
        assert metrics["models_tested"] == 2
        assert metrics["total_cost"] == 0.0033  # Sum of costs

        # Verify model-specific metrics
        assert "model_metrics" in metrics
        assert "gpt-4" in metrics["model_metrics"]
        assert "claude-3" in metrics["model_metrics"]

        gpt4_metrics = metrics["model_metrics"]["gpt-4"]
        assert gpt4_metrics["request_count"] == 2
        assert gpt4_metrics["successful_requests"] == 2
        assert gpt4_metrics["success_rate"] == 1.0
        assert gpt4_metrics["total_input_tokens"] == 45
        assert gpt4_metrics["total_output_tokens"] == 33

        claude_metrics = metrics["model_metrics"]["claude-3"]
        assert claude_metrics["request_count"] == 1
        assert claude_metrics["successful_requests"] == 0
        assert claude_metrics["success_rate"] == 0.0

    def test_test_context_creation_from_configuration(self, model_inference_service):
        """Test creating TestContext from TestConfiguration."""
        # Create mock test configuration
        mock_config = Mock(spec=TestConfiguration)
        mock_config.model_parameters = {"gpt-4": {"temperature": 0.7, "max_tokens": 150}}
        mock_config.evaluation_template = {"judge_count": 3, "consensus_threshold": 0.8}

        # Create context
        context = model_inference_service.create_test_context_from_configuration(
            test_id="test-789",
            test_name="Configuration Test",
            sample_id="sample-101",
            test_configuration=mock_config,
        )

        # Verify context creation
        assert context.test_id == "test-789"
        assert context.test_name == "Configuration Test"
        assert context.sample_id == "sample-101"
        assert context.model_configurations == {"gpt-4": {"temperature": 0.7, "max_tokens": 150}}
        assert context.evaluation_config["judge_count"] == 3
        assert context.evaluation_config["consensus_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(
        self, model_inference_service, mock_model_provider_service, test_context
    ):
        """Test concurrent processing with batch size limits."""
        # Setup large number of samples
        samples = [
            TestSample(
                prompt=f"Question {i}?", expected_output=f"Answer {i}", metadata={"index": i}
            )
            for i in range(10)
        ]

        model_configs = {
            "test-model": {
                "provider_id": "test-provider",
                "model_id": "test-model",
                "parameters": {"temperature": 0.7},
            }
        }

        # Track concurrent calls
        call_times = []

        async def mock_process_sample(*args, **kwargs):
            call_times.append(datetime.utcnow())
            await asyncio.sleep(0.1)  # Simulate processing time
            return ModelResponseDTO.create_success_response(
                provider_id="test-provider",
                model_id="test-model",
                text="Test response",
                input_tokens=10,
                output_tokens=10,
                latency_ms=100,
            )

        mock_model_provider_service.process_test_sample.side_effect = mock_process_sample

        # Execute with small batch size
        start_time = datetime.utcnow()
        results = await model_inference_service.process_test_samples_batch(
            samples=samples,
            model_configs=model_configs,
            test_context=test_context,
            batch_size=3,  # Small batch size to test concurrency control
        )
        end_time = datetime.utcnow()

        # Verify processing completed
        assert len(results["test-model"]) == 10

        # Verify processing time is reasonable (should be faster than sequential)
        total_time = (end_time - start_time).total_seconds()
        assert total_time < 2.0  # Should be much faster than 10 * 0.1 seconds

    @pytest.mark.asyncio
    async def test_error_recovery_in_batch(
        self, model_inference_service, mock_model_provider_service, test_context
    ):
        """Test error recovery in batch processing."""
        samples = [
            TestSample(prompt="Good question", expected_output="Good answer"),
            TestSample(prompt="Bad question", expected_output="Bad answer"),
            TestSample(prompt="Another good question", expected_output="Another good answer"),
        ]

        model_configs = {
            "test-model": {
                "provider_id": "test-provider",
                "model_id": "test-model",
                "parameters": {},
            }
        }

        # Setup mixed success/failure responses
        def mock_process_sample(sample, **kwargs):
            if "Bad" in sample.prompt:
                raise Exception("Simulated processing error")
            return ModelResponseDTO.create_success_response(
                provider_id="test-provider",
                model_id="test-model",
                text=f"Response to: {sample.prompt}",
                input_tokens=10,
                output_tokens=10,
                latency_ms=100,
            )

        mock_model_provider_service.process_test_sample.side_effect = mock_process_sample

        # Execute
        results = await model_inference_service.process_test_samples_batch(
            samples=samples, model_configs=model_configs, test_context=test_context
        )

        # Verify mixed results
        responses = results["test-model"]
        assert len(responses) == 3

        # Check that good samples succeeded and bad sample failed
        assert responses[0].is_successful()  # Good question
        assert responses[1].status == ResponseStatus.ERROR  # Bad question
        assert responses[2].is_successful()  # Another good question

    @pytest.mark.asyncio
    async def test_context_propagation(
        self, model_inference_service, mock_model_provider_service, test_context
    ):
        """Test that test context is properly propagated through the call chain."""
        test_sample = TestSample(
            prompt="Context test",
            expected_output="Context response",
            metadata={"test_key": "test_value"},
        )

        model_config = {
            "provider_id": "test-provider",
            "model_id": "test-model",
            "parameters": {"temperature": 0.5},
        }

        # Mock response
        mock_response = ModelResponseDTO.create_success_response(
            provider_id="test-provider",
            model_id="test-model",
            text="Context response",
            input_tokens=10,
            output_tokens=10,
            latency_ms=100,
        )
        mock_model_provider_service.process_test_sample.return_value = mock_response

        # Execute
        await model_inference_service.process_test_sample(
            sample=test_sample, model_config=model_config, test_context=test_context
        )

        # Verify context was passed to provider service
        mock_model_provider_service.process_test_sample.assert_called_once()
        call_kwargs = mock_model_provider_service.process_test_sample.call_args[1]

        assert "test_context" in call_kwargs
        provider_context = call_kwargs["test_context"]
        assert provider_context["test_id"] == test_context.test_id
        assert provider_context["test_name"] == test_context.test_name
        assert provider_context["sample_id"] == str(test_sample.id)


class TestContextIntegration:
    """Test TestContext integration."""

    def test_test_context_creation(self):
        """Test TestContext creation and attributes."""
        context = TestContext(
            test_id="test-123",
            test_name="Test Name",
            sample_id="sample-456",
            model_configurations={"gpt-4": {"temp": 0.7}},
            evaluation_config={"judges": 3},
        )

        assert context.test_id == "test-123"
        assert context.test_name == "Test Name"
        assert context.sample_id == "sample-456"
        assert context.model_configurations == {"gpt-4": {"temp": 0.7}}
        assert context.evaluation_config == {"judges": 3}
        assert isinstance(context.created_at, datetime)

    def test_test_context_defaults(self):
        """Test TestContext with default values."""
        context = TestContext(
            test_id="test-123",
            test_name="Test Name",
            sample_id="sample-456",
            model_configurations={},
        )

        assert context.evaluation_config == {}
        assert isinstance(context.created_at, datetime)
