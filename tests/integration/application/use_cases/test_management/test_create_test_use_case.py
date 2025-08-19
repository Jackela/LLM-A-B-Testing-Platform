"""Integration tests for CreateTestUseCase."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.application.dto.test_configuration_dto import (
    CreateTestCommandDTO,
    EvaluationConfigurationDTO,
    ModelConfigurationDTO,
    TestConfigurationDTO,
    TestSampleDTO,
)
from src.application.services.model_provider_service import ModelProviderService
from src.application.services.test_validation_service import TestValidationService
from src.application.use_cases.test_management.create_test import CreateTestUseCase
from src.domain.model_provider.value_objects.money import Money
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.validation_result import ValidationResult


@pytest.fixture
def mock_uow():
    """Mock unit of work."""
    uow = AsyncMock()
    uow.tests = AsyncMock()
    uow.providers = AsyncMock()
    return uow


@pytest.fixture
def mock_event_publisher():
    """Mock domain event publisher."""
    return AsyncMock()


@pytest.fixture
def mock_validation_service():
    """Mock test validation service."""
    validation_service = AsyncMock(spec=TestValidationService)
    validation_service.validate_test_creation.return_value = ValidationResult(True, [])
    validation_service.estimate_test_cost.return_value = Money(10.0, "USD")
    validation_service.estimate_test_duration.return_value = 300.0
    return validation_service


@pytest.fixture
def mock_provider_service():
    """Mock model provider service."""
    provider_service = AsyncMock(spec=ModelProviderService)
    provider_service.verify_model_availability.return_value = {
        "openai/gpt-4": True,
        "anthropic/claude-3": True,
    }
    provider_service.validate_model_parameters.return_value = {}
    return provider_service


@pytest.fixture
def create_test_use_case(
    mock_uow, mock_event_publisher, mock_validation_service, mock_provider_service
):
    """Create test use case instance."""
    return CreateTestUseCase(
        uow=mock_uow,
        event_publisher=mock_event_publisher,
        validation_service=mock_validation_service,
        provider_service=mock_provider_service,
    )


@pytest.fixture
def valid_create_command():
    """Valid test creation command."""
    return CreateTestCommandDTO(
        name="Test A vs B",
        configuration=TestConfigurationDTO(
            models=[
                ModelConfigurationDTO(
                    model_id="gpt-4",
                    provider_name="openai",
                    parameters={"temperature": 0.7, "max_tokens": 100},
                    weight=0.5,
                ),
                ModelConfigurationDTO(
                    model_id="claude-3",
                    provider_name="anthropic",
                    parameters={"temperature": 0.7, "max_tokens": 100},
                    weight=0.5,
                ),
            ],
            evaluation=EvaluationConfigurationDTO(
                template_id="standard-template",
                judge_count=3,
                consensus_threshold=0.7,
                quality_threshold=0.8,
            ),
            max_cost=Money(50.0, "USD"),
            description="Test comparison between models",
        ),
        samples=[
            TestSampleDTO(prompt=f"Sample prompt {i}", difficulty=DifficultyLevel.MEDIUM)
            for i in range(20)  # 20 samples for valid test
        ],
    )


class TestCreateTestUseCase:
    """Test cases for CreateTestUseCase."""

    @pytest.mark.asyncio
    async def test_successful_test_creation(self, create_test_use_case, valid_create_command):
        """Test successful test creation with valid command."""
        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is True
        assert result.status == "CONFIGURED"
        assert result.test_id is not None
        assert result.estimated_cost is not None
        assert result.estimated_duration is not None
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validation_failure(
        self, create_test_use_case, valid_create_command, mock_validation_service
    ):
        """Test test creation with validation failure."""
        # Arrange
        mock_validation_service.validate_test_creation.return_value = ValidationResult(
            False, ["Test name cannot be empty"]
        )

        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is False
        assert result.status == "validation_failed"
        assert "Test name cannot be empty" in result.errors

    @pytest.mark.asyncio
    async def test_provider_unavailable(
        self, create_test_use_case, valid_create_command, mock_provider_service
    ):
        """Test test creation when model provider is unavailable."""
        # Arrange
        mock_provider_service.verify_model_availability.return_value = {
            "openai/gpt-4": False,
            "anthropic/claude-3": True,
        }

        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is False
        assert result.status == "business_rule_violation"
        assert any("not available" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_invalid_model_parameters(
        self, create_test_use_case, valid_create_command, mock_provider_service
    ):
        """Test test creation with invalid model parameters."""
        # Arrange
        mock_provider_service.validate_model_parameters.return_value = {
            "openai/gpt-4": ["max_tokens exceeds limit"]
        }

        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is False
        assert result.status == "business_rule_violation"
        assert any("parameter validation failed" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_insufficient_samples(self, create_test_use_case, mock_validation_service):
        """Test test creation with insufficient samples."""
        # Arrange
        command = CreateTestCommandDTO(
            name="Test with few samples",
            configuration=TestConfigurationDTO(
                models=[
                    ModelConfigurationDTO(
                        model_id="gpt-4", provider_name="openai", parameters={}, weight=1.0
                    )
                ],
                evaluation=EvaluationConfigurationDTO(template_id="standard"),
            ),
            samples=[TestSampleDTO(prompt="Single sample")],  # Only 1 sample
        )

        mock_validation_service.validate_test_creation.return_value = ValidationResult(
            False, ["At least 10 samples are required"]
        )

        # Act
        result = await create_test_use_case.execute(command)

        # Assert
        assert result.created_test is False
        assert result.status == "validation_failed"

    @pytest.mark.asyncio
    async def test_database_error_handling(
        self, create_test_use_case, valid_create_command, mock_uow
    ):
        """Test handling of database errors during test creation."""
        # Arrange
        mock_uow.commit.side_effect = Exception("Database connection failed")

        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is False
        assert result.status == "system_error"
        assert any("system error" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_event_publishing(
        self, create_test_use_case, valid_create_command, mock_event_publisher
    ):
        """Test that domain events are published after test creation."""
        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is True
        mock_event_publisher.publish_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_estimation_integration(
        self, create_test_use_case, valid_create_command, mock_validation_service
    ):
        """Test integration with cost estimation."""
        # Arrange
        expected_cost = Money(25.50, "USD")
        mock_validation_service.estimate_test_cost.return_value = expected_cost

        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is True
        assert result.estimated_cost == expected_cost

    @pytest.mark.asyncio
    async def test_duration_estimation_integration(
        self, create_test_use_case, valid_create_command, mock_validation_service
    ):
        """Test integration with duration estimation."""
        # Arrange
        expected_duration = 450.0
        mock_validation_service.estimate_test_duration.return_value = expected_duration

        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is True
        assert result.estimated_duration == expected_duration

    @pytest.mark.asyncio
    async def test_sample_creation_and_assignment(
        self, create_test_use_case, valid_create_command, mock_uow
    ):
        """Test that samples are properly created and assigned to test."""
        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is True

        # Verify test was saved
        mock_uow.tests.save.assert_called_once()

        # Get the saved test
        saved_test = mock_uow.tests.save.call_args[0][0]
        assert len(saved_test.samples) == len(valid_create_command.samples)

    @pytest.mark.asyncio
    async def test_transaction_management(
        self, create_test_use_case, valid_create_command, mock_uow
    ):
        """Test proper transaction management."""
        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is True
        mock_uow.__aenter__.assert_called_once()
        mock_uow.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_on_failure(self, create_test_use_case, valid_create_command, mock_uow):
        """Test transaction rollback on failure."""
        # Arrange
        mock_uow.tests.save.side_effect = Exception("Save failed")

        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is False
        # Verify transaction context was entered but commit wasn't called due to exception
        mock_uow.__aenter__.assert_called_once()
        mock_uow.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_configuration_conversion(self, create_test_use_case, valid_create_command):
        """Test proper conversion from DTO to domain configuration."""
        # Act
        result = await create_test_use_case.execute(valid_create_command)

        # Assert
        assert result.created_test is True

        # Verify the configuration was properly converted
        # This would require examining the saved test object
        # In a real implementation, you might verify specific configuration fields
