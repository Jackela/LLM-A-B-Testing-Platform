"""Tests for Test Management domain entities."""

from datetime import datetime, timedelta
from typing import List
from uuid import UUID, uuid4

import pytest

from src.domain.test_management.entities.test import Test
from src.domain.test_management.entities.test_configuration import TestConfiguration
from src.domain.test_management.entities.test_sample import TestSample
from src.domain.test_management.events.test_events import TestCompleted, TestCreated, TestStarted
from src.domain.test_management.exceptions import (
    BusinessRuleViolation,
    InvalidStateTransition,
    ValidationError,
)
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.test_status import TestStatus


class TestTestConfiguration:
    """Tests for TestConfiguration entity."""

    def test_test_configuration_creation(self):
        """Test creating valid test configuration."""
        models = ["gpt-4", "claude-3"]
        config = TestConfiguration(
            models=models, max_tokens=1000, temperature=0.7, timeout_seconds=30
        )
        assert config.models == models
        assert config.max_tokens == 1000
        assert config.temperature == 0.7
        assert config.timeout_seconds == 30

    def test_test_configuration_requires_minimum_two_models(self):
        """Test that configuration requires at least 2 models."""
        with pytest.raises(BusinessRuleViolation):
            TestConfiguration(models=["gpt-4"], max_tokens=1000, temperature=0.7)  # Only one model

    def test_test_configuration_validates_temperature_range(self):
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            TestConfiguration(
                models=["gpt-4", "claude-3"],
                max_tokens=1000,
                temperature=2.0,  # Invalid temperature > 1.0
            )

        with pytest.raises(ValidationError):
            TestConfiguration(
                models=["gpt-4", "claude-3"],
                max_tokens=1000,
                temperature=-0.1,  # Invalid temperature < 0.0
            )

    def test_test_configuration_validates_max_tokens(self):
        """Test max_tokens validation."""
        with pytest.raises(ValidationError):
            TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=0)  # Invalid max_tokens

    def test_test_configuration_validation_result(self):
        """Test configuration validation method."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        result = config.validate()
        assert result.is_valid is True
        assert result.errors == ()

    def test_test_configuration_validation_with_errors(self):
        """Test configuration validation with errors."""
        # This should not raise during creation if we allow invalid configs
        # but validation should catch issues
        config = TestConfiguration(
            models=["gpt-4", "claude-3"],
            max_tokens=1000,
            temperature=0.7,
            timeout_seconds=-1,  # Invalid timeout
        )
        result = config.validate()
        assert result.is_valid is False
        assert len(result.errors) > 0


class TestTestSample:
    """Tests for TestSample entity."""

    def test_test_sample_creation(self):
        """Test creating test sample."""
        sample = TestSample(
            prompt="Test prompt",
            difficulty=DifficultyLevel.MEDIUM,
            expected_output="Expected response",
            tags=["tag1", "tag2"],
        )
        assert sample.prompt == "Test prompt"
        assert sample.difficulty == DifficultyLevel.MEDIUM
        assert sample.expected_output == "Expected response"
        assert sample.tags == ["tag1", "tag2"]
        assert sample.is_evaluated is False
        assert sample.evaluation_results == {}

    def test_test_sample_add_evaluation_result(self):
        """Test adding evaluation result to sample."""
        sample = TestSample(prompt="Test prompt", difficulty=DifficultyLevel.EASY)
        sample.add_evaluation_result("gpt-4", {"score": 0.85, "response": "AI response"})
        assert sample.evaluation_results["gpt-4"]["score"] == 0.85
        assert sample.evaluation_results["gpt-4"]["response"] == "AI response"
        assert sample.is_evaluated is True

    def test_test_sample_immutability_after_evaluation(self):
        """Test that sample becomes immutable after evaluation starts."""
        sample = TestSample(prompt="Test prompt", difficulty=DifficultyLevel.EASY)
        sample.add_evaluation_result("gpt-4", {"score": 0.85})

        # Should not be able to modify core properties after evaluation
        with pytest.raises(BusinessRuleViolation):
            sample.prompt = "Modified prompt"

    def test_test_sample_get_average_score(self):
        """Test calculating average score across evaluations."""
        sample = TestSample(prompt="Test prompt", difficulty=DifficultyLevel.MEDIUM)
        sample.add_evaluation_result("gpt-4", {"score": 0.8})
        sample.add_evaluation_result("claude-3", {"score": 0.9})

        assert sample.get_average_score() == 0.85

    def test_test_sample_get_average_score_no_evaluations(self):
        """Test average score with no evaluations."""
        sample = TestSample(prompt="Test prompt", difficulty=DifficultyLevel.EASY)
        assert sample.get_average_score() == 0.0

    def test_test_sample_weighted_score(self):
        """Test weighted score calculation based on difficulty."""
        sample = TestSample(prompt="Test prompt", difficulty=DifficultyLevel.HARD)
        sample.add_evaluation_result("gpt-4", {"score": 0.8})

        weighted_score = sample.get_weighted_score()
        expected_score = 0.8 * DifficultyLevel.HARD.score_factor()
        assert weighted_score == expected_score


class TestTest:
    """Tests for Test aggregate root."""

    def test_test_creation_with_factory_method(self):
        """Test creating test using factory method."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        assert test.name == "Test Name"
        assert test.configuration == config
        assert test.status == TestStatus.DRAFT
        assert test.samples == []
        assert isinstance(test.id, UUID)
        assert test.created_at is not None
        assert test.completed_at is None

        # Check domain event was published
        events = test.get_domain_events()
        assert len(events) == 1
        assert isinstance(events[0], TestCreated)
        assert events[0].test_id == test.id
        assert events[0].test_name == "Test Name"

    def test_test_creation_requires_valid_configuration(self):
        """Test that test creation requires valid configuration."""
        invalid_config = TestConfiguration(
            models=["gpt-4"], max_tokens=1000, temperature=0.7  # Only one model - invalid
        )
        with pytest.raises(BusinessRuleViolation):
            Test.create("Test Name", invalid_config)

    def test_test_add_sample_in_draft_state(self):
        """Test adding sample to draft test."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)
        sample = TestSample(prompt="Test prompt", difficulty=DifficultyLevel.EASY)

        test.add_sample(sample)
        assert len(test.samples) == 1
        assert test.samples[0] == sample

    def test_test_add_sample_fails_when_not_draft(self):
        """Test that adding sample fails when test is not in draft state."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)
        sample = TestSample(prompt="Test prompt", difficulty=DifficultyLevel.EASY)
        test.add_sample(sample)

        # Add minimum samples required for configuration
        for i in range(9):
            extra_sample = TestSample(prompt=f"Extra prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(extra_sample)

        # Transition to configured state
        test.configure()

        # Should not be able to add more samples
        another_sample = TestSample(prompt="Another prompt", difficulty=DifficultyLevel.MEDIUM)
        with pytest.raises(BusinessRuleViolation):
            test.add_sample(another_sample)

    def test_test_configure_transition(self):
        """Test configuring test."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)
        sample = TestSample(prompt="Test prompt", difficulty=DifficultyLevel.EASY)
        test.add_sample(sample)

        # Add minimum samples required for configuration
        for i in range(9):
            extra_sample = TestSample(prompt=f"Extra prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(extra_sample)

        test.configure()
        assert test.status == TestStatus.CONFIGURED

    def test_test_configure_fails_without_samples(self):
        """Test that configure fails without samples."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        with pytest.raises(BusinessRuleViolation):
            test.configure()

    def test_test_configure_enforces_sample_size_limits(self):
        """Test configure enforces sample size limits."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        # Add too few samples
        for i in range(5):  # Less than minimum of 10
            sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(sample)

        with pytest.raises(
            BusinessRuleViolation, match="Sample size must be between 10 and 10,000"
        ):
            test.configure()

    def test_test_start_from_configured_state(self):
        """Test starting test from configured state."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        # Add minimum samples
        for i in range(10):
            sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(sample)

        test.configure()
        test.start()

        assert test.status == TestStatus.RUNNING

        # Check domain event was published
        events = test.get_domain_events()
        started_events = [e for e in events if isinstance(e, TestStarted)]
        assert len(started_events) == 1
        assert started_events[0].test_id == test.id

    def test_test_start_fails_from_wrong_state(self):
        """Test that start fails from wrong state."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        with pytest.raises(InvalidStateTransition):
            test.start()  # Cannot start from DRAFT

    def test_test_complete_from_running_state(self):
        """Test completing test from running state."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        # Add samples and start test
        for i in range(10):
            sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(sample)

        test.configure()
        test.start()
        test.complete()

        assert test.status == TestStatus.COMPLETED
        assert test.completed_at is not None

        # Check domain event was published
        events = test.get_domain_events()
        completed_events = [e for e in events if isinstance(e, TestCompleted)]
        assert len(completed_events) == 1
        assert completed_events[0].test_id == test.id

    def test_test_complete_fails_from_wrong_state(self):
        """Test that complete fails from wrong state."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        with pytest.raises(InvalidStateTransition):
            test.complete()  # Cannot complete from DRAFT

    def test_test_cancel_from_any_non_terminal_state(self):
        """Test canceling test from non-terminal states."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        # Can cancel from DRAFT
        test.cancel()
        assert test.status == TestStatus.CANCELLED

        # Reset and test from CONFIGURED
        test = Test.create("Test Name 2", config)
        for i in range(10):
            sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(sample)
        test.configure()
        test.cancel()
        assert test.status == TestStatus.CANCELLED

    def test_test_fail_from_running_state(self):
        """Test failing test from running state."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        # Add samples and start test
        for i in range(10):
            sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(sample)

        test.configure()
        test.start()

        error_message = "Test execution failed"
        test.fail(error_message)

        assert test.status == TestStatus.FAILED
        assert test.failure_reason == error_message
        assert test.completed_at is not None

    def test_test_calculate_progress_no_samples(self):
        """Test progress calculation with no samples."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)
        assert test.calculate_progress() == 0.0

    def test_test_calculate_progress_partial_evaluation(self):
        """Test progress calculation with partial evaluation."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        # Add 10 samples
        for i in range(10):
            sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(sample)

        # Evaluate 3 samples
        for i in range(3):
            test.samples[i].add_evaluation_result("gpt-4", {"score": 0.8})

        assert test.calculate_progress() == 0.3  # 3/10

    def test_test_calculate_overall_score(self):
        """Test overall score calculation."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        # Add samples with different difficulties
        easy_sample = TestSample(prompt="Easy prompt", difficulty=DifficultyLevel.EASY)
        hard_sample = TestSample(prompt="Hard prompt", difficulty=DifficultyLevel.HARD)

        test.add_sample(easy_sample)
        test.add_sample(hard_sample)

        # Add evaluations
        easy_sample.add_evaluation_result("gpt-4", {"score": 0.8})
        hard_sample.add_evaluation_result("gpt-4", {"score": 0.7})

        # Score should be weighted by difficulty
        overall_score = test.calculate_overall_score()
        assert overall_score > 0

        # Hard sample should contribute more to the score
        easy_weighted = 0.8 * DifficultyLevel.EASY.score_factor()
        hard_weighted = 0.7 * DifficultyLevel.HARD.score_factor()
        expected_score = (easy_weighted + hard_weighted) / 2
        assert overall_score == expected_score

    def test_test_domain_events_management(self):
        """Test domain events are properly managed."""
        config = TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)
        test = Test.create("Test Name", config)

        # Should have creation event
        events = test.get_domain_events()
        assert len(events) == 1
        assert isinstance(events[0], TestCreated)

        # Clear events
        test.clear_domain_events()
        assert len(test.get_domain_events()) == 0

        # Add sample, configure, and start to generate more events
        for i in range(10):
            sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(sample)

        test.configure()
        test.start()

        events = test.get_domain_events()
        assert len(events) == 1  # Only TestStarted (creation was cleared)
        assert isinstance(events[0], TestStarted)
