"""Tests for Test Management domain value objects."""

from enum import Enum
from typing import List, Optional

import pytest

from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.test_status import TestStatus
from src.domain.test_management.value_objects.validation_result import ValidationResult


class TestTestStatus:
    """Tests for TestStatus value object."""

    def test_status_enum_values(self):
        """Test that all expected status values exist."""
        assert TestStatus.DRAFT
        assert TestStatus.CONFIGURED
        assert TestStatus.RUNNING
        assert TestStatus.COMPLETED
        assert TestStatus.FAILED
        assert TestStatus.CANCELLED

    def test_status_transitions_from_draft(self):
        """Test valid transitions from DRAFT status."""
        status = TestStatus.DRAFT
        assert status.can_transition_to(TestStatus.CONFIGURED) is True
        assert status.can_transition_to(TestStatus.CANCELLED) is True
        assert status.can_transition_to(TestStatus.RUNNING) is False
        assert status.can_transition_to(TestStatus.COMPLETED) is False
        assert status.can_transition_to(TestStatus.FAILED) is False

    def test_status_transitions_from_configured(self):
        """Test valid transitions from CONFIGURED status."""
        status = TestStatus.CONFIGURED
        assert status.can_transition_to(TestStatus.RUNNING) is True
        assert status.can_transition_to(TestStatus.CANCELLED) is True
        assert status.can_transition_to(TestStatus.DRAFT) is True
        assert status.can_transition_to(TestStatus.COMPLETED) is False
        assert status.can_transition_to(TestStatus.FAILED) is False

    def test_status_transitions_from_running(self):
        """Test valid transitions from RUNNING status."""
        status = TestStatus.RUNNING
        assert status.can_transition_to(TestStatus.COMPLETED) is True
        assert status.can_transition_to(TestStatus.FAILED) is True
        assert status.can_transition_to(TestStatus.CANCELLED) is True
        assert status.can_transition_to(TestStatus.DRAFT) is False
        assert status.can_transition_to(TestStatus.CONFIGURED) is False

    def test_terminal_status_no_transitions(self):
        """Test that terminal statuses cannot transition."""
        for terminal_status in [TestStatus.COMPLETED, TestStatus.FAILED, TestStatus.CANCELLED]:
            for other_status in TestStatus:
                if other_status != terminal_status:
                    assert terminal_status.can_transition_to(other_status) is False

    def test_status_is_terminal(self):
        """Test terminal status identification."""
        assert TestStatus.COMPLETED.is_terminal() is True
        assert TestStatus.FAILED.is_terminal() is True
        assert TestStatus.CANCELLED.is_terminal() is True
        assert TestStatus.DRAFT.is_terminal() is False
        assert TestStatus.CONFIGURED.is_terminal() is False
        assert TestStatus.RUNNING.is_terminal() is False

    def test_status_allows_modification(self):
        """Test which statuses allow test modification."""
        assert TestStatus.DRAFT.allows_modification() is True
        assert TestStatus.CONFIGURED.allows_modification() is False
        assert TestStatus.RUNNING.allows_modification() is False
        assert TestStatus.COMPLETED.allows_modification() is False
        assert TestStatus.FAILED.allows_modification() is False
        assert TestStatus.CANCELLED.allows_modification() is False


class TestDifficultyLevel:
    """Tests for DifficultyLevel value object."""

    def test_difficulty_enum_values(self):
        """Test that all expected difficulty values exist."""
        assert DifficultyLevel.EASY
        assert DifficultyLevel.MEDIUM
        assert DifficultyLevel.HARD

    def test_difficulty_ordering(self):
        """Test difficulty level ordering."""
        assert DifficultyLevel.EASY < DifficultyLevel.MEDIUM
        assert DifficultyLevel.MEDIUM < DifficultyLevel.HARD
        assert DifficultyLevel.EASY < DifficultyLevel.HARD

    def test_difficulty_weight(self):
        """Test difficulty weight calculation."""
        assert DifficultyLevel.EASY.weight() == 1.0
        assert DifficultyLevel.MEDIUM.weight() == 1.5
        assert DifficultyLevel.HARD.weight() == 2.0

    def test_difficulty_from_string(self):
        """Test creating difficulty from string."""
        assert DifficultyLevel.from_string("easy") == DifficultyLevel.EASY
        assert DifficultyLevel.from_string("MEDIUM") == DifficultyLevel.MEDIUM
        assert DifficultyLevel.from_string("Hard") == DifficultyLevel.HARD

    def test_difficulty_from_invalid_string(self):
        """Test error on invalid difficulty string."""
        with pytest.raises(ValueError):
            DifficultyLevel.from_string("invalid")

    def test_difficulty_score_factor(self):
        """Test score factor calculation for difficulty."""
        easy_factor = DifficultyLevel.EASY.score_factor()
        medium_factor = DifficultyLevel.MEDIUM.score_factor()
        hard_factor = DifficultyLevel.HARD.score_factor()

        assert easy_factor < medium_factor < hard_factor
        assert easy_factor == 0.8
        assert medium_factor == 1.0
        assert hard_factor == 1.2


class TestValidationResult:
    """Tests for ValidationResult value object."""

    def test_validation_result_creation_success(self):
        """Test creating successful validation result."""
        result = ValidationResult.success()
        assert result.is_valid is True
        assert result.errors == ()
        assert result.warnings == ()

    def test_validation_result_creation_with_errors(self):
        """Test creating validation result with errors."""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult.with_errors(errors)
        assert result.is_valid is False
        assert result.errors == tuple(errors)
        assert result.warnings == ()

    def test_validation_result_creation_with_warnings(self):
        """Test creating validation result with warnings."""
        warnings = ["Warning 1"]
        result = ValidationResult.with_warnings(warnings)
        assert result.is_valid is True
        assert result.errors == ()
        assert result.warnings == tuple(warnings)

    def test_validation_result_creation_with_errors_and_warnings(self):
        """Test creating validation result with both errors and warnings."""
        errors = ["Error 1"]
        warnings = ["Warning 1", "Warning 2"]
        result = ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        assert result.is_valid is False
        assert result.errors == tuple(errors)
        assert result.warnings == tuple(warnings)

    def test_validation_result_add_error(self):
        """Test adding error to validation result."""
        result = ValidationResult.success()
        new_result = result.add_error("New error")
        assert new_result.is_valid is False
        assert "New error" in new_result.errors
        # Original should remain unchanged
        assert result.is_valid is True

    def test_validation_result_add_warning(self):
        """Test adding warning to validation result."""
        result = ValidationResult.success()
        new_result = result.add_warning("New warning")
        assert new_result.is_valid is True
        assert "New warning" in new_result.warnings
        # Original should remain unchanged
        assert result.warnings == ()

    def test_validation_result_combine_success(self):
        """Test combining successful validation results."""
        result1 = ValidationResult.success()
        result2 = ValidationResult.success()
        combined = ValidationResult.combine([result1, result2])
        assert combined.is_valid is True
        assert combined.errors == ()
        assert combined.warnings == ()

    def test_validation_result_combine_with_errors(self):
        """Test combining validation results with errors."""
        result1 = ValidationResult.with_errors(["Error 1"])
        result2 = ValidationResult.with_errors(["Error 2"])
        combined = ValidationResult.combine([result1, result2])
        assert combined.is_valid is False
        assert "Error 1" in combined.errors
        assert "Error 2" in combined.errors

    def test_validation_result_combine_mixed(self):
        """Test combining mixed validation results."""
        result1 = ValidationResult.success()
        result2 = ValidationResult.with_warnings(["Warning 1"])
        result3 = ValidationResult.with_errors(["Error 1"])
        combined = ValidationResult.combine([result1, result2, result3])
        assert combined.is_valid is False
        assert "Error 1" in combined.errors
        assert "Warning 1" in combined.warnings

    def test_validation_result_has_errors(self):
        """Test error checking."""
        success_result = ValidationResult.success()
        error_result = ValidationResult.with_errors(["Error"])
        assert success_result.has_errors() is False
        assert error_result.has_errors() is True

    def test_validation_result_has_warnings(self):
        """Test warning checking."""
        success_result = ValidationResult.success()
        warning_result = ValidationResult.with_warnings(["Warning"])
        assert success_result.has_warnings() is False
        assert warning_result.has_warnings() is True

    def test_validation_result_immutability(self):
        """Test that validation result is immutable after creation."""
        result = ValidationResult.success()
        original_errors = result.errors
        original_warnings = result.warnings

        # Attempting to modify should not affect original
        try:
            result.errors.append("New error")
        except AttributeError:
            pass  # Expected if errors is tuple/frozen

        # Ensure original state is preserved
        assert len(original_errors) == 0
        assert len(original_warnings) == 0
