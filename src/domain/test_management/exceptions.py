"""Domain exceptions for Test Management."""

from typing import Any, Dict, Optional


class TestManagementDomainException(Exception):
    """Base exception for Test Management domain."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class BusinessRuleViolation(TestManagementDomainException):
    """Exception raised when business rules are violated."""

    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.rule_name = rule_name


class InvalidStateTransition(TestManagementDomainException):
    """Exception raised when invalid state transition is attempted."""

    def __init__(
        self, message: str, from_state: Optional[str] = None, to_state: Optional[str] = None
    ):
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state


class ValidationError(TestManagementDomainException):
    """Exception raised when validation fails."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.field_name = field_name


class TestNotFound(TestManagementDomainException):
    """Exception raised when test is not found."""

    def __init__(self, message: str, test_id: Optional[str] = None):
        super().__init__(message)
        self.test_id = test_id


class InvalidConfiguration(TestManagementDomainException):
    """Exception raised when test configuration is invalid."""

    pass


class SampleEvaluationError(TestManagementDomainException):
    """Exception raised when sample evaluation fails."""

    pass
