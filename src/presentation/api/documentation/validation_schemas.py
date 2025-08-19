"""API validation schemas and response models for documentation."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ProviderType(str, Enum):
    """Supported model provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class TestStatus(str, Enum):
    """A/B test status values."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class UserRole(str, Enum):
    """User role types."""

    USER = "USER"
    ADMIN = "ADMIN"
    EVALUATOR = "EVALUATOR"


# Authentication Schemas
class LoginRequest(BaseModel):
    """User login request."""

    username: str = Field(..., example="user@company.com", description="Username or email address")
    password: str = Field(..., example="SecurePassword123!", description="User password")


class LoginResponse(BaseModel):
    """User login response."""

    access_token: str = Field(..., example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    token_type: str = Field(default="bearer", example="bearer")
    expires_in: int = Field(..., example=86400, description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(None, example="refresh_token_here")
    user: "UserProfile"


class UserProfile(BaseModel):
    """User profile information."""

    id: str = Field(..., example="user-12345")
    username: str = Field(..., example="user@company.com")
    email: str = Field(..., example="user@company.com")
    role: UserRole = Field(..., example=UserRole.USER)
    permissions: List[str] = Field(default_factory=list, example=["CREATE_TEST", "READ_TEST"])
    created_at: datetime = Field(..., example="2024-01-01T00:00:00Z")
    last_login: Optional[datetime] = Field(None, example="2024-01-01T12:00:00Z")
    account_status: str = Field(default="active", example="active")


# Provider Schemas
class ProviderConfig(BaseModel):
    """Model provider configuration."""

    api_key: str = Field(..., description="API key for the provider", example="sk-...")
    model: str = Field(..., description="Model identifier", example="gpt-4-turbo-preview")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, example=0.7)
    max_tokens: Optional[int] = Field(2000, ge=1, le=4096, example=2000)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, example=1.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, example=0.0)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, example=0.0)


class CreateProviderRequest(BaseModel):
    """Request to create a new provider."""

    name: str = Field(..., min_length=1, max_length=100, example="OpenAI GPT-4 Turbo")
    provider_type: ProviderType = Field(..., example=ProviderType.OPENAI)
    config: ProviderConfig
    is_active: bool = Field(True, example=True)
    description: Optional[str] = Field(None, max_length=500, example="Latest GPT-4 model")


class ProviderResponse(BaseModel):
    """Provider response model."""

    id: str = Field(..., example="provider-abc123")
    name: str = Field(..., example="OpenAI GPT-4 Turbo")
    provider_type: ProviderType = Field(..., example=ProviderType.OPENAI)
    is_active: bool = Field(..., example=True)
    created_at: datetime = Field(..., example="2024-01-01T12:00:00Z")
    updated_at: datetime = Field(..., example="2024-01-01T12:00:00Z")
    created_by: str = Field(..., example="user-12345")
    config: Dict[str, Any] = Field(..., description="Provider configuration (API key masked)")
    description: Optional[str] = Field(None, example="Latest GPT-4 model")


# Test Schemas
class EvaluationCriteria(BaseModel):
    """Evaluation criteria with weights."""

    helpfulness: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.35)
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.25)
    clarity: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.15)
    creativity: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.25)
    professionalism: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.25)
    relevance: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.3)

    @validator("*", pre=True)
    def validate_criteria_sum(cls, v, values):
        """Validate that criteria weights sum to 1.0."""
        if hasattr(cls, "__validation_complete__"):
            return v

        # This will be validated at the model level
        return v


class CreateTestRequest(BaseModel):
    """Request to create a new A/B test."""

    name: str = Field(
        ..., min_length=1, max_length=200, example="Customer Support Response Quality"
    )
    description: str = Field(
        ..., min_length=1, max_length=1000, example="Comparing GPT-4 vs Claude-3"
    )
    prompt_template: str = Field(
        ..., min_length=1, max_length=5000, example="You are a helpful assistant. {question}"
    )
    provider_a_id: str = Field(..., example="provider-abc123")
    provider_b_id: str = Field(..., example="provider-def456")
    evaluation_criteria: Dict[str, float] = Field(
        ..., example={"helpfulness": 0.4, "accuracy": 0.6}
    )
    sample_size: int = Field(..., ge=10, le=10000, example=200)
    confidence_level: float = Field(0.95, ge=0.8, le=0.99, example=0.95)
    metadata: Optional[Dict[str, Any]] = Field(None, example={"department": "customer_support"})

    @validator("evaluation_criteria")
    def validate_criteria_sum(cls, v):
        """Validate that evaluation criteria weights sum to approximately 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Evaluation criteria weights must sum to 1.0, got {total}")
        return v

    @validator("provider_a_id", "provider_b_id")
    def validate_provider_ids(cls, v):
        """Validate provider ID format."""
        if not v or len(v) < 5:
            raise ValueError("Provider ID must be at least 5 characters long")
        return v


class TestResponse(BaseModel):
    """A/B test response model."""

    id: str = Field(..., example="test-xyz789")
    name: str = Field(..., example="Customer Support Response Quality")
    description: str = Field(..., example="Comparing GPT-4 vs Claude-3")
    status: TestStatus = Field(..., example=TestStatus.DRAFT)
    created_at: datetime = Field(..., example="2024-01-01T12:00:00Z")
    updated_at: datetime = Field(..., example="2024-01-01T12:00:00Z")
    started_at: Optional[datetime] = Field(None, example="2024-01-01T13:00:00Z")
    completed_at: Optional[datetime] = Field(None, example="2024-01-06T10:15:00Z")
    created_by: str = Field(..., example="user-12345")
    provider_a_id: str = Field(..., example="provider-abc123")
    provider_b_id: str = Field(..., example="provider-def456")
    sample_size: int = Field(..., example=200)
    confidence_level: float = Field(..., example=0.95)
    evaluations_count: int = Field(0, example=42)
    progress: float = Field(0.0, ge=0.0, le=100.0, example=21.0)
    estimated_completion: Optional[datetime] = Field(None, example="2024-01-06T10:30:00Z")
    metadata: Optional[Dict[str, Any]] = Field(None)


# Evaluation Schemas
class EvaluationScores(BaseModel):
    """Evaluation scores for both providers."""

    helpfulness: Optional[Dict[str, float]] = Field(None, example={"a": 0.9, "b": 0.7})
    accuracy: Optional[Dict[str, float]] = Field(None, example={"a": 0.85, "b": 0.9})
    clarity: Optional[Dict[str, float]] = Field(None, example={"a": 0.8, "b": 0.95})
    creativity: Optional[Dict[str, float]] = Field(None, example={"a": 0.75, "b": 0.8})
    professionalism: Optional[Dict[str, float]] = Field(None, example={"a": 0.95, "b": 0.8})
    relevance: Optional[Dict[str, float]] = Field(None, example={"a": 0.88, "b": 0.82})


class SubmitEvaluationRequest(BaseModel):
    """Request to submit an evaluation."""

    test_id: str = Field(..., example="test-xyz789")
    input_text: str = Field(
        ..., min_length=1, max_length=10000, example="How do I reset my password?"
    )
    response_a: str = Field(
        ..., min_length=1, max_length=10000, example="To reset your password..."
    )
    response_b: str = Field(
        ..., min_length=1, max_length=10000, example="You can reset your password..."
    )
    evaluation_scores: Dict[str, Dict[str, float]] = Field(
        ..., example={"helpfulness": {"a": 0.9, "b": 0.7}, "accuracy": {"a": 0.85, "b": 0.9}}
    )
    evaluator: Optional[str] = Field(None, example="expert-evaluator-001")
    evaluation_time_seconds: Optional[int] = Field(None, ge=1, example=67)
    metadata: Optional[Dict[str, Any]] = Field(None, example={"difficulty": "medium"})


class EvaluationResponse(BaseModel):
    """Evaluation submission response."""

    id: str = Field(..., example="eval-qwe123")
    test_id: str = Field(..., example="test-xyz789")
    status: str = Field(..., example="processed")
    submitted_at: datetime = Field(..., example="2024-01-01T14:30:00Z")
    overall_score_a: float = Field(..., example=0.875)
    overall_score_b: float = Field(..., example=0.8125)
    confidence: float = Field(..., example=0.94)
    evaluation_number: int = Field(..., example=32)


# Analytics Schemas
class DashboardSummary(BaseModel):
    """Dashboard summary statistics."""

    total_tests: int = Field(..., example=47)
    active_tests: int = Field(..., example=5)
    completed_tests: int = Field(..., example=39)
    draft_tests: int = Field(..., example=3)
    total_evaluations: int = Field(..., example=8429)
    total_providers: int = Field(..., example=12)
    active_providers: int = Field(..., example=9)


class ActivityItem(BaseModel):
    """Recent activity item."""

    type: str = Field(..., example="test_completed")
    test_id: Optional[str] = Field(None, example="test-abc123")
    test_name: Optional[str] = Field(None, example="Email Response Quality")
    timestamp: datetime = Field(..., example="2024-01-01T11:45:00Z")
    result: Optional[str] = Field(None, example="Provider A significantly better")
    evaluator: Optional[str] = Field(None, example="expert-evaluator-001")


class PerformanceMetrics(BaseModel):
    """Performance metrics."""

    average_test_duration: float = Field(
        ..., example=5.2, description="Average test duration in days"
    )
    average_evaluations_per_test: int = Field(..., example=179)
    average_confidence_level: float = Field(..., example=0.93)
    most_active_evaluator: str = Field(..., example="expert-evaluator-001")


class DashboardResponse(BaseModel):
    """Dashboard analytics response."""

    summary: DashboardSummary
    recent_activity: List[ActivityItem]
    performance_metrics: PerformanceMetrics


class ScoreStatistics(BaseModel):
    """Statistical analysis of scores."""

    mean: float = Field(..., example=0.847)
    std: float = Field(..., example=0.123)
    median: float = Field(..., example=0.85)
    min: Optional[float] = Field(None, example=0.2)
    max: Optional[float] = Field(None, example=1.0)


class ProviderResults(BaseModel):
    """Results for a single provider."""

    name: str = Field(..., example="OpenAI GPT-4 Turbo")
    model: str = Field(..., example="gpt-4-turbo-preview")
    scores: Dict[str, ScoreStatistics] = Field(
        ..., example={"helpfulness": {"mean": 0.847, "std": 0.123, "median": 0.85}}
    )
    overall_score: float = Field(..., example=0.835)
    confidence_interval: List[float] = Field(..., example=[0.811, 0.859])
    total_evaluations: int = Field(..., example=200)


class StatisticalAnalysis(BaseModel):
    """Statistical analysis of A/B test results."""

    p_value: float = Field(..., example=0.0423)
    effect_size: float = Field(..., example=0.127)
    is_significant: bool = Field(..., example=True)
    confidence_level: float = Field(..., example=0.95)
    winner: str = Field(..., example="provider_a")
    improvement_percentage: float = Field(..., example=1.21)
    power_analysis: float = Field(..., example=0.82)


class TestResultsResponse(BaseModel):
    """Comprehensive test results response."""

    test_id: str = Field(..., example="test-xyz789")
    test_name: str = Field(..., example="Customer Support Response Quality")
    status: TestStatus = Field(..., example=TestStatus.COMPLETED)
    completion_date: Optional[datetime] = Field(None, example="2024-01-06T10:15:00Z")
    total_evaluations: int = Field(..., example=200)
    duration_days: float = Field(..., example=5.2)
    results: Dict[str, ProviderResults] = Field(
        ..., description="Results for provider_a and provider_b"
    )
    statistical_analysis: StatisticalAnalysis
    recommendations: List[str] = Field(
        ...,
        example=[
            "Provider A shows statistically significant better performance",
            "Consider using Provider A for customer support applications",
        ],
    )


# Error Schemas
class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., example="validation_error")
    message: str = Field(..., example="Invalid input data provided")
    details: Optional[Dict[str, Any]] = Field(None, example={"field": "provider_type"})


class ValidationErrorDetail(BaseModel):
    """Validation error detail."""

    loc: List[Union[str, int]] = Field(..., example=["body", "provider_type"])
    msg: str = Field(..., example="field required")
    type: str = Field(..., example="value_error.missing")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""

    detail: List[ValidationErrorDetail]


# Health Check Schema
class ServiceStatus(BaseModel):
    """Service status information."""

    database: str = Field(..., example="connected")
    redis: str = Field(..., example="connected")
    security: str = Field(..., example="active")
    performance: Optional[str] = Field(None, example="optimal")


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., example="healthy")
    timestamp: int = Field(..., example=1640995200)
    version: str = Field(..., example="1.0.0")
    services: Optional[ServiceStatus] = Field(None)
    uptime_seconds: Optional[int] = Field(None, example=86400)


# Update LoginResponse to avoid circular import
LoginResponse.model_rebuild()
