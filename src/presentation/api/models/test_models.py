"""Test management API models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class TestStatus(str, Enum):
    """Test status enumeration."""

    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DifficultyLevel(str, Enum):
    """Difficulty level enumeration."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ModelConfiguration(BaseModel):
    """Model configuration for tests."""

    model_id: str
    provider: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @validator("parameters")
    def validate_parameters(cls, v):
        """Validate model parameters."""
        allowed_params = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        }
        if not all(key in allowed_params for key in v.keys()):
            raise ValueError(f"Invalid parameters. Allowed: {allowed_params}")
        return v


class CreateTestRequest(BaseModel):
    """Create test request model."""

    name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    model_a: ModelConfiguration
    model_b: ModelConfiguration
    evaluation_template_id: str
    sample_size: int = Field(..., ge=10, le=10000)
    difficulty_level: DifficultyLevel = DifficultyLevel.MEDIUM
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateTestRequest(BaseModel):
    """Update test request model."""

    name: Optional[str] = Field(None, min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    model_a: Optional[ModelConfiguration] = None
    model_b: Optional[ModelConfiguration] = None
    evaluation_template_id: Optional[str] = None
    sample_size: Optional[int] = Field(None, ge=10, le=10000)
    difficulty_level: Optional[DifficultyLevel] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class TestSample(BaseModel):
    """Test sample model."""

    id: str
    prompt: str
    expected_response: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestResponse(BaseModel):
    """Test response model."""

    id: str
    name: str
    description: Optional[str]
    status: TestStatus
    model_a: ModelConfiguration
    model_b: ModelConfiguration
    evaluation_template_id: str
    sample_size: int
    difficulty_level: DifficultyLevel
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: str
    progress: Optional[Dict[str, Any]] = None


class TestListResponse(BaseModel):
    """Test list response model."""

    tests: List[TestResponse]
    total: int
    page: int
    page_size: int


class StartTestRequest(BaseModel):
    """Start test request model."""

    concurrent_workers: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)


class TestProgress(BaseModel):
    """Test progress model."""

    total_samples: int
    completed_samples: int
    failed_samples: int
    success_rate: float
    estimated_completion: Optional[datetime]
    current_status: TestStatus


class AddSamplesRequest(BaseModel):
    """Add samples request model."""

    samples: List[TestSample] = Field(..., min_items=1, max_items=100)


class TestFilters(BaseModel):
    """Test filtering options."""

    status: Optional[List[TestStatus]] = None
    difficulty_level: Optional[List[DifficultyLevel]] = None
    tags: Optional[List[str]] = None
    created_by: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
