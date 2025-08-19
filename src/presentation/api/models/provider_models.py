"""Model provider API models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ProviderType(str, Enum):
    """Provider type enumeration."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ProviderStatus(str, Enum):
    """Provider status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ModelCategory(str, Enum):
    """Model category enumeration."""

    GENERAL = "general"
    CHAT = "chat"
    COMPLETION = "completion"
    CODE = "code"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


class RateLimits(BaseModel):
    """Rate limits model."""

    requests_per_minute: int = Field(..., ge=1)
    tokens_per_minute: int = Field(..., ge=100)
    concurrent_requests: int = Field(..., ge=1)


class ModelInfo(BaseModel):
    """Model information model."""

    id: str
    name: str
    category: ModelCategory
    description: Optional[str] = None
    max_tokens: int = Field(..., ge=1)
    supports_streaming: bool = False
    cost_per_1k_tokens: float = Field(..., ge=0)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """Provider response model."""

    id: str
    name: str
    provider_type: ProviderType
    status: ProviderStatus
    api_endpoint: str
    rate_limits: RateLimits
    models: List[ModelInfo]
    created_at: datetime
    updated_at: datetime
    last_health_check: Optional[datetime] = None


class ProviderListResponse(BaseModel):
    """Provider list response model."""

    providers: List[ProviderResponse]
    total: int


class TestConnectionRequest(BaseModel):
    """Test connection request model."""

    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    timeout_seconds: int = Field(default=30, ge=5, le=300)


class ConnectionTestResponse(BaseModel):
    """Connection test response model."""

    success: bool
    response_time_ms: int
    available_models: List[str]
    error: Optional[str] = None
    timestamp: datetime


class ProviderHealthResponse(BaseModel):
    """Provider health response model."""

    provider_id: str
    status: ProviderStatus
    response_time_ms: int
    available_models: int
    rate_limit_remaining: Dict[str, int]
    last_error: Optional[str] = None
    timestamp: datetime


class ModelUsageStats(BaseModel):
    """Model usage statistics."""

    model_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    total_tokens_used: int
    total_cost: float
    last_used: Optional[datetime] = None


class ProviderUsageResponse(BaseModel):
    """Provider usage response model."""

    provider_id: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    total_cost: float
    models: List[ModelUsageStats]
