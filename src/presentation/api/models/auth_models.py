"""Authentication-related API models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Login request model."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes in seconds


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str


class UserProfile(BaseModel):
    """User profile model."""

    username: str
    email: EmailStr
    role: str
    is_active: bool
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


class CreateUserRequest(BaseModel):
    """Create user request model."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: str = Field(default="user", pattern="^(admin|user|viewer)$")


class UpdateUserRequest(BaseModel):
    """Update user request model."""

    email: Optional[EmailStr] = None
    role: Optional[str] = Field(None, pattern="^(admin|user|viewer)$")
    is_active: Optional[bool] = None


class ChangePasswordRequest(BaseModel):
    """Change password request model."""

    current_password: str
    new_password: str = Field(..., min_length=6)
