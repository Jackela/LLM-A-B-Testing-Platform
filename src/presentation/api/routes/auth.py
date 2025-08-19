"""Authentication routes."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer

from ..auth.dependencies import get_current_active_user, require_admin
from ..auth.jwt_handler import (
    UserInDB,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_password_hash,
    refresh_access_token,
    verify_password,
)
from ..models.auth_models import (
    ChangePasswordRequest,
    CreateUserRequest,
    LoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UpdateUserRequest,
    UserProfile,
)

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate user and return JWT tokens."""
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user account")

    # Create tokens
    token_data = {
        "sub": user.username,
        "user_id": user.username,  # Using username as ID for simplicity
        "role": user.role,
    }

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    logger.info(f"User {user.username} logged in successfully")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800,  # 30 minutes
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token."""
    new_access_token = refresh_access_token(request.refresh_token)
    if not new_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=request.refresh_token,  # Keep same refresh token
        token_type="bearer",
        expires_in=1800,
    )


@router.get("/profile", response_model=UserProfile)
async def get_profile(current_user: UserInDB = Depends(get_current_active_user)):
    """Get current user profile."""
    return UserProfile(
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=datetime.utcnow(),  # Mock data
        last_login=datetime.utcnow(),  # Mock data
    )


@router.post("/users", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def create_user(request: CreateUserRequest, current_user: UserInDB = Depends(require_admin)):
    """Create new user (admin only)."""
    # In production, this would interact with a real database
    from ..auth.jwt_handler import FAKE_USERS_DB

    if request.username in FAKE_USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists"
        )

    # Create new user
    new_user = UserInDB(
        username=request.username,
        email=request.email,
        hashed_password=get_password_hash(request.password),
        role=request.role,
        is_active=True,
    )

    FAKE_USERS_DB[request.username] = new_user

    logger.info(f"User {request.username} created by {current_user.username}")

    return UserProfile(
        username=new_user.username,
        email=new_user.email,
        role=new_user.role,
        is_active=new_user.is_active,
        created_at=datetime.utcnow(),
    )


@router.put("/users/{username}", response_model=UserProfile)
async def update_user(
    username: str, request: UpdateUserRequest, current_user: UserInDB = Depends(require_admin)
):
    """Update user (admin only)."""
    from ..auth.jwt_handler import FAKE_USERS_DB

    user = FAKE_USERS_DB.get(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Update user fields
    if request.email:
        user.email = request.email
    if request.role:
        user.role = request.role
    if request.is_active is not None:
        user.is_active = request.is_active

    logger.info(f"User {username} updated by {current_user.username}")

    return UserProfile(
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=datetime.utcnow(),
    )


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest, current_user: UserInDB = Depends(get_current_active_user)
):
    """Change current user password."""
    # Verify current password
    if not verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect"
        )

    # Update password
    current_user.hashed_password = get_password_hash(request.new_password)

    logger.info(f"Password changed for user {current_user.username}")

    return {"message": "Password changed successfully"}


@router.post("/logout")
async def logout(current_user: UserInDB = Depends(get_current_active_user)):
    """Logout user (invalidate token - in production, maintain a blacklist)."""
    logger.info(f"User {current_user.username} logged out")
    return {"message": "Logged out successfully"}
