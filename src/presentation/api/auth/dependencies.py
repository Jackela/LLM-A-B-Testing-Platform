"""Authentication dependencies for FastAPI endpoints."""

import logging
from typing import List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt_handler import TokenData, UserInDB, get_user, verify_token

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserInDB:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token_data = verify_token(credentials.credentials)
        if token_data is None:
            raise credentials_exception

        user = get_user(username=token_data.username)
        if user is None:
            raise credentials_exception

        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")

        return user

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise credentials_exception


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


def require_roles(allowed_roles: List[str]):
    """Decorator to require specific roles."""

    def role_checker(current_user: UserInDB = Depends(get_current_active_user)) -> UserInDB:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of roles: {allowed_roles}. Current role: {current_user.role}",
            )
        return current_user

    return role_checker


# Role-based dependencies
require_admin = require_roles(["admin"])
require_user_or_admin = require_roles(["user", "admin"])
require_any_role = require_roles(["viewer", "user", "admin"])


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[UserInDB]:
    """Get current user if authenticated, None otherwise."""
    if not credentials:
        return None

    try:
        token_data = verify_token(credentials.credentials)
        if token_data is None:
            return None

        user = get_user(username=token_data.username)
        return user if user and user.is_active else None

    except Exception:
        return None
