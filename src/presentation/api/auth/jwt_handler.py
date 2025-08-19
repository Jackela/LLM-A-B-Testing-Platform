"""JWT token handling and authentication - Enhanced version with security integration."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from ....infrastructure.security.auth import TokenData as AuthTokenData
from ....infrastructure.security.auth import TokenType, get_auth_system
from ....infrastructure.security.secrets_manager import get_env_secrets

logger = logging.getLogger(__name__)

# Enhanced configuration using secrets manager
env_secrets = get_env_secrets()
auth_system = get_auth_system()

# Legacy compatibility - gradually migrate to enhanced auth system
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenData(BaseModel):
    """Legacy token data model for backward compatibility."""

    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None
    permissions: Optional[list] = None
    session_id: Optional[str] = None


class UserInDB(BaseModel):
    """Legacy user model for backward compatibility."""

    username: str
    email: str
    hashed_password: str
    role: str = "user"
    is_active: bool = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return auth_system.verify_password(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return auth_system.get_password_hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token - legacy compatibility wrapper."""
    # Try to find user in enhanced auth system
    username = data.get("sub")
    if username:
        user = auth_system.users.get(username)
        if user:
            # Use enhanced system
            tokens = auth_system.create_tokens(user)
            return tokens["access_token"]

    # Fallback to legacy implementation
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=auth_system.access_token_expire_minutes)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, auth_system.secret_key, algorithm=auth_system.algorithm)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token - legacy compatibility wrapper."""
    username = data.get("sub")
    if username:
        user = auth_system.users.get(username)
        if user:
            tokens = auth_system.create_tokens(user)
            return tokens["refresh_token"]

    # Fallback to legacy implementation
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=auth_system.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, auth_system.secret_key, algorithm=auth_system.algorithm)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
    """Verify and decode JWT token - enhanced version."""
    try:
        # Map legacy string to enum
        if token_type == "access":
            token_enum = TokenType.ACCESS
        elif token_type == "refresh":
            token_enum = TokenType.REFRESH
        else:
            token_enum = TokenType.ACCESS

        # Use enhanced auth system
        auth_token_data = auth_system.verify_token(token, token_enum)
        if not auth_token_data:
            return None

        # Convert to legacy format for backward compatibility
        return TokenData(
            username=auth_token_data.username,
            user_id=auth_token_data.user_id,
            role=auth_token_data.role.value,
            permissions=auth_token_data.permissions,
            session_id=auth_token_data.session_id,
        )

    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        return None


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """Create new access token from refresh token - enhanced version."""
    tokens = auth_system.refresh_access_token(refresh_token)
    if tokens:
        return tokens["access_token"]
    return None


def authenticate_user(
    username: str, password: str, ip_address: str = None, mfa_code: str = None
) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials with enhanced security."""
    success, message, user = auth_system.authenticate_user(username, password, ip_address, mfa_code)

    if success and user:
        # Return user data in legacy format for backward compatibility
        return {
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "is_active": user.is_active,
            "user_id": user.id,
            "permissions": [p.value for p in user.permissions],
            "mfa_enabled": user.security.mfa.enabled,
        }

    return None


def get_user(username: str) -> Optional[Dict[str, Any]]:
    """Get user by username - enhanced version."""
    user = auth_system.users.get(username)
    if user:
        return {
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "is_active": user.is_active,
            "user_id": user.id,
            "permissions": [p.value for p in user.permissions],
        }
    return None


def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Verify API key and return user data."""
    user = auth_system.verify_api_key(api_key)
    if user:
        return {
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "is_active": user.is_active,
            "user_id": user.id,
            "permissions": [p.value for p in user.permissions],
        }
    return None


def create_user(username: str, email: str, password: str, role: str = "user") -> tuple[bool, str]:
    """Create new user with enhanced validation."""
    from ...infrastructure.security.auth import Role

    # Map string role to enum
    role_enum = Role.USER
    if role in ["admin", "super_admin", "manager", "analyst", "viewer", "api_user"]:
        role_enum = Role(role)

    return auth_system.create_user(username, email, password, role_enum)


def enable_mfa(username: str) -> tuple[bool, str, Optional[list]]:
    """Enable MFA for user."""
    success, uri, backup_codes = auth_system.enable_mfa(username)
    return success, uri, backup_codes


def create_api_key(username: str, name: str = None) -> tuple[bool, str]:
    """Create API key for user."""
    user = auth_system.users.get(username)
    if not user:
        return False, "User not found"

    return auth_system.create_api_key(user.id, name)


def revoke_api_key(api_key: str) -> bool:
    """Revoke API key."""
    return auth_system.revoke_api_key(api_key)


# Legacy compatibility - REMOVED hardcoded credentials for security
# Users should be created via the enhanced auth system or migration scripts
FAKE_USERS_DB = {
    # Removed hardcoded credentials - use auth_system.create_user() instead
    # or implement proper database-backed user management
}
