"""Enhanced authentication system with MFA and RBAC."""

import hashlib
import logging
import os
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import jwt
import pyotp
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from .secrets_manager import get_env_secrets

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """User roles with hierarchical permissions."""

    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    USER = "user"
    VIEWER = "viewer"
    API_USER = "api_user"


class Permission(str, Enum):
    """Fine-grained permissions."""

    # Test management
    CREATE_TEST = "create_test"
    READ_TEST = "read_test"
    UPDATE_TEST = "update_test"
    DELETE_TEST = "delete_test"
    START_TEST = "start_test"
    STOP_TEST = "stop_test"

    # Analytics
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_ANALYTICS = "export_analytics"
    ADVANCED_ANALYTICS = "advanced_analytics"

    # User management
    CREATE_USER = "create_user"
    READ_USER = "read_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    ASSIGN_ROLES = "assign_roles"

    # System
    SYSTEM_CONFIG = "system_config"
    VIEW_LOGS = "view_logs"
    SYSTEM_HEALTH = "system_health"

    # API access
    API_ACCESS = "api_access"
    BULK_OPERATIONS = "bulk_operations"


# Role-permission mappings
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.SUPER_ADMIN: set(Permission),  # All permissions
    Role.ADMIN: {
        Permission.CREATE_TEST,
        Permission.READ_TEST,
        Permission.UPDATE_TEST,
        Permission.DELETE_TEST,
        Permission.START_TEST,
        Permission.STOP_TEST,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_ANALYTICS,
        Permission.ADVANCED_ANALYTICS,
        Permission.CREATE_USER,
        Permission.READ_USER,
        Permission.UPDATE_USER,
        Permission.ASSIGN_ROLES,
        Permission.SYSTEM_CONFIG,
        Permission.VIEW_LOGS,
        Permission.SYSTEM_HEALTH,
        Permission.API_ACCESS,
        Permission.BULK_OPERATIONS,
    },
    Role.MANAGER: {
        Permission.CREATE_TEST,
        Permission.READ_TEST,
        Permission.UPDATE_TEST,
        Permission.START_TEST,
        Permission.STOP_TEST,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_ANALYTICS,
        Permission.ADVANCED_ANALYTICS,
        Permission.READ_USER,
        Permission.API_ACCESS,
    },
    Role.ANALYST: {
        Permission.READ_TEST,
        Permission.UPDATE_TEST,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_ANALYTICS,
        Permission.ADVANCED_ANALYTICS,
        Permission.API_ACCESS,
    },
    Role.USER: {
        Permission.CREATE_TEST,
        Permission.READ_TEST,
        Permission.UPDATE_TEST,
        Permission.VIEW_ANALYTICS,
        Permission.API_ACCESS,
    },
    Role.VIEWER: {Permission.READ_TEST, Permission.VIEW_ANALYTICS},
    Role.API_USER: {
        Permission.API_ACCESS,
        Permission.READ_TEST,
        Permission.CREATE_TEST,
        Permission.VIEW_ANALYTICS,
        Permission.BULK_OPERATIONS,
    },
}


@dataclass
class MFASettings:
    """Multi-factor authentication settings."""

    enabled: bool = False
    secret: Optional[str] = None
    backup_codes: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None


@dataclass
class SecuritySettings:
    """User security settings."""

    password_changed_at: datetime = field(default_factory=datetime.utcnow)
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    last_login: Optional[datetime] = None
    login_history: List[Tuple[datetime, str]] = field(default_factory=list)  # (timestamp, ip)
    require_password_change: bool = False
    mfa: MFASettings = field(default_factory=MFASettings)


@dataclass
class User:
    """Enhanced user model with security features."""

    id: str
    username: str
    email: str
    hashed_password: str
    role: Role
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    api_keys: List[str] = field(default_factory=list)
    additional_permissions: Set[Permission] = field(default_factory=set)

    @property
    def permissions(self) -> Set[Permission]:
        """Get effective permissions including role and additional."""
        base_permissions = ROLE_PERMISSIONS.get(self.role, set())
        return base_permissions | self.additional_permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def is_account_locked(self) -> bool:
        """Check if account is locked."""
        if not self.security.account_locked_until:
            return False
        return datetime.utcnow() < self.security.account_locked_until


class TokenType(str, Enum):
    """Token types."""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    PASSWORD_RESET = "password_reset"
    EMAIL_VERIFICATION = "email_verification"


@dataclass
class TokenData:
    """Token data with enhanced information."""

    user_id: str
    username: str
    role: Role
    token_type: TokenType
    permissions: List[str] = field(default_factory=list)
    api_key_id: Optional[str] = None
    session_id: Optional[str] = None


class EnhancedAuthSystem:
    """Enhanced authentication system with MFA and RBAC."""

    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.env_secrets = get_env_secrets()
        self.secret_key = self.env_secrets.get_jwt_secret()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        self.max_login_attempts = 5
        self.account_lockout_duration = timedelta(minutes=15)

        # In-memory user store (replace with database)
        self.users: Dict[str, User] = {}
        self.refresh_tokens: Dict[str, str] = {}  # token -> user_id
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id

        # Initialize with default admin user
        self._create_default_users()

    def _create_default_users(self) -> None:
        """Create default users for system initialization."""
        admin_user = User(
            id=str(uuid.uuid4()),
            username="admin",
            email="admin@example.com",
            hashed_password=self.get_password_hash("Admin123!"),
            role=Role.ADMIN,
            is_active=True,
            is_verified=True,
        )

        self.users[admin_user.username] = admin_user
        logger.info("Default admin user created")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return self.pwd_context.hash(password)

    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password meets security requirements."""
        issues = []

        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        if not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one special character")

        return len(issues) == 0, issues

    def create_user(
        self, username: str, email: str, password: str, role: Role = Role.USER
    ) -> Tuple[bool, str]:
        """Create new user with validation."""
        # Check if user exists
        if username in self.users:
            return False, "Username already exists"

        if any(u.email == email for u in self.users.values()):
            return False, "Email already registered"

        # Validate password
        is_strong, issues = self.validate_password_strength(password)
        if not is_strong:
            return False, "; ".join(issues)

        # Create user
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            hashed_password=self.get_password_hash(password),
            role=role,
        )

        self.users[username] = user
        logger.info(f"User created: {username} with role {role}")
        return True, "User created successfully"

    def authenticate_user(
        self, username: str, password: str, ip_address: str = None, mfa_code: str = None
    ) -> Tuple[bool, str, Optional[User]]:
        """Authenticate user with enhanced security checks."""
        user = self.users.get(username)
        if not user:
            return False, "Invalid credentials", None

        # Check if account is locked
        if user.is_account_locked():
            return False, "Account is temporarily locked due to failed login attempts", None

        # Check if account is active
        if not user.is_active:
            return False, "Account is deactivated", None

        # Verify password
        if not self.verify_password(password, user.hashed_password):
            # Increment failed attempts
            user.security.failed_login_attempts += 1

            if user.security.failed_login_attempts >= self.max_login_attempts:
                user.security.account_locked_until = (
                    datetime.utcnow() + self.account_lockout_duration
                )
                logger.warning(
                    f"Account locked for user {username} due to {self.max_login_attempts} failed attempts"
                )
                return (
                    False,
                    f"Account locked due to {self.max_login_attempts} failed login attempts",
                    None,
                )

            logger.warning(
                f"Failed login attempt {user.security.failed_login_attempts} for user {username}"
            )
            return False, "Invalid credentials", None

        # Check MFA if enabled
        if user.security.mfa.enabled:
            if not mfa_code:
                return False, "MFA code required", None

            if not self.verify_mfa_code(user, mfa_code):
                user.security.failed_login_attempts += 1
                return False, "Invalid MFA code", None

        # Successful login
        user.security.failed_login_attempts = 0
        user.security.account_locked_until = None
        user.security.last_login = datetime.utcnow()

        # Add to login history (keep last 10)
        if ip_address:
            user.security.login_history.append((datetime.utcnow(), ip_address))
            if len(user.security.login_history) > 10:
                user.security.login_history.pop(0)

        logger.info(f"Successful login for user {username} from {ip_address or 'unknown IP'}")
        return True, "Login successful", user

    def enable_mfa(self, username: str) -> Tuple[bool, str, Optional[str]]:
        """Enable MFA for user and return setup information."""
        user = self.users.get(username)
        if not user:
            return False, "User not found", None

        if user.security.mfa.enabled:
            return False, "MFA already enabled", None

        # Generate MFA secret
        secret = pyotp.random_base32()
        user.security.mfa.secret = secret
        user.security.mfa.enabled = True

        # Generate backup codes
        backup_codes = [secrets.token_hex(8) for _ in range(10)]
        user.security.mfa.backup_codes = backup_codes

        # Generate QR code URL
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user.email, issuer_name="LLM A/B Testing Platform"
        )

        logger.info(f"MFA enabled for user {username}")
        return True, totp_uri, backup_codes

    def verify_mfa_code(self, user: User, code: str) -> bool:
        """Verify MFA code or backup code."""
        if not user.security.mfa.enabled or not user.security.mfa.secret:
            return False

        # Check TOTP code
        totp = pyotp.TOTP(user.security.mfa.secret)
        if totp.verify(code, valid_window=1):
            user.security.mfa.last_used = datetime.utcnow()
            return True

        # Check backup codes
        if code in user.security.mfa.backup_codes:
            user.security.mfa.backup_codes.remove(code)
            logger.info(f"Backup code used for user {user.username}")
            return True

        return False

    def create_tokens(self, user: User, session_id: str = None) -> Dict[str, str]:
        """Create access and refresh tokens."""
        session_id = session_id or str(uuid.uuid4())

        # Access token
        access_token_data = {
            "sub": user.username,
            "user_id": user.id,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "session_id": session_id,
            "type": TokenType.ACCESS.value,
        }

        access_token = self._create_jwt_token(
            access_token_data, timedelta(minutes=self.access_token_expire_minutes)
        )

        # Refresh token
        refresh_token_data = {
            "sub": user.username,
            "user_id": user.id,
            "session_id": session_id,
            "type": TokenType.REFRESH.value,
        }

        refresh_token = self._create_jwt_token(
            refresh_token_data, timedelta(days=self.refresh_token_expire_days)
        )

        # Store refresh token
        self.refresh_tokens[refresh_token] = user.id

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
        }

    def _create_jwt_token(self, data: Dict, expires_delta: timedelta) -> str:
        """Create JWT token with expiration."""
        to_encode = data.copy()
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})

        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(
        self, token: str, token_type: TokenType = TokenType.ACCESS
    ) -> Optional[TokenData]:
        """Verify JWT token and return token data."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check token type
            if payload.get("type") != token_type.value:
                return None

            # Extract data
            username = payload.get("sub")
            user_id = payload.get("user_id")
            role = payload.get("role")
            permissions = payload.get("permissions", [])
            session_id = payload.get("session_id")

            if not username or not user_id:
                return None

            # Verify user still exists and is active
            user = self.users.get(username)
            if not user or not user.is_active:
                return None

            return TokenData(
                user_id=user_id,
                username=username,
                role=Role(role),
                token_type=token_type,
                permissions=permissions,
                session_id=session_id,
            )

        except (jwt.ExpiredSignatureError, jwt.JWTError, ValueError) as e:
            logger.warning(f"Token verification failed: {e}")
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Create new access token from refresh token."""
        token_data = self.verify_token(refresh_token, TokenType.REFRESH)
        if not token_data:
            return None

        user = self.users.get(token_data.username)
        if not user:
            return None

        # Create new access token only
        access_token_data = {
            "sub": user.username,
            "user_id": user.id,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "session_id": token_data.session_id,
            "type": TokenType.ACCESS.value,
        }

        access_token = self._create_jwt_token(
            access_token_data, timedelta(minutes=self.access_token_expire_minutes)
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
        }

    def create_api_key(self, user_id: str, name: str = None) -> Tuple[bool, str]:
        """Create API key for user."""
        user = next((u for u in self.users.values() if u.id == user_id), None)
        if not user:
            return False, "User not found"

        # Generate API key
        api_key = f"llm_ab_{secrets.token_urlsafe(32)}"

        # Store API key
        self.api_keys[api_key] = user_id
        user.api_keys.append(api_key)

        logger.info(f"API key created for user {user.username}")
        return True, api_key

    def verify_api_key(self, api_key: str) -> Optional[User]:
        """Verify API key and return user."""
        user_id = self.api_keys.get(api_key)
        if not user_id:
            return None

        user = next((u for u in self.users.values() if u.id == user_id), None)
        if not user or not user.is_active:
            return None

        return user

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""
        if api_key in self.api_keys:
            user_id = self.api_keys[api_key]
            del self.api_keys[api_key]

            # Remove from user's list
            user = next((u for u in self.users.values() if u.id == user_id), None)
            if user and api_key in user.api_keys:
                user.api_keys.remove(api_key)

            logger.info(f"API key revoked: {api_key[:20]}...")
            return True

        return False

    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return user.has_permission(permission)

    def require_permissions(self, *permissions: Permission):
        """Decorator to require specific permissions."""

        def decorator(func):
            def wrapper(user: User, *args, **kwargs):
                for permission in permissions:
                    if not self.has_permission(user, permission):
                        raise PermissionError(f"Permission '{permission.value}' required")
                return func(user, *args, **kwargs)

            return wrapper

        return decorator


# Global auth system instance
_auth_system: Optional[EnhancedAuthSystem] = None


def get_auth_system() -> EnhancedAuthSystem:
    """Get global authentication system."""
    global _auth_system
    if _auth_system is None:
        _auth_system = EnhancedAuthSystem()
    return _auth_system
