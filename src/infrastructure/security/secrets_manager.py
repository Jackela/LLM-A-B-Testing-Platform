"""Secrets management with rotation and secure storage."""

import base64
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class SecretMetadata:
    """Metadata for a secret."""

    name: str
    created_at: datetime
    last_rotated: datetime
    rotation_interval: int  # days
    tags: List[str]
    is_active: bool = True


class SecretsManager:
    """Manage secrets with encryption and rotation."""

    def __init__(self, storage_path: str = "secrets"):
        self.storage_path = storage_path
        self._master_key = self._get_or_create_master_key()
        self._cipher = Fernet(self._master_key)
        self._secrets: Dict[str, Any] = {}
        self._metadata: Dict[str, SecretMetadata] = {}
        self._load_secrets()

    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key."""
        key_path = f"{self.storage_path}/.master_key"

        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return f.read()
        else:
            # Create storage directory
            os.makedirs(self.storage_path, exist_ok=True)

            # Generate new master key
            key = Fernet.generate_key()
            with open(key_path, "wb") as f:
                f.write(key)

            # Secure the key file (Unix-like systems)
            try:
                os.chmod(key_path, 0o600)
            except OSError:
                logger.warning("Could not set secure permissions on master key file")

            return key

    def _load_secrets(self) -> None:
        """Load secrets from encrypted storage."""
        secrets_path = f"{self.storage_path}/secrets.enc"
        metadata_path = f"{self.storage_path}/metadata.enc"

        try:
            # Load secrets
            if os.path.exists(secrets_path):
                with open(secrets_path, "rb") as f:
                    encrypted_data = f.read()
                decrypted_data = self._cipher.decrypt(encrypted_data)
                self._secrets = json.loads(decrypted_data.decode())

            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    encrypted_data = f.read()
                decrypted_data = self._cipher.decrypt(encrypted_data)
                metadata_dict = json.loads(decrypted_data.decode())

                # Convert to SecretMetadata objects
                for name, data in metadata_dict.items():
                    data["created_at"] = datetime.fromisoformat(data["created_at"])
                    data["last_rotated"] = datetime.fromisoformat(data["last_rotated"])
                    self._metadata[name] = SecretMetadata(**data)

        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            self._secrets = {}
            self._metadata = {}

    def _save_secrets(self) -> None:
        """Save secrets to encrypted storage."""
        os.makedirs(self.storage_path, exist_ok=True)

        try:
            # Save secrets
            secrets_path = f"{self.storage_path}/secrets.enc"
            secrets_json = json.dumps(self._secrets).encode()
            encrypted_secrets = self._cipher.encrypt(secrets_json)

            with open(secrets_path, "wb") as f:
                f.write(encrypted_secrets)

            # Save metadata
            metadata_path = f"{self.storage_path}/metadata.enc"
            metadata_dict = {}
            for name, metadata in self._metadata.items():
                data = asdict(metadata)
                data["created_at"] = data["created_at"].isoformat()
                data["last_rotated"] = data["last_rotated"].isoformat()
                metadata_dict[name] = data

            metadata_json = json.dumps(metadata_dict).encode()
            encrypted_metadata = self._cipher.encrypt(metadata_json)

            with open(metadata_path, "wb") as f:
                f.write(encrypted_metadata)

        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise

    def store_secret(
        self, name: str, value: str, rotation_interval: int = 90, tags: List[str] = None
    ) -> None:
        """Store a secret with metadata."""
        tags = tags or []
        now = datetime.utcnow()

        self._secrets[name] = value
        self._metadata[name] = SecretMetadata(
            name=name,
            created_at=now,
            last_rotated=now,
            rotation_interval=rotation_interval,
            tags=tags,
        )

        self._save_secrets()
        logger.info(f"Secret '{name}' stored successfully")

    def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret value."""
        return self._secrets.get(name)

    def rotate_secret(self, name: str, new_value: str) -> None:
        """Rotate a secret to a new value."""
        if name not in self._secrets:
            raise ValueError(f"Secret '{name}' not found")

        self._secrets[name] = new_value
        if name in self._metadata:
            self._metadata[name].last_rotated = datetime.utcnow()

        self._save_secrets()
        logger.info(f"Secret '{name}' rotated successfully")

    def delete_secret(self, name: str) -> None:
        """Delete a secret."""
        if name in self._secrets:
            del self._secrets[name]
        if name in self._metadata:
            del self._metadata[name]

        self._save_secrets()
        logger.info(f"Secret '{name}' deleted")

    def list_secrets(self) -> List[str]:
        """List all secret names."""
        return list(self._secrets.keys())

    def get_secrets_needing_rotation(self) -> List[str]:
        """Get secrets that need rotation based on their interval."""
        now = datetime.utcnow()
        needing_rotation = []

        for name, metadata in self._metadata.items():
            if not metadata.is_active:
                continue

            days_since_rotation = (now - metadata.last_rotated).days
            if days_since_rotation >= metadata.rotation_interval:
                needing_rotation.append(name)

        return needing_rotation

    def get_secret_metadata(self, name: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret."""
        return self._metadata.get(name)

    def export_secrets(self, include_values: bool = False) -> Dict[str, Any]:
        """Export secrets metadata and optionally values."""
        export_data = {}

        for name, metadata in self._metadata.items():
            export_data[name] = {
                "metadata": asdict(metadata),
                "value": self._secrets[name] if include_values else "***HIDDEN***",
            }

        return export_data


class EnvironmentSecretsProvider:
    """Provide secrets from environment variables with fallback."""

    def __init__(self, secrets_manager: SecretsManager):
        self.secrets_manager = secrets_manager
        self._cache: Dict[str, str] = {}

    def get_secret(
        self, name: str, env_var: str = None, default: str = None, required: bool = True
    ) -> Optional[str]:
        """Get secret from environment, secrets manager, or default."""
        # Check cache first
        cache_key = f"{name}:{env_var}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try environment variable
        env_var = env_var or name.upper()
        value = os.getenv(env_var)

        if value:
            self._cache[cache_key] = value
            return value

        # Try secrets manager
        value = self.secrets_manager.get_secret(name)
        if value:
            self._cache[cache_key] = value
            return value

        # Use default or raise error
        if default is not None:
            self._cache[cache_key] = default
            return default

        if required:
            raise ValueError(f"Required secret '{name}' not found in environment or secrets store")

        return None

    def get_database_url(self) -> str:
        """Get database URL with all components."""
        host = self.get_secret("db_host", "DATABASE_HOST", "localhost")
        port = self.get_secret("db_port", "DATABASE_PORT", "5432")
        user = self.get_secret("db_user", "DATABASE_USER", "postgres")
        password = self.get_secret("db_password", "DATABASE_PASSWORD", "password")
        database = self.get_secret("db_name", "DATABASE_NAME", "llm_ab_testing")

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def get_redis_url(self) -> str:
        """Get Redis URL."""
        host = self.get_secret("redis_host", "REDIS_HOST", "localhost")
        port = self.get_secret("redis_port", "REDIS_PORT", "6379")
        password = self.get_secret("redis_password", "REDIS_PASSWORD", required=False)

        if password:
            return f"redis://:{password}@{host}:{port}/0"
        return f"redis://{host}:{port}/0"

    def get_jwt_secret(self) -> str:
        """Get JWT secret key."""
        return self.get_secret("jwt_secret", "JWT_SECRET_KEY", required=True)

    def get_encryption_key(self) -> str:
        """Get encryption key."""
        return self.get_secret("encryption_key", "ENCRYPTION_KEY", required=True)

    def get_api_keys(self) -> Dict[str, str]:
        """Get all external API keys."""
        return {
            "openai": self.get_secret("openai_api_key", "OPENAI_API_KEY", required=False),
            "anthropic": self.get_secret("anthropic_api_key", "ANTHROPIC_API_KEY", required=False),
            "google": self.get_secret("google_api_key", "GOOGLE_API_KEY", required=False),
        }

    def clear_cache(self) -> None:
        """Clear the secrets cache."""
        self._cache.clear()


# Global instances
_secrets_manager: Optional[SecretsManager] = None
_env_provider: Optional[EnvironmentSecretsProvider] = None


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        storage_path = os.getenv("SECRETS_STORAGE_PATH", "secrets")
        _secrets_manager = SecretsManager(storage_path)
    return _secrets_manager


def get_env_secrets() -> EnvironmentSecretsProvider:
    """Get global environment secrets provider."""
    global _env_provider
    if _env_provider is None:
        _env_provider = EnvironmentSecretsProvider(get_secrets_manager())
    return _env_provider
