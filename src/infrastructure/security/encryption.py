"""Data encryption utilities for sensitive information."""

import base64
import logging
import os
from typing import Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class DataEncryption:
    """Handle encryption/decryption of sensitive data."""

    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize encryption with key from environment or parameter."""
        if encryption_key:
            self.key = encryption_key.encode()
        else:
            env_key = os.getenv("ENCRYPTION_KEY")
            if not env_key:
                raise ValueError(
                    "ENCRYPTION_KEY environment variable is required for security. "
                    "Never use default encryption keys in production."
                )
            self.key = env_key.encode()

        # Derive key using PBKDF2
        salt = b"stable_salt"  # In production, use random salt per record
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(self.key))
        self.cipher = Fernet(key)

    def encrypt(self, data: str) -> str:
        """Encrypt sensitive string data."""
        try:
            if not data:
                return data

            encrypted_data = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt encrypted string data."""
        try:
            if not encrypted_data:
                return encrypted_data

            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def encrypt_dict(self, data: dict, sensitive_fields: list[str]) -> dict:
        """Encrypt specific fields in a dictionary."""
        result = data.copy()
        for field in sensitive_fields:
            if field in result and result[field]:
                result[field] = self.encrypt(str(result[field]))
        return result

    def decrypt_dict(self, data: dict, sensitive_fields: list[str]) -> dict:
        """Decrypt specific fields in a dictionary."""
        result = data.copy()
        for field in sensitive_fields:
            if field in result and result[field]:
                result[field] = self.decrypt(result[field])
        return result


class PIIHandler:
    """Handle PII data anonymization and tokenization."""

    def __init__(self, encryption: DataEncryption):
        self.encryption = encryption
        self.pii_fields = {
            "email",
            "username",
            "phone",
            "address",
            "ssn",
            "credit_card",
            "api_key",
            "token",
            "password",
        }

    def anonymize_email(self, email: str) -> str:
        """Anonymize email address."""
        if not email or "@" not in email:
            return email

        local, domain = email.split("@", 1)
        if len(local) <= 2:
            return f"**@{domain}"
        return f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}@{domain}"

    def tokenize_sensitive_data(self, data: dict) -> dict:
        """Replace sensitive data with tokens."""
        result = {}
        for key, value in data.items():
            if key.lower() in self.pii_fields and value:
                # Create deterministic token for same data
                token_data = f"{key}:{value}"
                result[f"{key}_token"] = self.encryption.encrypt(token_data)
                result[key] = self._mask_value(str(value), key)
            else:
                result[key] = value
        return result

    def _mask_value(self, value: str, field_type: str) -> str:
        """Mask sensitive values based on type."""
        if field_type == "email":
            return self.anonymize_email(value)
        elif field_type in ["password", "token", "api_key"]:
            return "***REDACTED***"
        elif len(value) > 4:
            return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
        else:
            return "*" * len(value)


# Global encryption instance
_encryption_instance: Optional[DataEncryption] = None


def get_encryption() -> DataEncryption:
    """Get global encryption instance."""
    global _encryption_instance
    if _encryption_instance is None:
        _encryption_instance = DataEncryption()
    return _encryption_instance


def get_pii_handler() -> PIIHandler:
    """Get global PII handler instance."""
    return PIIHandler(get_encryption())
