"""Advanced input validation and sanitization system."""

import html
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import bleach

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AttackType(str, Enum):
    """Types of detected attacks."""

    XSS = "xss"
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    LDAP_INJECTION = "ldap_injection"
    XPATH_INJECTION = "xpath_injection"
    XXE = "xxe"
    CSRF = "csrf"
    PATH_TRAVERSAL = "path_traversal"
    BUFFER_OVERFLOW = "buffer_overflow"
    FORMAT_STRING = "format_string"
    CODE_INJECTION = "code_injection"


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    cleaned_value: Any
    issues: List[str]
    severity: ValidationSeverity
    detected_attacks: List[AttackType]
    confidence: float  # 0.0 - 1.0


class SecurityInputValidator:
    """Advanced input validation with attack detection."""

    def __init__(self):
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onclick\s*=",
            r"onmouseover\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<applet[^>]*>",
            r"<meta[^>]*>",
            r"eval\s*\(",
            r"alert\s*\(",
            r"confirm\s*\(",
            r"prompt\s*\(",
            r"document\.(cookie|write|writeln)",
            r"window\.(location|open)",
        ]

        # SQL injection patterns
        self.sql_patterns = [
            r"(\bunion\b|\bselect\b|\binsert\b|\bupdate\b|\bdelete\b|\bdrop\b|\bcreate\b|\balter\b)",
            r'(\bor\b|\band\b)\s+[\'"]*\d+[\'"]*\s*[=<>]',
            r'[\'"]\s*;\s*--',
            r'[\'"]\s*;\s*#',
            r'[\'"]\s*;\s*/\*',
            r"\|\|.*\(.*\)",
            r"0x[0-9a-f]+",
            r"char\s*\(\s*\d+\s*\)",
            r"ascii\s*\(\s*.*\s*\)",
            r"substring\s*\(",
            r"@@version",
            r"\bexec\s*\(",
            r"\bsp_\w+",
            r"\bxp_\w+",
        ]

        # Command injection patterns
        self.command_patterns = [
            r"[;&|`$(){}[\]\\]",
            r"\b(cat|ls|pwd|id|whoami|uname|ps|netstat|ifconfig|ping|nslookup|dig)\b",
            r"\b(rm|mv|cp|chmod|chown|su|sudo|passwd)\b",
            r"\b(nc|netcat|telnet|ssh|ftp|wget|curl)\b",
            r"\b(python|perl|ruby|bash|sh|cmd|powershell)\b",
            r"[>|<]+",
            r"&&|\|\|",
        ]

        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\./+",
            r"\.\.\\+",
            r"%2e%2e%2f",
            r"%2e%2e\\",
            r"%252e%252e%252f",
            r"\.\.%2f",
            r"\.\.%5c",
        ]

        # LDAP injection patterns
        self.ldap_patterns = [
            r"[()&|!*=<>~]",
            r"\x00",
            r"objectClass=",
            r"cn=",
            r"uid=",
            r"dc=",
        ]

        # Compile patterns for performance
        self.compiled_xss = [re.compile(pattern, re.IGNORECASE) for pattern in self.xss_patterns]
        self.compiled_sql = [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_patterns]
        self.compiled_command = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.command_patterns
        ]
        self.compiled_path = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.path_traversal_patterns
        ]
        self.compiled_ldap = [re.compile(pattern, re.IGNORECASE) for pattern in self.ldap_patterns]

        # Allowed tags and attributes for HTML sanitization
        self.allowed_tags = [
            "p",
            "br",
            "strong",
            "em",
            "u",
            "i",
            "b",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "ul",
            "ol",
            "li",
            "blockquote",
            "code",
            "pre",
        ]

        self.allowed_attributes = {
            "*": ["class"],
            "a": ["href", "title"],
            "img": ["src", "alt", "width", "height"],
        }

        # Maximum lengths for different input types
        self.max_lengths = {
            "string": 10000,
            "text": 50000,
            "name": 100,
            "email": 254,
            "url": 2000,
            "password": 128,
            "username": 50,
            "json": 100000,
        }

    def _detect_attacks(self, value: str) -> Tuple[List[AttackType], float]:
        """Detect potential attacks in input."""
        if not isinstance(value, str):
            return [], 0.0

        detected_attacks = []
        confidence_scores = []

        # Check for XSS
        xss_matches = sum(1 for pattern in self.compiled_xss if pattern.search(value))
        if xss_matches > 0:
            detected_attacks.append(AttackType.XSS)
            confidence_scores.append(min(xss_matches / 5.0, 1.0))

        # Check for SQL injection
        sql_matches = sum(1 for pattern in self.compiled_sql if pattern.search(value))
        if sql_matches > 0:
            detected_attacks.append(AttackType.SQL_INJECTION)
            confidence_scores.append(min(sql_matches / 3.0, 1.0))

        # Check for command injection
        cmd_matches = sum(1 for pattern in self.compiled_command if pattern.search(value))
        if cmd_matches > 0:
            detected_attacks.append(AttackType.COMMAND_INJECTION)
            confidence_scores.append(min(cmd_matches / 3.0, 1.0))

        # Check for path traversal
        path_matches = sum(1 for pattern in self.compiled_path if pattern.search(value))
        if path_matches > 0:
            detected_attacks.append(AttackType.PATH_TRAVERSAL)
            confidence_scores.append(min(path_matches / 2.0, 1.0))

        # Check for LDAP injection
        ldap_matches = sum(1 for pattern in self.compiled_ldap if pattern.search(value))
        if ldap_matches > 1:  # More lenient as these chars are common
            detected_attacks.append(AttackType.LDAP_INJECTION)
            confidence_scores.append(min(ldap_matches / 4.0, 1.0))

        # Check for XXE patterns
        if "<!DOCTYPE" in value.upper() or "<!ENTITY" in value.upper():
            detected_attacks.append(AttackType.XXE)
            confidence_scores.append(0.8)

        # Check for code injection patterns
        code_patterns = ["eval(", "exec(", "system(", "shell_exec(", "passthru("]
        if any(pattern in value.lower() for pattern in code_patterns):
            detected_attacks.append(AttackType.CODE_INJECTION)
            confidence_scores.append(0.9)

        # Calculate overall confidence
        max_confidence = max(confidence_scores) if confidence_scores else 0.0

        return detected_attacks, max_confidence

    def _sanitize_string(self, value: str, allow_html: bool = False) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return str(value)

        # Normalize Unicode
        value = unicodedata.normalize("NFKC", value)

        # Remove null bytes and control characters
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

        if allow_html:
            # Use bleach to sanitize HTML
            value = bleach.clean(
                value, tags=self.allowed_tags, attributes=self.allowed_attributes, strip=True
            )
        else:
            # Escape HTML entities
            value = html.escape(value, quote=True)

        # Limit length
        max_len = self.max_lengths.get("string", 10000)
        if len(value) > max_len:
            value = value[:max_len]

        return value.strip()

    def _validate_email(self, value: str) -> Tuple[bool, List[str]]:
        """Validate email address."""
        issues = []

        if len(value) > self.max_lengths["email"]:
            issues.append(f"Email too long (max {self.max_lengths['email']} chars)")

        # Basic email regex (RFC 5322 compliant subset)
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

        if not email_pattern.match(value):
            issues.append("Invalid email format")

        # Check for suspicious patterns
        if ".." in value or value.startswith(".") or value.endswith("."):
            issues.append("Invalid email structure")

        return len(issues) == 0, issues

    def _validate_url(self, value: str) -> Tuple[bool, List[str]]:
        """Validate URL."""
        issues = []

        if len(value) > self.max_lengths["url"]:
            issues.append(f"URL too long (max {self.max_lengths['url']} chars)")

        # Basic URL validation
        url_pattern = re.compile(r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?$")

        if not url_pattern.match(value):
            issues.append("Invalid URL format")

        # Check for suspicious schemes
        if re.match(r"^(javascript|vbscript|data|file):", value, re.IGNORECASE):
            issues.append("Dangerous URL scheme detected")

        return len(issues) == 0, issues

    def _validate_json(self, value: str) -> Tuple[bool, Any, List[str]]:
        """Validate and parse JSON."""
        issues = []
        parsed_value = None

        if len(value) > self.max_lengths["json"]:
            issues.append(f"JSON too long (max {self.max_lengths['json']} chars)")
            return False, None, issues

        try:
            parsed_value = json.loads(value)

            # Check for excessive nesting (DoS protection)
            def count_depth(obj, depth=0):
                if depth > 20:  # Max depth limit
                    return depth
                if isinstance(obj, dict):
                    return max(count_depth(v, depth + 1) for v in obj.values()) if obj else depth
                elif isinstance(obj, list):
                    return max(count_depth(item, depth + 1) for item in obj) if obj else depth
                return depth

            if count_depth(parsed_value) > 20:
                issues.append("JSON nesting too deep (max 20 levels)")

        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON: {str(e)}")
        except (ValueError, RecursionError) as e:
            issues.append(f"JSON parsing error: {str(e)}")

        return len(issues) == 0, parsed_value, issues

    def validate_input(
        self,
        value: Any,
        input_type: str = "string",
        allow_html: bool = False,
        required: bool = True,
        custom_patterns: List[str] = None,
    ) -> ValidationResult:
        """Validate input with comprehensive security checks."""

        # Handle None/empty values
        if value is None or (isinstance(value, str) and not value.strip()):
            if required:
                return ValidationResult(
                    is_valid=False,
                    cleaned_value=None,
                    issues=["Value is required"],
                    severity=ValidationSeverity.ERROR,
                    detected_attacks=[],
                    confidence=0.0,
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    cleaned_value=None,
                    issues=[],
                    severity=ValidationSeverity.INFO,
                    detected_attacks=[],
                    confidence=0.0,
                )

        issues = []
        detected_attacks = []
        confidence = 0.0
        cleaned_value = value
        severity = ValidationSeverity.INFO

        # Convert to string for analysis
        str_value = str(value) if not isinstance(value, str) else value

        # Detect attacks
        detected_attacks, confidence = self._detect_attacks(str_value)

        if detected_attacks:
            severity = ValidationSeverity.CRITICAL if confidence > 0.8 else ValidationSeverity.ERROR
            issues.extend(
                [f"Potential {attack.value} attack detected" for attack in detected_attacks]
            )

        # Type-specific validation
        if input_type == "email":
            is_valid, email_issues = self._validate_email(str_value)
            issues.extend(email_issues)
            if is_valid:
                cleaned_value = str_value.lower().strip()

        elif input_type == "url":
            is_valid, url_issues = self._validate_url(str_value)
            issues.extend(url_issues)
            if is_valid:
                cleaned_value = str_value.strip()

        elif input_type == "json":
            is_valid, parsed_json, json_issues = self._validate_json(str_value)
            issues.extend(json_issues)
            if is_valid:
                cleaned_value = parsed_json

        elif input_type in ["string", "text", "name", "username", "password"]:
            cleaned_value = self._sanitize_string(str_value, allow_html)

            # Length validation
            max_len = self.max_lengths.get(input_type, 10000)
            if len(str_value) > max_len:
                issues.append(f"Input too long (max {max_len} characters)")
                severity = max(severity, ValidationSeverity.WARNING)

            # Username specific validation
            if input_type == "username":
                if not re.match(r"^[a-zA-Z0-9_.-]+$", cleaned_value):
                    issues.append("Username contains invalid characters")
                    severity = max(severity, ValidationSeverity.ERROR)

            # Password specific validation
            elif input_type == "password":
                if len(cleaned_value) < 8:
                    issues.append("Password too short (minimum 8 characters)")
                    severity = max(severity, ValidationSeverity.ERROR)

        # Custom pattern validation
        if custom_patterns:
            for pattern in custom_patterns:
                try:
                    if re.search(pattern, str_value, re.IGNORECASE):
                        issues.append(f"Input matches restricted pattern: {pattern}")
                        severity = max(severity, ValidationSeverity.WARNING)
                except re.error:
                    logger.warning(f"Invalid regex pattern: {pattern}")

        # Final validation check
        is_valid = (
            len(
                [
                    issue
                    for issue in issues
                    if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
                ]
            )
            == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            cleaned_value=cleaned_value,
            issues=issues,
            severity=severity,
            detected_attacks=detected_attacks,
            confidence=confidence,
        )

    def validate_batch(
        self, inputs: Dict[str, Any], validation_rules: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ValidationResult]:
        """Validate multiple inputs at once."""
        results = {}

        for field_name, value in inputs.items():
            rules = validation_rules.get(field_name, {})

            result = self.validate_input(
                value=value,
                input_type=rules.get("type", "string"),
                allow_html=rules.get("allow_html", False),
                required=rules.get("required", True),
                custom_patterns=rules.get("patterns", None),
            )

            results[field_name] = result

        return results

    def is_safe_filename(self, filename: str) -> bool:
        """Check if filename is safe."""
        if not filename or filename in [".", ".."]:
            return False

        # Check for path traversal
        if any(pattern.search(filename) for pattern in self.compiled_path):
            return False

        # Check for dangerous extensions
        dangerous_extensions = {
            ".exe",
            ".bat",
            ".cmd",
            ".com",
            ".scr",
            ".pif",
            ".vbs",
            ".js",
            ".jar",
            ".php",
            ".asp",
            ".aspx",
            ".jsp",
            ".py",
            ".rb",
            ".sh",
        }

        _, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        if f".{ext.lower()}" in dangerous_extensions:
            return False

        # Check for control characters
        if any(ord(c) < 32 for c in filename):
            return False

        return True

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        if not filename:
            return "unnamed_file"

        # Remove path components
        filename = filename.split("/")[-1].split("\\")[-1]

        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
            max_name_len = 255 - len(ext) - 1
            filename = f"{name[:max_name_len]}.{ext}"

        return filename or "unnamed_file"


# Global validator instance
_input_validator: Optional[SecurityInputValidator] = None


def get_input_validator() -> SecurityInputValidator:
    """Get global input validator instance."""
    global _input_validator
    if _input_validator is None:
        _input_validator = SecurityInputValidator()
    return _input_validator
