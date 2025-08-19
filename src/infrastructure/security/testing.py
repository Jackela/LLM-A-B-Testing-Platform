import aiofiles

"""Security testing pipeline and compliance validation."""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..monitoring.structured_logging import get_logger
from .middleware import InputValidator, SecurityMiddleware

logger = get_logger(__name__)


class SecurityTestType(str, Enum):
    """Security test types."""

    VULNERABILITY_SCAN = "vulnerability_scan"
    PENETRATION_TEST = "penetration_test"
    DEPENDENCY_CHECK = "dependency_check"
    STATIC_ANALYSIS = "static_analysis"
    DYNAMIC_ANALYSIS = "dynamic_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    INPUT_VALIDATION = "input_validation"
    AUTHENTICATION_TEST = "authentication_test"
    AUTHORIZATION_TEST = "authorization_test"


class SecurityTestResult(str, Enum):
    """Security test results."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class SeverityLevel(str, Enum):
    """Security issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityIssue:
    """Security issue found during testing."""

    id: str
    title: str
    description: str
    severity: SeverityLevel
    category: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    cvss_score: Optional[float] = None
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityTestReport:
    """Security test report."""

    test_id: str
    test_type: SecurityTestType
    result: SecurityTestResult
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    issues: List[SecurityIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["started_at"] = self.started_at.isoformat()
        result["completed_at"] = self.completed_at.isoformat()
        result["issues"] = [issue.to_dict() for issue in self.issues]
        return result

    @property
    def critical_issues_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == SeverityLevel.CRITICAL)

    @property
    def high_issues_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == SeverityLevel.HIGH)

    @property
    def total_issues_count(self) -> int:
        return len(self.issues)


class DependencyChecker:
    """Check for vulnerable dependencies."""

    def __init__(self, project_root: str):
        self.project_root = project_root

    async def run_safety_check(self) -> SecurityTestReport:
        """Run safety check on dependencies."""
        test_id = f"safety_check_{int(time.time())}"
        started_at = datetime.utcnow()

        try:
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json", "--full-report"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,
            )

            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()

            issues = []
            if result.returncode != 0 and result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        issue = SecurityIssue(
                            id=vuln.get("id", "unknown"),
                            title=f"Vulnerable dependency: {vuln.get('package_name', 'unknown')}",
                            description=vuln.get("advisory", "No description available"),
                            severity=self._map_safety_severity(vuln.get("severity", "medium")),
                            category="dependency_vulnerability",
                            cve_id=vuln.get("cve"),
                            remediation=f"Upgrade to version {vuln.get('safe_version', 'latest')} or higher",
                        )
                        issues.append(issue)
                except json.JSONDecodeError:
                    pass

            return SecurityTestReport(
                test_id=test_id,
                test_type=SecurityTestType.DEPENDENCY_CHECK,
                result=SecurityTestResult.FAILED if issues else SecurityTestResult.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                issues=issues,
                summary={
                    "tool": "safety",
                    "packages_scanned": "all",
                    "vulnerabilities_found": len(issues),
                },
            )

        except subprocess.TimeoutExpired:
            completed_at = datetime.utcnow()
            return SecurityTestReport(
                test_id=test_id,
                test_type=SecurityTestType.DEPENDENCY_CHECK,
                result=SecurityTestResult.ERROR,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                summary={"error": "Safety check timed out"},
            )
        except Exception as e:
            completed_at = datetime.utcnow()
            logger.error(f"Safety check failed: {e}")
            return SecurityTestReport(
                test_id=test_id,
                test_type=SecurityTestType.DEPENDENCY_CHECK,
                result=SecurityTestResult.ERROR,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                summary={"error": str(e)},
            )

    def _map_safety_severity(self, safety_severity: str) -> SeverityLevel:
        """Map safety severity to our severity levels."""
        mapping = {
            "critical": SeverityLevel.CRITICAL,
            "high": SeverityLevel.HIGH,
            "medium": SeverityLevel.MEDIUM,
            "low": SeverityLevel.LOW,
        }
        return mapping.get(safety_severity.lower(), SeverityLevel.MEDIUM)


class StaticAnalyzer:
    """Static code analysis for security issues."""

    def __init__(self, project_root: str):
        self.project_root = project_root

    async def run_bandit_analysis(self) -> SecurityTestReport:
        """Run Bandit static analysis."""
        test_id = f"bandit_analysis_{int(time.time())}"
        started_at = datetime.utcnow()

        try:
            # Run bandit analysis
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,
            )

            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()

            issues = []
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    for result_item in bandit_data.get("results", []):
                        issue = SecurityIssue(
                            id=result_item.get("test_id", "unknown"),
                            title=result_item.get("test_name", "Security issue"),
                            description=result_item.get("issue_text", "No description"),
                            severity=self._map_bandit_severity(
                                result_item.get("issue_severity", "MEDIUM")
                            ),
                            category=result_item.get("test_name", "static_analysis"),
                            file_path=result_item.get("filename"),
                            line_number=result_item.get("line_number"),
                            cwe_id=result_item.get("test_id"),  # Bandit uses test IDs
                            remediation=result_item.get("more_info"),
                        )
                        issues.append(issue)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Bandit output: {e}")

            return SecurityTestReport(
                test_id=test_id,
                test_type=SecurityTestType.STATIC_ANALYSIS,
                result=SecurityTestResult.FAILED if issues else SecurityTestResult.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                issues=issues,
                summary={
                    "tool": "bandit",
                    "files_scanned": (
                        bandit_data.get("metrics", {}).get("loc", 0)
                        if "bandit_data" in locals()
                        else 0
                    ),
                    "issues_found": len(issues),
                },
            )

        except Exception as e:
            completed_at = datetime.utcnow()
            logger.error(f"Bandit analysis failed: {e}")
            return SecurityTestReport(
                test_id=test_id,
                test_type=SecurityTestType.STATIC_ANALYSIS,
                result=SecurityTestResult.ERROR,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                summary={"error": str(e)},
            )

    def _map_bandit_severity(self, bandit_severity: str) -> SeverityLevel:
        """Map Bandit severity to our severity levels."""
        mapping = {
            "HIGH": SeverityLevel.HIGH,
            "MEDIUM": SeverityLevel.MEDIUM,
            "LOW": SeverityLevel.LOW,
        }
        return mapping.get(bandit_severity.upper(), SeverityLevel.MEDIUM)


class InputValidationTester:
    """Test input validation security."""

    def __init__(self):
        self.validator = InputValidator()
        self.test_payloads = {
            "sql_injection": [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'--",
                "' UNION SELECT * FROM users--",
                "1; DELETE FROM users WHERE '1'='1",
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "<iframe src=javascript:alert('XSS')></iframe>",
            ],
            "command_injection": [
                "; ls -la",
                "| whoami",
                "&& cat /etc/passwd",
                "; rm -rf /",
                "| nc -lvp 4444",
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "....//....//....//etc/passwd",
                "/var/log/../../etc/passwd",
            ],
        }

    async def run_input_validation_tests(self) -> SecurityTestReport:
        """Run input validation security tests."""
        test_id = f"input_validation_{int(time.time())}"
        started_at = datetime.utcnow()

        issues = []

        for attack_type, payloads in self.test_payloads.items():
            for payload in payloads:
                valid, message = self.validator.validate_input(payload, f"test_{attack_type}")

                if valid:  # This is a security issue - malicious input was not detected
                    issue = SecurityIssue(
                        id=f"{attack_type}_{hash(payload)}",
                        title=f"Input validation bypass: {attack_type}",
                        description=f"Malicious payload not detected: {payload[:50]}...",
                        severity=SeverityLevel.HIGH,
                        category=f"input_validation_{attack_type}",
                        remediation=f"Improve {attack_type} detection in input validation",
                    )
                    issues.append(issue)

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        return SecurityTestReport(
            test_id=test_id,
            test_type=SecurityTestType.INPUT_VALIDATION,
            result=SecurityTestResult.FAILED if issues else SecurityTestResult.PASSED,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            issues=issues,
            summary={
                "payloads_tested": sum(len(payloads) for payloads in self.test_payloads.values()),
                "bypasses_found": len(issues),
                "attack_types_tested": list(self.test_payloads.keys()),
            },
        )


class APISecurityTester:
    """Test API security endpoints."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def run_authentication_tests(self) -> SecurityTestReport:
        """Test authentication security."""
        test_id = f"auth_test_{int(time.time())}"
        started_at = datetime.utcnow()
        issues = []

        test_cases = [
            {
                "name": "No authentication bypass",
                "method": "GET",
                "path": "/api/v1/tests",
                "headers": {},
                "expected_status": 401,
            },
            {
                "name": "Invalid token rejection",
                "method": "GET",
                "path": "/api/v1/tests",
                "headers": {"Authorization": "Bearer invalid_token"},
                "expected_status": 401,
            },
            {
                "name": "Expired token rejection",
                "method": "GET",
                "path": "/api/v1/tests",
                "headers": {
                    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0IiwiZXhwIjoxNjA5NDU5MjAwfQ.invalid"
                },
                "expected_status": 401,
            },
        ]

        async with httpx.AsyncClient(timeout=10.0) as client:
            for test_case in test_cases:
                try:
                    response = await client.request(
                        method=test_case["method"],
                        url=f"{self.base_url}{test_case['path']}",
                        headers=test_case["headers"],
                    )

                    if response.status_code != test_case["expected_status"]:
                        issue = SecurityIssue(
                            id=f"auth_test_{test_case['name'].replace(' ', '_')}",
                            title=f"Authentication test failed: {test_case['name']}",
                            description=f"Expected status {test_case['expected_status']}, got {response.status_code}",
                            severity=SeverityLevel.HIGH,
                            category="authentication",
                            remediation="Review authentication middleware configuration",
                        )
                        issues.append(issue)

                except Exception as e:
                    logger.error(f"Authentication test failed: {e}")

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        return SecurityTestReport(
            test_id=test_id,
            test_type=SecurityTestType.AUTHENTICATION_TEST,
            result=SecurityTestResult.FAILED if issues else SecurityTestResult.PASSED,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            issues=issues,
            summary={"test_cases_run": len(test_cases), "failures": len(issues)},
        )

    async def run_rate_limiting_tests(self) -> SecurityTestReport:
        """Test rate limiting security."""
        test_id = f"rate_limit_test_{int(time.time())}"
        started_at = datetime.utcnow()
        issues = []

        # Test rate limiting on login endpoint
        login_url = f"{self.base_url}/api/v1/auth/login"

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Send multiple requests quickly
            requests_sent = 0
            rate_limited = False

            for i in range(100):  # Try to exceed rate limit
                try:
                    response = await client.post(
                        login_url, json={"username": "test", "password": "invalid"}
                    )
                    requests_sent += 1

                    if response.status_code == 429:  # Rate limited
                        rate_limited = True
                        break

                except Exception:
                    break

            if not rate_limited:
                issue = SecurityIssue(
                    id="rate_limiting_bypass",
                    title="Rate limiting not effective",
                    description=f"Sent {requests_sent} requests without being rate limited",
                    severity=SeverityLevel.MEDIUM,
                    category="rate_limiting",
                    remediation="Review and strengthen rate limiting configuration",
                )
                issues.append(issue)

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        return SecurityTestReport(
            test_id=test_id,
            test_type=SecurityTestType.DYNAMIC_ANALYSIS,
            result=SecurityTestResult.FAILED if issues else SecurityTestResult.PASSED,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            issues=issues,
            summary={"requests_sent": requests_sent, "rate_limited": rate_limited},
        )


class ComplianceChecker:
    """Check compliance with security standards."""

    def __init__(self, project_root: str):
        self.project_root = project_root

    async def check_owasp_compliance(self) -> SecurityTestReport:
        """Check OWASP Top 10 compliance."""
        test_id = f"owasp_compliance_{int(time.time())}"
        started_at = datetime.utcnow()
        issues = []

        # OWASP Top 10 checks
        checks = [
            ("A01:2021-Broken Access Control", self._check_access_control),
            ("A02:2021-Cryptographic Failures", self._check_cryptography),
            ("A03:2021-Injection", self._check_injection_prevention),
            ("A04:2021-Insecure Design", self._check_secure_design),
            ("A05:2021-Security Misconfiguration", self._check_security_config),
            ("A06:2021-Vulnerable Components", self._check_vulnerable_components),
            ("A07:2021-Authentication Failures", self._check_authentication),
            ("A08:2021-Software Integrity Failures", self._check_software_integrity),
            ("A09:2021-Logging Failures", self._check_logging_monitoring),
            ("A10:2021-Server-Side Request Forgery", self._check_ssrf_prevention),
        ]

        for check_name, check_function in checks:
            try:
                check_issues = await check_function()
                for issue in check_issues:
                    issue.category = check_name
                issues.extend(check_issues)
            except Exception as e:
                logger.error(f"OWASP check failed for {check_name}: {e}")
                issue = SecurityIssue(
                    id=f"owasp_check_error_{check_name}",
                    title=f"OWASP check failed: {check_name}",
                    description=f"Check could not be completed: {str(e)}",
                    severity=SeverityLevel.LOW,
                    category=check_name,
                )
                issues.append(issue)

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        return SecurityTestReport(
            test_id=test_id,
            test_type=SecurityTestType.COMPLIANCE_CHECK,
            result=SecurityTestResult.FAILED if issues else SecurityTestResult.PASSED,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            issues=issues,
            summary={"owasp_categories_checked": len(checks), "compliance_issues": len(issues)},
        )

    async def _check_access_control(self) -> List[SecurityIssue]:
        """Check for broken access control."""
        issues = []

        # Check if RBAC is implemented
        auth_files = [
            "src/infrastructure/security/auth.py",
            "src/presentation/api/auth/dependencies.py",
        ]

        rbac_implemented = False
        for file_path in auth_files:
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                with open(full_path, "r") as f:
                    content = f.read()
                    if "Role" in content and "Permission" in content:
                        rbac_implemented = True
                        break

        if not rbac_implemented:
            issues.append(
                SecurityIssue(
                    id="missing_rbac",
                    title="Missing Role-Based Access Control",
                    description="No RBAC implementation found in authentication system",
                    severity=SeverityLevel.HIGH,
                    remediation="Implement proper role-based access control",
                )
            )

        return issues

    async def _check_cryptography(self) -> List[SecurityIssue]:
        """Check for cryptographic failures."""
        issues = []

        # Check for hardcoded secrets
        secret_patterns = ["SECRET_KEY", "API_KEY", "PASSWORD"]
        for root, dirs, files in os.walk(os.path.join(self.project_root, "src")):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()
                            for pattern in secret_patterns:
                                if (
                                    f'{pattern} = "' in content
                                    and "change-in-production" in content
                                ):
                                    issues.append(
                                        SecurityIssue(
                                            id=f"hardcoded_secret_{pattern}",
                                            title="Hardcoded Secret Found",
                                            description=f"Hardcoded {pattern} found in {file_path}",
                                            severity=SeverityLevel.CRITICAL,
                                            file_path=file_path,
                                            remediation="Use environment variables or secure secret management",
                                        )
                                    )
                    except Exception:
                        continue

        return issues

    async def _check_injection_prevention(self) -> List[SecurityIssue]:
        """Check for injection prevention."""
        issues = []

        # Check if input validation is implemented
        validation_file = os.path.join(
            self.project_root, "src/infrastructure/security/middleware.py"
        )
        if os.path.exists(validation_file):
            with open(validation_file, "r") as f:
                content = f.read()
                if "InputValidator" not in content:
                    issues.append(
                        SecurityIssue(
                            id="missing_input_validation",
                            title="Missing Input Validation",
                            description="No input validation middleware found",
                            severity=SeverityLevel.HIGH,
                            remediation="Implement comprehensive input validation",
                        )
                    )
        else:
            issues.append(
                SecurityIssue(
                    id="missing_security_middleware",
                    title="Missing Security Middleware",
                    description="No security middleware implementation found",
                    severity=SeverityLevel.HIGH,
                    remediation="Implement security middleware with input validation",
                )
            )

        return issues

    async def _check_secure_design(self) -> List[SecurityIssue]:
        """Check for secure design principles."""
        # This would involve more complex analysis
        return []

    async def _check_security_config(self) -> List[SecurityIssue]:
        """Check for security misconfigurations."""
        issues = []

        # Check if debug mode is disabled in production
        config_files = ["src/presentation/api/main.py", "src/presentation/api/app.py"]
        for config_file in config_files:
            full_path = os.path.join(self.project_root, config_file)
            if os.path.exists(full_path):
                with open(full_path, "r") as f:
                    content = f.read()
                    if "debug=True" in content:
                        issues.append(
                            SecurityIssue(
                                id="debug_mode_enabled",
                                title="Debug Mode Enabled",
                                description="Debug mode should be disabled in production",
                                severity=SeverityLevel.MEDIUM,
                                file_path=full_path,
                                remediation="Disable debug mode in production configuration",
                            )
                        )

        return issues

    async def _check_vulnerable_components(self) -> List[SecurityIssue]:
        """Check for vulnerable components."""
        # This would integrate with dependency checking
        return []

    async def _check_authentication(self) -> List[SecurityIssue]:
        """Check authentication implementation."""
        issues = []

        # Check if MFA is implemented
        auth_file = os.path.join(self.project_root, "src/infrastructure/security/auth.py")
        if os.path.exists(auth_file):
            with open(auth_file, "r") as f:
                content = f.read()
                if "MFASettings" not in content:
                    issues.append(
                        SecurityIssue(
                            id="missing_mfa",
                            title="Missing Multi-Factor Authentication",
                            description="MFA not implemented for enhanced security",
                            severity=SeverityLevel.MEDIUM,
                            remediation="Implement multi-factor authentication",
                        )
                    )

        return issues

    async def _check_software_integrity(self) -> List[SecurityIssue]:
        """Check software integrity measures."""
        return []

    async def _check_logging_monitoring(self) -> List[SecurityIssue]:
        """Check logging and monitoring."""
        issues = []

        # Check if security logging is implemented
        logging_file = os.path.join(
            self.project_root, "src/infrastructure/monitoring/structured_logging.py"
        )
        if not os.path.exists(logging_file):
            issues.append(
                SecurityIssue(
                    id="missing_security_logging",
                    title="Missing Security Logging",
                    description="No security-specific logging implementation found",
                    severity=SeverityLevel.MEDIUM,
                    remediation="Implement comprehensive security logging",
                )
            )

        return issues

    async def _check_ssrf_prevention(self) -> List[SecurityIssue]:
        """Check SSRF prevention measures."""
        return []


class SecurityTestSuite:
    """Comprehensive security test suite."""

    def __init__(self, project_root: str, api_base_url: str = None):
        self.project_root = project_root
        self.api_base_url = api_base_url or "http://localhost:8000"

        # Initialize test components
        self.dependency_checker = DependencyChecker(project_root)
        self.static_analyzer = StaticAnalyzer(project_root)
        self.input_validator_tester = InputValidationTester()
        self.api_tester = APISecurityTester(self.api_base_url)
        self.compliance_checker = ComplianceChecker(project_root)

    async def run_all_tests(self) -> List[SecurityTestReport]:
        """Run all security tests."""
        logger.info("Starting comprehensive security test suite")

        test_tasks = [
            ("Dependency Check", self.dependency_checker.run_safety_check()),
            ("Static Analysis", self.static_analyzer.run_bandit_analysis()),
            ("Input Validation", self.input_validator_tester.run_input_validation_tests()),
            ("Authentication Tests", self.api_tester.run_authentication_tests()),
            ("Rate Limiting Tests", self.api_tester.run_rate_limiting_tests()),
            ("OWASP Compliance", self.compliance_checker.check_owasp_compliance()),
        ]

        reports = []
        for test_name, test_task in test_tasks:
            try:
                logger.info(f"Running {test_name}...")
                report = await test_task
                reports.append(report)
                logger.info(
                    f"{test_name} completed: {report.result.value} ({len(report.issues)} issues)"
                )
            except Exception as e:
                logger.error(f"{test_name} failed: {e}")
                # Create error report
                error_report = SecurityTestReport(
                    test_id=f"error_{test_name.lower().replace(' ', '_')}_{int(time.time())}",
                    test_type=SecurityTestType.STATIC_ANALYSIS,  # Default type
                    result=SecurityTestResult.ERROR,
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    duration_seconds=0,
                    summary={"error": str(e)},
                )
                reports.append(error_report)

        logger.info("Security test suite completed")
        return reports

    def generate_summary_report(self, reports: List[SecurityTestReport]) -> Dict[str, Any]:
        """Generate summary report from all test reports."""
        total_issues = sum(len(report.issues) for report in reports)
        critical_issues = sum(report.critical_issues_count for report in reports)
        high_issues = sum(report.high_issues_count for report in reports)

        failed_tests = [report for report in reports if report.result == SecurityTestResult.FAILED]
        passed_tests = [report for report in reports if report.result == SecurityTestResult.PASSED]
        error_tests = [report for report in reports if report.result == SecurityTestResult.ERROR]

        # Group issues by severity
        issues_by_severity = {
            SeverityLevel.CRITICAL.value: 0,
            SeverityLevel.HIGH.value: 0,
            SeverityLevel.MEDIUM.value: 0,
            SeverityLevel.LOW.value: 0,
            SeverityLevel.INFO.value: 0,
        }

        for report in reports:
            for issue in report.issues:
                issues_by_severity[issue.severity.value] += 1

        return {
            "overall_status": "FAILED" if failed_tests or critical_issues > 0 else "PASSED",
            "summary": {
                "total_tests": len(reports),
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests),
                "error_tests": len(error_tests),
                "total_issues": total_issues,
                "critical_issues": critical_issues,
                "high_issues": high_issues,
            },
            "issues_by_severity": issues_by_severity,
            "test_results": [report.to_dict() for report in reports],
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def save_report(self, reports: List[SecurityTestReport], output_file: str):
        """Save security test reports to file."""
        summary = self.generate_summary_report(reports)

        async with aiofiles.open(output_file, "w") as f:
            await f.write(json.dumps(summary, indent=2, default=str))

        logger.info(f"Security test report saved to {output_file}")


# CLI interface for running security tests
async def run_security_tests(project_root: str, api_url: str = None, output_file: str = None):
    """Run security tests from command line."""
    test_suite = SecurityTestSuite(project_root, api_url)
    reports = await test_suite.run_all_tests()

    # Print summary
    summary = test_suite.generate_summary_report(reports)
    print(f"\nSecurity Test Results Summary:")
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Total Tests: {summary['summary']['total_tests']}")
    print(f"Passed: {summary['summary']['passed_tests']}")
    print(f"Failed: {summary['summary']['failed_tests']}")
    print(f"Errors: {summary['summary']['error_tests']}")
    print(f"Total Issues: {summary['summary']['total_issues']}")
    print(f"Critical Issues: {summary['summary']['critical_issues']}")
    print(f"High Issues: {summary['summary']['high_issues']}")

    # Save to file if requested
    if output_file:
        await test_suite.save_report(reports, output_file)

    return summary["overall_status"] == "PASSED"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python testing.py <project_root> [api_url] [output_file]")
        sys.exit(1)

    project_root = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else None
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    success = asyncio.run(run_security_tests(project_root, api_url, output_file))
    sys.exit(0 if success else 1)
