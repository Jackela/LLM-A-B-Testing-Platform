"""Real lock-audit boundary: tool and malformed-output failures stay failures."""

import json
import subprocess
import unittest
from unittest.mock import patch

from src.infrastructure.security.scanner_runtime import (
    audit_locked_runtime,
    parse_audit_result,
    parse_bandit_result,
)


class DependencyAudit(unittest.TestCase):
    def test_bandit_clean_and_vulnerable_reports(self):
        for issues, code in [([], 0), ([{"test_id": "B999", "issue_severity": "HIGH"}], 1)]:
            report = {"results": issues, "errors": [], "metrics": {"_totals": {"loc": 1}}}
            self.assertEqual(parse_bandit_result(json.dumps(report), code), report)

    def test_bandit_errors_cannot_be_reported_as_a_clean_scan(self):
        clean = {"results": [], "errors": [], "metrics": {"_totals": {"loc": 1}}}
        cases = [
            ("", 0),
            (json.dumps(clean), 2),
            (json.dumps(clean), 1),
            (json.dumps({**clean, "errors": ["cannot parse source"]}), 0),
            (json.dumps({**clean, "metrics": {}}), 0),
            (json.dumps({**clean, "results": [{}]}), 1),
        ]
        for stdout, code in cases:
            with self.subTest(stdout=stdout, code=code), self.assertRaises(ValueError):
                parse_bandit_result(stdout, code)

    def report(self, vulnerabilities=None, **extra):
        return json.dumps(
            {
                "dependencies": [
                    {"name": "fixture", "version": "1", "vulns": vulnerabilities or [], **extra}
                ]
            }
        )

    def test_clean_and_vulnerable_reports(self):
        self.assertEqual(parse_audit_result(self.report(), 0), [])
        vulnerability = {"id": "GHSA-fixture", "description": "fixture", "fix_versions": ["2"]}
        self.assertEqual(
            parse_audit_result(self.report([vulnerability]), 1),
            [{**vulnerability, "package_name": "fixture"}],
        )

    def test_errors_and_inconsistent_reports_are_rejected(self):
        cases = [
            ("", 0),
            ("not json", 1),
            ("[]", 0),
            ('{"dependencies": []}', 0),
            (self.report(), 2),
            (self.report(), 1),
            (self.report([{"id": "fixture"}]), 0),
            (self.report(skip_reason="unsupported"), 0),
            (self.report([{}]), 1),
            (self.report([{"id": "fixture", "fix_versions": "2"}]), 1),
        ]
        for stdout, code in cases:
            with self.subTest(stdout=stdout, code=code), self.assertRaises(ValueError):
                parse_audit_result(stdout, code)

    def test_missing_exporter_or_scanner_cannot_report_success(self):
        with patch(
            "src.infrastructure.security.scanner_runtime.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["export"]),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                audit_locked_runtime(".")
        with patch(
            "src.infrastructure.security.scanner_runtime.subprocess.run",
            side_effect=[
                subprocess.CompletedProcess([], 0, "fixture==1\n", ""),
                subprocess.CompletedProcess([], 2, "", "missing tool"),
            ],
        ):
            with self.assertRaises(ValueError):
                audit_locked_runtime(".")

    def test_exported_lock_is_passed_to_scanner(self):
        def tool(argv, **kwargs):
            if "pip_audit" not in argv:
                self.assertTrue(kwargs["check"])
                return subprocess.CompletedProcess(argv, 0, "fixture==1\n", "")
            from pathlib import Path

            self.assertEqual(Path(argv[argv.index("-r") + 1]).read_text(), "fixture==1\n")
            self.assertIn("--no-deps", argv)
            self.assertNotIn("shell", kwargs)
            return subprocess.CompletedProcess(argv, 0, self.report(), "")

        with patch("src.infrastructure.security.scanner_runtime.subprocess.run", side_effect=tool):
            self.assertEqual(audit_locked_runtime("."), [])
