"""Audit the canonical runtime lock; scanner failures never mean a clean report."""

import json
import subprocess  # nosec B404 # Fixed local scanner commands without a shell.
import sys
import tempfile
from pathlib import Path
from typing import Any


def parse_audit_result(stdout: str, returncode: int) -> list[dict[str, Any]]:
    if returncode not in (0, 1):
        raise ValueError(f"pip-audit failed with exit code {returncode}")
    data = json.loads(stdout)
    dependencies = data.get("dependencies") if isinstance(data, dict) else None
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("pip-audit did not report any audited dependencies")
    issues = []
    for dependency in dependencies:
        if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
            raise ValueError("Malformed audited dependency")
        vulnerabilities = dependency.get("vulns")
        if not isinstance(vulnerabilities, list) or dependency.get("skip_reason"):
            raise ValueError(f"Dependency was not audited: {dependency['name']}")
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, dict) or not vulnerability.get("id"):
                raise ValueError("Malformed vulnerability record")
            fixes = vulnerability.get("fix_versions", [])
            if not isinstance(fixes, list) or not all(isinstance(item, str) for item in fixes):
                raise ValueError("Malformed fixed versions")
            issues.append({**vulnerability, "package_name": dependency["name"]})
    if bool(issues) != (returncode == 1):
        raise ValueError("pip-audit exit status contradicts its vulnerability report")
    return issues


def audit_locked_runtime(project_root: str) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    export = (
        subprocess.run(  # nosec B603 # Known Python and repository-owned export script; no shell.
            [sys.executable, str(root / "tools/export_locked_requirements.py")],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
    )
    if not export.stdout.strip():
        raise ValueError("Runtime lock export was empty")
    with tempfile.TemporaryDirectory(prefix="llm-dependency-audit-") as temporary:
        requirements = Path(temporary) / "requirements.txt"
        requirements.write_text(export.stdout, encoding="utf-8")
        result = subprocess.run(  # nosec B603 # Fixed local scanner argv and generated lock file; no shell.
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--disable-pip",
                "--no-deps",
                "-r",
                str(requirements),
                "--format",
                "json",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=300,
        )
    return parse_audit_result(result.stdout, result.returncode)
