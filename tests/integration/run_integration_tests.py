"""Comprehensive integration test runner for LLM A/B Testing Platform."""

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Orchestrates and runs all integration tests."""

    def __init__(self):
        self.test_directory = Path(__file__).parent
        self.project_root = self.test_directory.parent.parent
        self.results: Dict[str, Dict] = {}

    def run_test_suite(self, test_file: str, description: str) -> Dict:
        """Run a specific test suite and return results."""
        logger.info(f"🧪 Running {description}...")

        start_time = time.time()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(self.test_directory / test_file),
                    "-v",
                    "--tb=short",
                    "--no-header",
                ],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )

            execution_time = time.time() - start_time

            # Parse pytest output
            stdout_lines = result.stdout.split("\n")
            stderr_lines = result.stderr.split("\n")

            # Count test results
            passed_count = len([line for line in stdout_lines if "PASSED" in line])
            failed_count = len([line for line in stdout_lines if "FAILED" in line])
            error_count = len([line for line in stdout_lines if "ERROR" in line])
            skipped_count = len([line for line in stdout_lines if "SKIPPED" in line])

            # Extract summary line
            summary_line = ""
            for line in stdout_lines:
                if "passed" in line and ("failed" in line or "error" in line or "skipped" in line):
                    summary_line = line.strip()
                    break

            success = result.returncode == 0

            test_result = {
                "success": success,
                "return_code": result.returncode,
                "execution_time": execution_time,
                "passed": passed_count,
                "failed": failed_count,
                "errors": error_count,
                "skipped": skipped_count,
                "summary": summary_line,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            if success:
                logger.info(f"✅ {description} completed successfully ({execution_time:.2f}s)")
                logger.info(
                    f"   📊 {passed_count} passed, {failed_count} failed, {error_count} errors"
                )
            else:
                logger.error(f"❌ {description} failed ({execution_time:.2f}s)")
                logger.error(
                    f"   📊 {passed_count} passed, {failed_count} failed, {error_count} errors"
                )

                # Log first few error lines
                error_lines = [line for line in stderr_lines if line.strip()]
                if error_lines:
                    logger.error(f"   🔍 First error: {error_lines[0][:100]}...")

            return test_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {description} crashed: {str(e)}")

            return {
                "success": False,
                "return_code": -1,
                "execution_time": execution_time,
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "skipped": 0,
                "summary": f"Test suite crashed: {str(e)}",
                "stdout": "",
                "stderr": str(e),
            }

    def run_simple_component_tests(self) -> Dict:
        """Run simple component tests first."""
        logger.info("🔧 Running simple component validation...")

        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, "test_security_simple.py"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )

            execution_time = time.time() - start_time

            # Parse output for test results
            stdout = result.stdout
            success_indicators = [
                "All security component tests passed",
                "Security Components Validated",
                "tests passed",
            ]

            success = any(indicator in stdout for indicator in success_indicators)

            return {
                "success": success,
                "return_code": result.returncode,
                "execution_time": execution_time,
                "summary": "Component validation completed",
                "stdout": stdout,
                "stderr": result.stderr,
            }

        except Exception as e:
            return {
                "success": False,
                "return_code": -1,
                "execution_time": time.time() - start_time,
                "summary": f"Component tests crashed: {str(e)}",
                "stdout": "",
                "stderr": str(e),
            }

    def check_dependencies(self) -> Dict:
        """Check that all required dependencies are installed."""
        logger.info("📦 Checking dependencies...")

        required_packages = [
            "fastapi",
            "sqlalchemy",
            "alembic",
            "redis",
            "pytest",
            "httpx",
            "passlib",
            "bleach",
            "pyotp",
            "slowapi",
        ]

        missing_packages = []

        for package in required_packages:
            try:
                result = subprocess.run(
                    [sys.executable, "-c", f"import {package}"], capture_output=True, text=True
                )

                if result.returncode != 0:
                    missing_packages.append(package)
            except Exception:
                missing_packages.append(package)

        if missing_packages:
            logger.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
            return {
                "success": False,
                "missing_packages": missing_packages,
                "message": f"Missing required packages: {', '.join(missing_packages)}",
            }
        else:
            logger.info("✅ All dependencies available")
            return {
                "success": True,
                "missing_packages": [],
                "message": "All dependencies satisfied",
            }

    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        total_tests = sum(
            result.get("passed", 0) + result.get("failed", 0) + result.get("errors", 0)
            for result in self.results.values()
            if isinstance(result, dict)
        )
        total_passed = sum(
            result.get("passed", 0) for result in self.results.values() if isinstance(result, dict)
        )
        total_failed = sum(
            result.get("failed", 0) for result in self.results.values() if isinstance(result, dict)
        )
        total_errors = sum(
            result.get("errors", 0) for result in self.results.values() if isinstance(result, dict)
        )

        overall_success = all(result.get("success", False) for result in self.results.values())
        total_time = sum(
            result.get("execution_time", 0)
            for result in self.results.values()
            if isinstance(result, dict)
        )

        report = f"""
{'='*80}
🧪 LLM A/B Testing Platform - Integration Test Report
{'='*80}

📊 OVERALL SUMMARY:
   Status: {'✅ PASSED' if overall_success else '❌ FAILED'}
   Total Tests: {total_tests}
   Passed: {total_passed}
   Failed: {total_failed}
   Errors: {total_errors}
   Success Rate: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%
   Total Execution Time: {total_time:.2f} seconds

📋 TEST SUITE RESULTS:
"""

        for suite_name, result in self.results.items():
            if isinstance(result, dict):
                status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                execution_time = result.get("execution_time", 0)
                passed = result.get("passed", 0)
                failed = result.get("failed", 0)
                errors = result.get("errors", 0)

                report += f"""
   {suite_name}:
     Status: {status}
     Time: {execution_time:.2f}s
     Results: {passed} passed, {failed} failed, {errors} errors
"""

                if not result.get("success", False) and result.get("stderr"):
                    # Show first error for failed suites
                    error_lines = result["stderr"].split("\n")[:3]
                    report += f"     Error: {' '.join(error_lines).strip()[:100]}...\n"

        report += f"""
{'='*80}
🔧 SYSTEM VALIDATION:
"""

        # Add dependency check results
        if "dependencies" in self.results:
            dep_result = self.results["dependencies"]
            if dep_result.get("success", False):
                report += "   ✅ All dependencies satisfied\n"
            else:
                missing = dep_result.get("missing_packages", [])
                report += f"   ❌ Missing dependencies: {', '.join(missing)}\n"

        # Add component validation results
        if "components" in self.results:
            comp_result = self.results["components"]
            if comp_result.get("success", False):
                report += "   ✅ All security components functional\n"
            else:
                report += "   ⚠️ Some security components have issues\n"

        report += f"""
{'='*80}
📈 RECOMMENDATIONS:
"""

        if overall_success:
            report += """   🎉 All tests passed! The system is ready for deployment.
   
   Next steps:
   • Run performance benchmarks
   • Deploy to staging environment
   • Conduct user acceptance testing
   • Prepare production deployment
"""
        else:
            report += """   🔧 Some tests failed. Review and fix issues before deployment.
   
   Priority actions:
   • Review failed test output above
   • Fix critical security or database issues first
   • Re-run tests after fixes
   • Consider staging environment testing
"""

        report += f"\n{'='*80}\n"

        return report

    def run_all_tests(self) -> bool:
        """Run all integration tests in sequence."""
        logger.info("🚀 Starting comprehensive integration test suite...")

        # Check dependencies first
        self.results["dependencies"] = self.check_dependencies()

        if not self.results["dependencies"]["success"]:
            logger.error("❌ Dependencies check failed. Installing missing packages...")
            # Try to install missing packages
            missing = self.results["dependencies"]["missing_packages"]
            for package in missing:
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True,
                    )
                except Exception as e:
                    logger.error(f"Failed to install {package}: {e}")

        # Run simple component tests first
        self.results["components"] = self.run_simple_component_tests()

        # Define integration test suites
        test_suites = [
            ("test_api_integration.py", "API Integration Tests"),
            ("test_database_integration.py", "Database Integration Tests"),
            ("test_security_integration.py", "Security Integration Tests"),
        ]

        # Run each test suite
        for test_file, description in test_suites:
            self.results[description] = self.run_test_suite(test_file, description)

        # Generate and display report
        report = self.generate_report()
        print(report)

        # Save report to file
        report_file = self.project_root / "integration_test_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"📄 Detailed report saved to: {report_file}")

        # Return overall success
        return all(result.get("success", False) for result in self.results.values())


def main():
    """Main entry point for integration tests."""
    runner = IntegrationTestRunner()

    try:
        success = runner.run_all_tests()

        if success:
            logger.info("🎉 All integration tests completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Some integration tests failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⏹️ Integration tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Integration test runner crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
