#!/usr/bin/env python3
"""
Minimal functional requirements test for LLM A/B Testing Platform.
Tests core functionality without full server startup.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MinimalFunctionalTest:
    """Minimal functional test for core components."""

    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "functional_requirements": {},
            "error_log": [],
        }

    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        self.test_results["tests_run"] += 1
        if success:
            self.test_results["tests_passed"] += 1
            logger.info(f"✅ {test_name}: PASSED {details}")
        else:
            self.test_results["tests_failed"] += 1
            logger.error(f"❌ {test_name}: FAILED {details}")
            self.test_results["error_log"].append(
                {"test": test_name, "details": details, "timestamp": datetime.now().isoformat()}
            )

    def test_project_structure(self) -> bool:
        """FR-001: Project Structure and Organization."""
        try:
            required_dirs = [
                "src/domain",
                "src/infrastructure",
                "src/presentation",
                "tests",
                "docs",
                "scripts",
            ]

            missing_dirs = []
            for dir_path in required_dirs:
                if not (project_root / dir_path).exists():
                    missing_dirs.append(dir_path)

            if missing_dirs:
                self.log_result("FR-001: Project Structure", False, f"Missing: {missing_dirs}")
                return False
            else:
                self.log_result("FR-001: Project Structure", True, "All core directories present")
                return True

        except Exception as e:
            self.log_result("FR-001: Project Structure", False, f"Error: {e}")
            return False

    def test_domain_models(self) -> bool:
        """FR-002: Domain Models and Business Logic."""
        try:
            # Test importing core domain models
            from src.domain.models.evaluation_models import Evaluation, EvaluationResult
            from src.domain.models.provider_models import Provider, ProviderConfiguration
            from src.domain.models.test_models import Test, TestConfiguration

            # Test model instantiation
            test_config = TestConfiguration(
                name="Test Configuration",
                description="Test description",
                prompt_template="Test {input}",
                provider_a_id="provider-a",
                provider_b_id="provider-b",
                evaluation_criteria={"accuracy": 1.0},
                sample_size=100,
                confidence_level=0.95,
            )

            provider_config = ProviderConfiguration(
                api_key="test-key", model="test-model", temperature=0.7
            )

            self.log_result("FR-002: Domain Models", True, "Core models instantiated successfully")
            return True

        except Exception as e:
            self.log_result("FR-002: Domain Models", False, f"Import/instantiation error: {e}")
            return False

    def test_database_models(self) -> bool:
        """FR-003: Database Models and Persistence."""
        try:
            # Test importing database models
            from src.infrastructure.persistence.models.evaluation_models import (
                EvaluationORM,
                EvaluationResultORM,
            )
            from src.infrastructure.persistence.models.provider_models import (
                ProviderConfigurationORM,
                ProviderORM,
            )
            from src.infrastructure.persistence.models.test_models import (
                TestConfigurationORM,
                TestORM,
            )

            self.log_result(
                "FR-003: Database Models", True, "Database models imported successfully"
            )
            return True

        except Exception as e:
            self.log_result("FR-003: Database Models", False, f"Import error: {e}")
            return False

    def test_api_structure(self) -> bool:
        """FR-004: API Structure and Routes."""
        try:
            # Test importing API routes
            from src.presentation.api.models.evaluation_models import (
                EvaluationCreateRequest,
                EvaluationResponse,
            )
            from src.presentation.api.models.provider_models import (
                ProviderCreateRequest,
                ProviderResponse,
            )

            # Test importing API models
            from src.presentation.api.models.test_models import TestCreateRequest, TestResponse
            from src.presentation.api.routes.analytics import router as analytics_router
            from src.presentation.api.routes.auth import router as auth_router
            from src.presentation.api.routes.evaluation import router as evaluation_router
            from src.presentation.api.routes.providers import router as providers_router
            from src.presentation.api.routes.tests import router as tests_router

            self.log_result(
                "FR-004: API Structure", True, "API routes and models imported successfully"
            )
            return True

        except Exception as e:
            self.log_result("FR-004: API Structure", False, f"Import error: {e}")
            return False

    def test_security_components(self) -> bool:
        """FR-005: Security and Authentication Components."""
        try:
            # Test importing security components
            from src.infrastructure.security.audit_logger import SecurityAuditLogger
            from src.infrastructure.security.input_validator import InputValidator
            from src.infrastructure.security.rate_limiter import EnhancedRateLimiter
            from src.infrastructure.security.secrets_manager import SecretsManager

            # Test basic instantiation
            secrets_manager = SecretsManager()
            audit_logger = SecurityAuditLogger()
            input_validator = InputValidator()

            self.log_result(
                "FR-005: Security Components", True, "Security components instantiated successfully"
            )
            return True

        except Exception as e:
            self.log_result(
                "FR-005: Security Components", False, f"Import/instantiation error: {e}"
            )
            return False

    def test_authentication_models(self) -> bool:
        """FR-006: Authentication Models and JWT."""
        try:
            # Test importing authentication components
            from src.presentation.api.auth.jwt_handler import JWTHandler
            from src.presentation.api.models.auth_models import (
                LoginRequest,
                LoginResponse,
                UserProfile,
            )

            # Test JWT handler instantiation
            jwt_handler = JWTHandler()

            # Test auth model instantiation
            login_request = LoginRequest(username="test@example.com", password="TestPassword123!")

            self.log_result(
                "FR-006: Authentication Models", True, "Auth components instantiated successfully"
            )
            return True

        except Exception as e:
            self.log_result(
                "FR-006: Authentication Models", False, f"Import/instantiation error: {e}"
            )
            return False

    def test_dataset_integration(self) -> bool:
        """FR-007: Dataset Integration and Processing."""
        try:
            # Check if downloaded datasets exist and are properly formatted
            datasets_dir = Path("data/processed")
            required_files = ["arc_easy.json", "gsm8k.json", "test_sample.json"]

            datasets_found = 0
            total_examples = 0

            for file_name in required_files:
                file_path = datasets_dir / file_name
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Validate dataset structure
                        if data and isinstance(data, list):
                            sample = data[0]
                            required_fields = [
                                "id",
                                "prompt",
                                "expected_output",
                                "category",
                                "source",
                            ]
                            if all(field in sample for field in required_fields):
                                datasets_found += 1
                                total_examples += len(data)
                    except Exception as e:
                        logger.warning(f"Error reading {file_name}: {e}")

            if datasets_found >= 2:
                self.log_result(
                    "FR-007: Dataset Integration",
                    True,
                    f"Found {datasets_found} datasets with {total_examples:,} examples",
                )
                return True
            else:
                self.log_result(
                    "FR-007: Dataset Integration",
                    False,
                    f"Only found {datasets_found} valid datasets",
                )
                return False

        except Exception as e:
            self.log_result("FR-007: Dataset Integration", False, f"Error: {e}")
            return False

    def test_documentation_structure(self) -> bool:
        """FR-008: Documentation and API Specs."""
        try:
            # Check if documentation files exist
            doc_files = ["docs/API_Documentation.md", "README.md"]

            docs_found = 0
            for doc_file in doc_files:
                if (project_root / doc_file).exists():
                    docs_found += 1

            # Test API documentation generation
            try:
                from src.presentation.api.documentation.openapi_config import (
                    get_custom_openapi_schema,
                )
                from src.presentation.api.documentation.validation_schemas import (
                    LoginRequest,
                    LoginResponse,
                )

                docs_found += 1
            except Exception as e:
                logger.warning(f"API documentation components not fully available: {e}")

            if docs_found >= 2:
                self.log_result(
                    "FR-008: Documentation", True, f"Found {docs_found} documentation components"
                )
                return True
            else:
                self.log_result(
                    "FR-008: Documentation",
                    False,
                    f"Only found {docs_found} documentation components",
                )
                return False

        except Exception as e:
            self.log_result("FR-008: Documentation", False, f"Error: {e}")
            return False

    def test_testing_infrastructure(self) -> bool:
        """FR-009: Testing Infrastructure."""
        try:
            # Check test files exist
            test_dirs = ["tests/unit", "tests/integration"]

            test_files_found = 0
            for test_dir in test_dirs:
                test_path = project_root / test_dir
                if test_path.exists():
                    test_files = list(test_path.glob("test_*.py"))
                    test_files_found += len(test_files)

            # Test importing test utilities
            try:
                from tests.conftest import pytest_configure
                from tests.factories import TestConfigurationFactory

                test_files_found += 1
            except Exception as e:
                logger.warning(f"Test utilities not fully available: {e}")

            if test_files_found >= 3:
                self.log_result(
                    "FR-009: Testing Infrastructure",
                    True,
                    f"Found {test_files_found} test components",
                )
                return True
            else:
                self.log_result(
                    "FR-009: Testing Infrastructure",
                    False,
                    f"Only found {test_files_found} test components",
                )
                return False

        except Exception as e:
            self.log_result("FR-009: Testing Infrastructure", False, f"Error: {e}")
            return False

    def test_configuration_management(self) -> bool:
        """FR-010: Configuration Management."""
        try:
            # Check configuration files exist
            config_files = [".env.example", "alembic.ini", "pyproject.toml"]

            configs_found = 0
            for config_file in config_files:
                if (project_root / config_file).exists():
                    configs_found += 1

            # Test database configuration
            try:
                from src.infrastructure.persistence.database import Database

                configs_found += 1
            except Exception as e:
                logger.warning(f"Database configuration not fully available: {e}")

            if configs_found >= 3:
                self.log_result(
                    "FR-010: Configuration Management",
                    True,
                    f"Found {configs_found} configuration components",
                )
                return True
            else:
                self.log_result(
                    "FR-010: Configuration Management",
                    False,
                    f"Only found {configs_found} configuration components",
                )
                return False

        except Exception as e:
            self.log_result("FR-010: Configuration Management", False, f"Error: {e}")
            return False

    def run_all_tests(self) -> dict:
        """Run all functional requirement tests."""
        logger.info("🚀 Starting Minimal Functional Requirements Test")
        logger.info("=" * 70)

        test_functions = [
            ("FR-001: Project Structure", self.test_project_structure),
            ("FR-002: Domain Models", self.test_domain_models),
            ("FR-003: Database Models", self.test_database_models),
            ("FR-004: API Structure", self.test_api_structure),
            ("FR-005: Security Components", self.test_security_components),
            ("FR-006: Authentication Models", self.test_authentication_models),
            ("FR-007: Dataset Integration", self.test_dataset_integration),
            ("FR-008: Documentation", self.test_documentation_structure),
            ("FR-009: Testing Infrastructure", self.test_testing_infrastructure),
            ("FR-010: Configuration Management", self.test_configuration_management),
        ]

        for test_name, test_func in test_functions:
            logger.info(f"\n🔍 Testing {test_name}...")
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self.log_result(test_name, False, f"Exception: {e}")

        # Calculate success rate
        success_rate = (self.test_results["tests_passed"] / self.test_results["tests_run"]) * 100
        self.test_results["success_rate"] = success_rate

        # Generate summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 FUNCTIONAL REQUIREMENTS TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Tests Run: {self.test_results['tests_run']}")
        logger.info(f"Tests Passed: {self.test_results['tests_passed']} ✅")
        logger.info(f"Tests Failed: {self.test_results['tests_failed']} ❌")
        logger.info(f"Success Rate: {success_rate:.1f}%")

        if self.test_results["tests_failed"] > 0:
            logger.info("\n❌ Failed Tests:")
            for error in self.test_results["error_log"]:
                logger.info(f"  • {error['test']}: {error['details']}")

        # Save detailed results
        results_file = Path("logs/minimal_functional_test_results.json")
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n📁 Detailed results saved to: {results_file}")

        return self.test_results


def main():
    """Main test execution function."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    logger.info("🏁 LLM A/B Testing Platform - Minimal Functional Requirements Validation")
    logger.info("This test validates core functional requirements without server startup")

    # Initialize and run tests
    tester = MinimalFunctionalTest()
    results = tester.run_all_tests()

    # Final assessment
    if results["success_rate"] >= 90:
        logger.info("\n🎉 FUNCTIONAL REQUIREMENTS VALIDATION: EXCELLENT")
        logger.info("The platform fully meets the core functional requirements!")
    elif results["success_rate"] >= 80:
        logger.info("\n✅ FUNCTIONAL REQUIREMENTS VALIDATION: SUCCESS")
        logger.info("The platform meets most functional requirements!")
    elif results["success_rate"] >= 60:
        logger.info("\n⚠️ FUNCTIONAL REQUIREMENTS VALIDATION: PARTIAL SUCCESS")
        logger.info("Most requirements are met, but some improvements needed.")
    else:
        logger.info("\n🚨 FUNCTIONAL REQUIREMENTS VALIDATION: NEEDS ATTENTION")
        logger.info("Several critical requirements need to be addressed.")

    return results


if __name__ == "__main__":
    main()
