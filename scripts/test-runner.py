#!/usr/bin/env python3
"""
Enhanced test runner with better error handling and debugging capabilities.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    import redis
except ImportError:
    redis = None


class TestRunner:
    """Enhanced test runner with comprehensive error handling."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.setup_logging()
        self.project_root = Path(__file__).parent.parent
        self.test_results: Dict[str, any] = {}
        
    def setup_logging(self):
        """Set up logging configuration."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('test-runner.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_environment(self) -> bool:
        """Check if the environment is properly set up."""
        self.logger.info("🔍 Checking environment setup...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 11:
            self.logger.error(f"❌ Python 3.11+ required, found {python_version.major}.{python_version.minor}")
            return False
        
        # Check Poetry installation
        try:
            result = subprocess.run(['poetry', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("❌ Poetry not found or not working")
                return False
            self.logger.info(f"✅ Poetry version: {result.stdout.strip()}")
        except FileNotFoundError:
            self.logger.error("❌ Poetry not installed")
            return False
        
        # Check if we're in a Poetry environment
        try:
            result = subprocess.run(['poetry', 'env', 'info'], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning("⚠️ Not in a Poetry virtual environment")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not check Poetry environment: {e}")
        
        self.logger.info("✅ Environment check completed")
        return True
    
    def check_services(self) -> Tuple[bool, bool]:
        """Check if required services (PostgreSQL, Redis) are available."""
        self.logger.info("🔧 Checking service availability...")
        
        postgres_ok = self._check_postgres()
        redis_ok = self._check_redis()
        
        if postgres_ok and redis_ok:
            self.logger.info("✅ All services are available")
        else:
            self.logger.error("❌ Some services are not available")
        
        return postgres_ok, redis_ok
    
    def _check_postgres(self) -> bool:
        """Check PostgreSQL connectivity."""
        if psycopg2 is None:
            self.logger.warning("⚠️ psycopg2 not available, skipping PostgreSQL check")
            return False
            
        try:
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                user='postgres',
                password='postgres',
                database='test_db',
                connect_timeout=10
            )
            conn.close()
            self.logger.info("✅ PostgreSQL connection successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ PostgreSQL connection failed: {e}")
            self.logger.error("💡 Make sure PostgreSQL is running with correct credentials")
            return False
    
    def _check_redis(self) -> bool:
        """Check Redis connectivity."""
        if redis is None:
            self.logger.warning("⚠️ redis module not available, skipping Redis check")
            return False
            
        try:
            r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=10)
            r.ping()
            self.logger.info("✅ Redis connection successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ Redis connection failed: {e}")
            self.logger.error("💡 Make sure Redis is running on default port 6379")
            return False
    
    def setup_database(self) -> bool:
        """Set up the test database."""
        self.logger.info("🔧 Setting up test database...")
        
        try:
            # Run Alembic migrations
            result = subprocess.run(
                ['poetry', 'run', 'alembic', 'upgrade', 'head'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                self.logger.error(f"❌ Database migration failed: {result.stderr}")
                return False
            
            self.logger.info("✅ Database migration completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Database migration timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Database setup failed: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install project dependencies."""
        self.logger.info("📦 Installing dependencies...")
        
        try:
            result = subprocess.run(
                ['poetry', 'install', '--with', 'dev'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                self.logger.error(f"❌ Dependency installation failed: {result.stderr}")
                return False
            
            self.logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Dependency installation timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Dependency installation failed: {e}")
            return False
    
    def run_tests(self, test_type: str = "all", coverage: bool = True) -> bool:
        """Run tests with enhanced error handling."""
        self.logger.info(f"🧪 Running {test_type} tests...")
        
        test_commands = {
            "unit": ["poetry", "run", "pytest", "tests/unit/", "-v", "--tb=short"],
            "integration": ["poetry", "run", "pytest", "tests/integration/", "-v", "--tb=short", "-m", "integration"],
            "e2e": ["poetry", "run", "pytest", "tests/e2e/", "-v", "--tb=short", "-m", "e2e", "--timeout=300"],
            "all": ["poetry", "run", "pytest", "tests/", "-v", "--tb=short"]
        }
        
        if test_type not in test_commands:
            self.logger.error(f"❌ Unknown test type: {test_type}")
            return False
        
        cmd = test_commands[test_type]
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=xml",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-fail-under=80"
            ])
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=not self.verbose,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            end_time = time.time()
            
            duration = end_time - start_time
            self.test_results[test_type] = {
                "success": result.returncode == 0,
                "duration": duration,
                "returncode": result.returncode
            }
            
            if result.returncode == 0:
                self.logger.info(f"✅ {test_type} tests passed in {duration:.2f}s")
                return True
            else:
                self.logger.error(f"❌ {test_type} tests failed (exit code: {result.returncode})")
                if result.stderr:
                    self.logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ {test_type} tests timed out after 30 minutes")
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to run {test_type} tests: {e}")
            return False
    
    def run_quality_checks(self) -> bool:
        """Run code quality checks."""
        self.logger.info("🔍 Running code quality checks...")
        
        checks = [
            (["poetry", "run", "black", "--check", "src", "tests"], "Code formatting"),
            (["poetry", "run", "isort", "--check-only", "src", "tests"], "Import sorting"),
            (["poetry", "run", "flake8", "src", "tests"], "Linting"),
            (["poetry", "run", "mypy", "src"], "Type checking")
        ]
        
        all_passed = True
        
        for cmd, description in checks:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    self.logger.info(f"✅ {description} passed")
                else:
                    self.logger.error(f"❌ {description} failed")
                    if result.stdout:
                        self.logger.error(f"Output: {result.stdout}")
                    if result.stderr:
                        self.logger.error(f"Error: {result.stderr}")
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"❌ {description} timed out")
                all_passed = False
            except Exception as e:
                self.logger.error(f"❌ {description} failed with exception: {e}")
                all_passed = False
        
        return all_passed
    
    def run_security_scan(self) -> bool:
        """Run security scans."""
        self.logger.info("🔒 Running security scans...")
        
        scans = [
            (["poetry", "run", "bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"], "Bandit security scan"),
            (["poetry", "run", "safety", "check", "--json", "--output", "safety-report.json"], "Safety dependency check")
        ]
        
        all_passed = True
        
        for cmd, description in scans:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                # Security tools might return non-zero on findings, so we check for critical issues
                if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found but not critical
                    self.logger.info(f"✅ {description} completed")
                else:
                    self.logger.error(f"❌ {description} failed with critical issues")
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"❌ {description} timed out")
                all_passed = False
            except Exception as e:
                self.logger.error(f"❌ {description} failed with exception: {e}")
                all_passed = False
        
        return all_passed
    
    def generate_report(self) -> None:
        """Generate a comprehensive test report."""
        self.logger.info("📊 Generating test report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for r in self.test_results.values() if r.get("success", False)),
                "total_duration": sum(r.get("duration", 0) for r in self.test_results.values())
            }
        }
        
        # Save JSON report
        with open("test-report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("📊 TEST SUMMARY")
        print("="*50)
        print(f"⏰ Timestamp: {report['timestamp']}")
        print(f"✅ Passed: {report['summary']['passed']}/{report['summary']['total_tests']}")
        print(f"⏱️ Total Duration: {report['summary']['total_duration']:.2f}s")
        
        for test_type, result in self.test_results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            duration = result.get("duration", 0)
            print(f"  {test_type}: {status} ({duration:.2f}s)")
        
        print("="*50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced test runner")
    parser.add_argument("--test-type", choices=["unit", "integration", "e2e", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality checks")
    parser.add_argument("--skip-security", action="store_true", help="Skip security scans")
    parser.add_argument("--skip-services", action="store_true", help="Skip service checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    
    success = True
    
    # Environment check
    if not runner.check_environment():
        sys.exit(1)
    
    # Install dependencies
    if not runner.install_dependencies():
        sys.exit(1)
    
    # Service checks
    if not args.skip_services:
        postgres_ok, redis_ok = runner.check_services()
        if not (postgres_ok and redis_ok):
            print("⚠️ Some services are not available. Tests may fail.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        # Database setup
        if postgres_ok and not runner.setup_database():
            sys.exit(1)
    
    # Quality checks
    if not args.skip_quality:
        if not runner.run_quality_checks():
            success = False
    
    # Security scans
    if not args.skip_security:
        if not runner.run_security_scan():
            success = False
    
    # Run tests
    if not runner.run_tests(args.test_type, not args.no_coverage):
        success = False
    
    # Generate report
    runner.generate_report()
    
    if success:
        print("\n🎉 All checks passed!")
        sys.exit(0)
    else:
        print("\n💥 Some checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()