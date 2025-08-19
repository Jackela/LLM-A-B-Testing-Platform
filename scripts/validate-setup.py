#!/usr/bin/env python3
"""
Validation script for LLM A/B Testing Platform setup.
This script validates database connectivity, environment configuration, and dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Run: poetry install --with dev")
    sys.exit(1)

# Optional imports for validation
PSYCOPG2_AVAILABLE = False
REDIS_AVAILABLE = False

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    print("Note: psycopg2 not available, skipping PostgreSQL validation")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    print("Note: redis not available, skipping Redis validation")


class SetupValidator:
    """Validates the development setup."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_environment(self) -> bool:
        """Validate environment variables."""
        print("[INFO] Validating environment configuration...")
        
        required_vars = ['DATABASE_URL']
        optional_vars = ['REDIS_URL', 'TESTING']
        
        for var in required_vars:
            if not os.getenv(var):
                self.errors.append(f"Missing required environment variable: {var}")
                
        for var in optional_vars:
            if not os.getenv(var):
                self.warnings.append(f"Optional environment variable not set: {var}")
                
        return len(self.errors) == 0
    
    def validate_database_connection(self) -> bool:
        """Validate database connectivity."""
        print("[INFO] Validating database connection...")
        
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            self.errors.append("DATABASE_URL not set")
            return False
            
        try:
            engine = create_engine(database_url, echo=False)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                print(f"[OK] Database connected: {version.split(',')[0]}")
                
                # Test if extensions are available
                try:
                    conn.execute(text("SELECT uuid_generate_v4()"))
                    print("[OK] UUID extension available")
                except Exception:
                    self.warnings.append("UUID extension not available")
                    
                return True
                
        except Exception as e:
            self.errors.append(f"Database connection failed: {e}")
            return False
    
    def validate_redis_connection(self) -> bool:
        """Validate Redis connectivity."""
        print("🔍 Validating Redis connection...")
        
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        try:
            r = redis.from_url(redis_url)
            r.ping()
            info = r.info()
            print(f"✅ Redis connected: version {info['redis_version']}")
            return True
            
        except Exception as e:
            self.errors.append(f"Redis connection failed: {e}")
            return False
    
    def validate_alembic_config(self) -> bool:
        """Validate Alembic configuration."""
        print("🔍 Validating Alembic configuration...")
        
        # Check if alembic.ini exists
        alembic_ini = Path("alembic.ini")
        if not alembic_ini.exists():
            self.errors.append("alembic.ini not found")
            return False
            
        # Check if alembic env.py exists
        alembic_env = Path("alembic/env.py")
        if not alembic_env.exists():
            self.errors.append("alembic/env.py not found")
            return False
            
        try:
            # Try running alembic check
            result = subprocess.run(
                ["poetry", "run", "alembic", "check"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Alembic configuration valid")
                return True
            else:
                # Alembic check might fail if no migrations exist yet, which is OK
                if "No revision files" in result.stderr:
                    print("⚠️ No Alembic revisions found (this is OK for new setup)")
                    return True
                else:
                    self.errors.append(f"Alembic check failed: {result.stderr}")
                    return False
                    
        except subprocess.TimeoutExpired:
            self.errors.append("Alembic check timed out")
            return False
        except Exception as e:
            self.errors.append(f"Failed to run alembic check: {e}")
            return False
    
    def validate_poetry_dependencies(self) -> bool:
        """Validate Poetry dependencies."""
        print("🔍 Validating Poetry dependencies...")
        
        try:
            result = subprocess.run(
                ["poetry", "check"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Poetry dependencies valid")
                return True
            else:
                self.errors.append(f"Poetry check failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.errors.append("Poetry check timed out")
            return False
        except Exception as e:
            self.errors.append(f"Failed to run poetry check: {e}")
            return False
    
    def validate_test_environment(self) -> bool:
        """Validate test environment setup."""
        print("🔍 Validating test environment...")
        
        try:
            # Try running a simple test
            result = subprocess.run(
                ["poetry", "run", "python", "-c", 
                 "import src; print('✅ Source imports working')"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(result.stdout.strip())
                return True
            else:
                self.errors.append(f"Source import test failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.errors.append(f"Test environment validation failed: {e}")
            return False
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🚀 Starting setup validation...\n")
        
        validations = [
            self.validate_environment,
            self.validate_poetry_dependencies,
            self.validate_database_connection,
            self.validate_redis_connection,
            self.validate_alembic_config,
            self.validate_test_environment,
        ]
        
        all_passed = True
        for validation in validations:
            try:
                if not validation():
                    all_passed = False
            except Exception as e:
                self.errors.append(f"Validation error: {e}")
                all_passed = False
            print()
        
        self.print_summary()
        return all_passed and len(self.errors) == 0
    
    def print_summary(self):
        """Print validation summary."""
        print("=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        if not self.errors and not self.warnings:
            print("🎉 All validations passed! Setup is ready.")
            return
        
        if self.errors:
            print(f"❌ {len(self.errors)} error(s) found:")
            for error in self.errors:
                print(f"   • {error}")
            print()
        
        if self.warnings:
            print(f"⚠️ {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"   • {warning}")
            print()
        
        if self.errors:
            print("❌ Setup validation failed. Please fix the errors above.")
        else:
            print("✅ Setup validation passed with warnings.")


def main():
    """Main validation function."""
    # Load environment from .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        print("📁 Loading environment from .env file...")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key, value)
    else:
        print("⚠️ No .env file found. Using system environment variables.")
    
    validator = SetupValidator()
    success = validator.run_validation()
    
    if success:
        print("\n🎯 Next steps:")
        print("   1. Start development services: docker-compose -f docker-compose.dev.yml up -d")
        print("   2. Run migrations: poetry run alembic upgrade head")
        print("   3. Run tests: poetry run pytest tests/unit/ -v")
        print("   4. Start the API: poetry run uvicorn src.presentation.api.main:app --reload")
        sys.exit(0)
    else:
        print("\n❌ Please fix the validation errors before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()