#!/usr/bin/env python3
"""
Simple test script for Alembic configuration.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_alembic_import():
    """Test if we can import the Alembic environment."""
    print("[INFO] Testing Alembic configuration...")
    
    try:
        # Test if we can import the alembic env
        sys.path.insert(0, str(Path("alembic")))
        import env
        print("[OK] Alembic env.py imports successfully")
        
        # Test database URL resolution
        database_url = env.get_database_url()
        print(f"[OK] Database URL resolved: {database_url.split('://')[0]}://***")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Alembic test failed: {e}")
        return False

def test_database_imports():
    """Test database-related imports."""
    print("[INFO] Testing database imports...")
    
    try:
        from src.infrastructure.persistence.database import Base
        print("[OK] Database Base import successful")
        
        from src.infrastructure.persistence.models import test_models
        print("[OK] Test models import successful")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Database imports failed: {e}")
        return False

def main():
    """Run simple validation tests."""
    print("Starting basic validation tests...\n")
    
    # Set environment variable if not set
    if not os.getenv('DATABASE_URL'):
        os.environ['DATABASE_URL'] = 'postgresql://postgres:postgres@localhost:5432/test_db'
        print("[INFO] Set default DATABASE_URL for testing")
    
    tests = [
        test_database_imports,
        test_alembic_import,
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"[ERROR] Test exception: {e}")
            all_passed = False
        print()
    
    if all_passed:
        print("[SUCCESS] All basic tests passed!")
        print("\nNext steps:")
        print("1. Start PostgreSQL: docker-compose -f docker-compose.dev.yml up -d postgres")
        print("2. Run migration: poetry run alembic upgrade head")
        print("3. Test CI workflow: git push to trigger GitHub Actions")
    else:
        print("[FAILED] Some tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()