#!/usr/bin/env python3
"""
Test script to verify MultiMind SDK import fixes.
"""

import os
import sys

def test_import_with_warnings_disabled():
    """Test import with backend warnings disabled."""
    print("Testing import with MULTIMIND_SHOW_BACKEND_WARNINGS=false...")
    os.environ['MULTIMIND_SHOW_BACKEND_WARNINGS'] = 'false'
    
    try:
        import multimind
        print(f"✅ Successfully imported multimind version {multimind.__version__}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_import_with_warnings_enabled():
    """Test import with backend warnings enabled."""
    print("\nTesting import with MULTIMIND_SHOW_BACKEND_WARNINGS=true...")
    os.environ['MULTIMIND_SHOW_BACKEND_WARNINGS'] = 'true'
    
    try:
        import multimind
        print(f"✅ Successfully imported multimind version {multimind.__version__}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    try:
        import multimind
        
        # Test configuration
        multimind.configure_warnings(show_backend_warnings=False, log_level='WARNING')
        print("✅ Warning configuration works")
        
        # Test core imports
        from multimind.core import MultiMind
        print("✅ Core module imports work")
        
        # Test vector store imports
        from multimind.vector_store import VectorStore
        print("✅ Vector store imports work")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 MultiMind SDK Import Test")
    print("=" * 40)
    
    success_count = 0
    total_tests = 3
    
    if test_import_with_warnings_disabled():
        success_count += 1
    
    if test_import_with_warnings_enabled():
        success_count += 1
    
    if test_basic_functionality():
        success_count += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The import fixes are working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1) 