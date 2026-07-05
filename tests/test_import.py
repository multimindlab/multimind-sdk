#!/usr/bin/env python3
"""Test script to verify MultiMind SDK import fixes.

These tests use ``assert`` (not ``return True``) so they pass under pytest 8+
which warns and will eventually error on tests that return non-None.
"""

import os
import sys


def test_import_with_warnings_disabled(monkeypatch):
    """Test import with backend warnings disabled."""
    monkeypatch.setenv("MULTIMIND_SHOW_BACKEND_WARNINGS", "false")
    import multimind

    assert multimind.__version__, "multimind.__version__ should be a non-empty string"


def test_import_with_warnings_enabled(monkeypatch):
    """Test import with backend warnings enabled."""
    monkeypatch.setenv("MULTIMIND_SHOW_BACKEND_WARNINGS", "true")
    import multimind

    assert multimind.__version__, "multimind.__version__ should be a non-empty string"


def test_basic_functionality():
    """Test basic functionality: configure_warnings + core/vector-store imports."""
    import multimind

    multimind.configure_warnings(show_backend_warnings=False, log_level="WARNING")

    from multimind.core import MultiMind  # noqa: F401
    from multimind.vector_store import VectorStore  # noqa: F401

    assert MultiMind is not None
    assert VectorStore is not None


if __name__ == "__main__":
    print("🧪 MultiMind SDK Import Test")
    print("=" * 40)

    os.environ["MULTIMIND_SHOW_BACKEND_WARNINGS"] = "false"
    try:
        test_import_with_warnings_disabled.__wrapped__ if hasattr(
            test_import_with_warnings_disabled, "__wrapped__"
        ) else None
        import multimind  # noqa: F401

        print("✅ Import OK")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)
    print("🎉 Manual smoke check passed.")
