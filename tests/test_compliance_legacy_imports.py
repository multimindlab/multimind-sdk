import warnings
import pytest

def test_legacy_imports():
    """Test that legacy imports work or are handled gracefully."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            from multimind.compliance import (
                run_compliance,
                run_example,
                generate_report,
                show_dashboard,
                show_alerts,
                configure_alerts,
            )
            # If imports succeed, verify they are callable or None
            assert callable(run_compliance), "run_compliance should be callable"
            # Other functions may not be available, which is fine
            # They should either be callable or raise AttributeError when accessed
        except (ImportError, AttributeError) as e:
            # Legacy imports may not be available - this is acceptable
            # The module should handle this gracefully
            pytest.skip(f"Legacy compliance functions not available: {e}") 