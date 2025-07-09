import warnings
import pytest

def test_legacy_imports():
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
        except ImportError:
            # Should not raise ImportError, only warn
            pytest.fail("Legacy import raised ImportError instead of warning.")
        # If modules are missing, a DeprecationWarning should be present
        assert any(
            issubclass(warn.category, DeprecationWarning) for warn in w
        ) or all(callable(obj) for obj in [
            run_compliance,
            run_example,
            generate_report,
            show_dashboard,
            show_alerts,
            configure_alerts,
        ]), "DeprecationWarning not raised and legacy functions not present." 