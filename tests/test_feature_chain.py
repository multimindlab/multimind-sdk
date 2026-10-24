"""Every registered feature must be reachable component -> API -> dashboard -> docs."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_feature_chain.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_feature_chain", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_feature_chain"] = module
    spec.loader.exec_module(module)
    return module


def test_feature_chain_has_no_gaps():
    result = _load().check()
    assert result["features"], "feature registry is empty"
    assert result["gaps"] == [], "\n".join(result["gaps"])


def test_feature_chain_reports_component_state():
    result = _load().check()
    states = {row["name"]: row["component"] for row in result["features"]}
    assert states["chat"] == "available"
    assert all(s == "available" or s.startswith("unavailable: ") for s in states.values())
