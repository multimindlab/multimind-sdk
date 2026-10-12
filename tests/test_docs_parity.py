"""Every CLI command and platform route must be mentioned in docs/."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_docs_parity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_docs_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_commands_and_routes_are_documented():
    parity = _load_script()
    assert parity.cli_commands(), "no CLI commands found"
    assert parity.routes(), "no platform routes found"
    gaps = parity.check()
    assert not gaps, "undocumented:\n" + "\n".join(gaps)
