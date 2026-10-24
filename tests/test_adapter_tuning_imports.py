"""Regression test: adapter_tuning must not reference undefined names (TaskType)."""

import subprocess
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "multimind" / "fine_tuning" / "adapter_tuning.py"


def test_adapter_tuning_has_no_undefined_names():
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821", str(MODULE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_adapter_tuning_task_type_resolves():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from multimind.fine_tuning import adapter_tuning

    assert adapter_tuning.TaskType.CAUSAL_LM is not None
