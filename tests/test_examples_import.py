"""Every ``examples/...`` path cited in docs/feature-status.md must exist, and every
Python example cited there must compile and (when it has a ``__main__`` guard)
import without side effects.
"""

import importlib
import py_compile
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_STATUS = REPO_ROOT / "docs" / "feature-status.md"

_PATH_RE = re.compile(r"(?<![A-Za-z0-9_/])(examples/[A-Za-z0-9_./-]+)")


def _cited_paths():
    text = FEATURE_STATUS.read_text(encoding="utf-8")
    seen = []
    for match in _PATH_RE.finditer(text):
        path = match.group(1).rstrip(".")
        if path not in seen:
            seen.append(path)
    return seen


CITED_PATHS = _cited_paths()
CITED_PY_FILES = [p for p in CITED_PATHS if p.endswith(".py")]


def test_feature_status_cites_examples():
    assert CITED_PATHS, "docs/feature-status.md cites no examples/ paths"


@pytest.mark.parametrize("rel_path", CITED_PATHS)
def test_cited_example_path_exists(rel_path):
    assert (REPO_ROOT / rel_path).exists(), f"{rel_path} is cited in feature-status.md but missing"


@pytest.mark.parametrize("rel_path", CITED_PY_FILES)
def test_cited_example_compiles(rel_path):
    path = REPO_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} missing (covered by the existence test)")
    py_compile.compile(str(path), doraise=True)


def _has_main_guard(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    return 'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source


@pytest.mark.parametrize("rel_path", CITED_PY_FILES)
def test_cited_example_imports_without_side_effects(rel_path, monkeypatch):
    path = REPO_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} missing (covered by the existence test)")
    if not _has_main_guard(path):
        pytest.skip(f"{rel_path} has no __main__ guard; importing would run it")

    module_name = rel_path[: -len(".py")].replace("/", ".")
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop(module_name, None)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(f"{rel_path} needs an optional dependency: {exc}")
    assert module.__file__ and Path(module.__file__).resolve() == path.resolve()
