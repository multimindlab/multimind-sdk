"""Shadow-AI inventory scanner: static discovery of AI usage in a project.

``scan_project`` walks a directory tree and reports AI SDK dependencies
(from requirements/pyproject/package.json/Pipfile manifests), AI API base
URLs and env-var reads in source files, and hardcoded AI API keys (reusing
the :class:`multimind.compliance.guard.PIIDetector` patterns). Findings
record the key TYPE and location only — never a key or env-var value.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..compliance.guard import PIIDetector

# package -> {provider, kind (sdk/framework), data_flow (external/local)}
AI_PACKAGE_REGISTRY: Dict[str, Dict[str, str]] = {
    "openai": {"provider": "openai", "kind": "sdk", "data_flow": "external"},
    "anthropic": {"provider": "anthropic", "kind": "sdk", "data_flow": "external"},
    "google-generativeai": {"provider": "google", "kind": "sdk", "data_flow": "external"},
    "google-genai": {"provider": "google", "kind": "sdk", "data_flow": "external"},
    "google-cloud-aiplatform": {"provider": "google", "kind": "sdk", "data_flow": "external"},
    "mistralai": {"provider": "mistral", "kind": "sdk", "data_flow": "external"},
    "groq": {"provider": "groq", "kind": "sdk", "data_flow": "external"},
    "cohere": {"provider": "cohere", "kind": "sdk", "data_flow": "external"},
    "litellm": {"provider": "multi", "kind": "framework", "data_flow": "external"},
    "crewai": {"provider": "multi", "kind": "framework", "data_flow": "external"},
    "autogen": {"provider": "multi", "kind": "framework", "data_flow": "external"},
    "pyautogen": {"provider": "multi", "kind": "framework", "data_flow": "external"},
    "ag2": {"provider": "multi", "kind": "framework", "data_flow": "external"},
    "transformers": {"provider": "huggingface", "kind": "sdk", "data_flow": "local"},
    "sentence-transformers": {"provider": "huggingface", "kind": "sdk", "data_flow": "local"},
    "huggingface-hub": {"provider": "huggingface", "kind": "sdk", "data_flow": "external"},
    "ollama": {"provider": "ollama", "kind": "sdk", "data_flow": "local"},
    "replicate": {"provider": "replicate", "kind": "sdk", "data_flow": "external"},
    "together": {"provider": "together", "kind": "sdk", "data_flow": "external"},
    "openrouter": {"provider": "openrouter", "kind": "sdk", "data_flow": "external"},
    # JS / npm package names (package.json)
    "@anthropic-ai/sdk": {"provider": "anthropic", "kind": "sdk", "data_flow": "external"},
    "@google/generative-ai": {"provider": "google", "kind": "sdk", "data_flow": "external"},
    "@google/genai": {"provider": "google", "kind": "sdk", "data_flow": "external"},
    "@mistralai/mistralai": {"provider": "mistral", "kind": "sdk", "data_flow": "external"},
    "cohere-ai": {"provider": "cohere", "kind": "sdk", "data_flow": "external"},
    "groq-sdk": {"provider": "groq", "kind": "sdk", "data_flow": "external"},
    "llamaindex": {"provider": "multi", "kind": "framework", "data_flow": "external"},
    "together-ai": {"provider": "together", "kind": "sdk", "data_flow": "external"},
    "ai": {"provider": "multi", "kind": "framework", "data_flow": "external"},
}

# Prefix families checked after exact registry lookup.
_REGISTRY_PREFIXES: List[Tuple[str, Dict[str, str]]] = [
    ("langchain", {"provider": "multi", "kind": "framework", "data_flow": "external"}),
    ("llama-index", {"provider": "multi", "kind": "framework", "data_flow": "external"}),
    ("azure-ai-", {"provider": "azure", "kind": "sdk", "data_flow": "external"}),
    ("@langchain/", {"provider": "multi", "kind": "framework", "data_flow": "external"}),
    ("@ai-sdk/", {"provider": "multi", "kind": "framework", "data_flow": "external"}),
]

# Known AI credential env-var names (names only; values are never read
# into findings).
AI_ENV_KEYS: Dict[str, str] = {
    "OPENAI_API_KEY": "openai",
    "AZURE_OPENAI_API_KEY": "azure",
    "ANTHROPIC_API_KEY": "anthropic",
    "GOOGLE_API_KEY": "google",
    "GEMINI_API_KEY": "google",
    "MISTRAL_API_KEY": "mistral",
    "GROQ_API_KEY": "groq",
    "COHERE_API_KEY": "cohere",
    "DEEPSEEK_API_KEY": "deepseek",
    "TOGETHER_API_KEY": "together",
    "OPENROUTER_API_KEY": "openrouter",
    "REPLICATE_API_TOKEN": "replicate",
    "HUGGINGFACE_API_KEY": "huggingface",
    "HUGGINGFACEHUB_API_TOKEN": "huggingface",
    "HF_TOKEN": "huggingface",
}

_ENV_KEY_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in AI_ENV_KEYS) + r")\b")

_URL_PROVIDERS: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"api\.openai\.com"), "openai"),
    (re.compile(r"api\.anthropic\.com"), "anthropic"),
    (re.compile(r"generativelanguage\.googleapis\.com"), "google"),
    (re.compile(r"api\.mistral\.ai"), "mistral"),
    (re.compile(r"api\.groq\.com"), "groq"),
    (re.compile(r"api\.deepseek\.com"), "deepseek"),
    (re.compile(r"openrouter\.ai"), "openrouter"),
    (re.compile(r"api\.together\.xyz"), "together"),
    (re.compile(r"api\.cohere\.(?:ai|com)"), "cohere"),
    (re.compile(r"bedrock-runtime"), "aws-bedrock"),
]

# Longest prefix first; only matches classifiable as an AI provider key are
# reported (avoids flagging every high-entropy string in source).
_AI_KEY_PREFIXES: List[Tuple[str, str]] = [
    ("sk-ant-", "anthropic"),
    ("sk-", "openai"),
    ("AKIA", "aws"),
    ("gsk_", "groq"),
    ("hf_", "huggingface"),
    ("r8_", "replicate"),
    ("AIza", "google"),
]

# Supplemental prefixes PIIDetector's api_key patterns do not cover.
_EXTRA_KEY_RE = re.compile(r"\b(?:gsk_|hf_|r8_)[A-Za-z0-9]{16,}")

DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "venv",
        ".venv",
        "__pycache__",
        "dist",
        "build",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "site-packages",
        ".eggs",
    }
)
DEFAULT_MAX_FILE_BYTES = 1_000_000
DEFAULT_MAX_FILES = 5000

_SOURCE_EXTS_PY = frozenset({".py"})
_SOURCE_EXTS_JS = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})
_MANIFEST_NAMES = ("pyproject.toml", "package.json", "Pipfile")


@dataclass
class Finding:
    """One inventory finding; ``evidence`` never contains a secret value."""

    kind: str  # dependency | api_url | env_read | hardcoded_key | env_set
    provider: str
    evidence: str
    file: Optional[str] = None
    line: Optional[int] = None
    data_flow: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InventoryReport:
    """Result of :func:`scan_project`; skipped files are counted, not hidden."""

    root: str
    findings: List[Finding] = field(default_factory=list)
    scanned_files: int = 0
    skipped_files: int = 0

    @property
    def hardcoded_keys(self) -> List[Finding]:
        return [f for f in self.findings if f.kind == "hardcoded_key"]

    def summary(self) -> Dict[str, Any]:
        return {
            "findings": len(self.findings),
            "by_provider": dict(Counter(f.provider for f in self.findings)),
            "by_kind": dict(Counter(f.kind for f in self.findings)),
            "by_data_flow": dict(Counter(f.data_flow or "unknown" for f in self.findings)),
            "scanned_files": self.scanned_files,
            "skipped_files": self.skipped_files,
        }

    def risks(self) -> Dict[str, Any]:
        external = sorted({f.provider for f in self.findings if f.data_flow == "external"})
        return {
            "external_data_flow_providers": external,
            "hardcoded_keys": [f.to_dict() for f in self.hardcoded_keys],
        }

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "root": self.root,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary(),
            "risks": self.risks(),
        }
        if self.skipped_files:
            out["note"] = (
                f"skipped {self.skipped_files} file(s) (size/count caps or read errors); "
                "results may be incomplete"
            )
        return out


def _normalize_pkg(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _lookup_package(name: str) -> Optional[Dict[str, str]]:
    name = _normalize_pkg(name)
    if not name:
        return None
    info = AI_PACKAGE_REGISTRY.get(name)
    if info is not None:
        return info
    for prefix, prefix_info in _REGISTRY_PREFIXES:
        if name.startswith(prefix):
            return prefix_info
    return None


_REQ_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_PIPFILE_KEY_RE = re.compile(r'^\s*"?([A-Za-z0-9._-]+)"?\s*=')
_QUOTED_DEP_RE = re.compile(r'["\']([A-Za-z0-9@/._-]+)')


def _dep_finding(name: str, info: Dict[str, str], rel_path: str, line: int) -> Finding:
    return Finding(
        kind="dependency",
        provider=info["provider"],
        evidence=name,
        file=rel_path,
        line=line,
        data_flow=info["data_flow"],
    )


def _scan_requirements(text: str, rel_path: str) -> List[Finding]:
    findings = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _REQ_NAME_RE.match(line)
        if not m:
            continue
        info = _lookup_package(m.group(0))
        if info is not None:
            findings.append(_dep_finding(_normalize_pkg(m.group(0)), info, rel_path, lineno))
    return findings


def _scan_pipfile(text: str, rel_path: str) -> List[Finding]:
    findings = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        m = _PIPFILE_KEY_RE.match(raw)
        if not m:
            continue
        info = _lookup_package(m.group(1))
        if info is not None:
            findings.append(_dep_finding(_normalize_pkg(m.group(1)), info, rel_path, lineno))
    return findings


def _pyproject_dep_names(text: str) -> List[str]:
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        tomllib = None
    if tomllib is not None:
        try:
            data = tomllib.loads(text)
        except Exception:
            data = None
        if data is not None:
            names: List[str] = []
            project = data.get("project", {})
            deps = list(project.get("dependencies", []))
            for extra in project.get("optional-dependencies", {}).values():
                deps.extend(extra)
            for spec in deps:
                m = _REQ_NAME_RE.match(str(spec).strip())
                if m:
                    names.append(m.group(0))
            poetry = data.get("tool", {}).get("poetry", {})
            names.extend(poetry.get("dependencies", {}))
            names.extend(poetry.get("dev-dependencies", {}))
            return names
    # Fallback (Python < 3.11 or unparsable TOML): registry-match quoted strings.
    return [m.group(1) for m in _QUOTED_DEP_RE.finditer(text)]


def _scan_pyproject(text: str, rel_path: str) -> List[Finding]:
    findings = []
    seen = set()
    for name in _pyproject_dep_names(text):
        norm = _normalize_pkg(name)
        if norm in seen:
            continue
        info = _lookup_package(norm)
        if info is not None:
            seen.add(norm)
            findings.append(_dep_finding(norm, info, rel_path, 0))
    return findings


def _scan_package_json(text: str, rel_path: str) -> List[Finding]:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    findings = []
    for section in ("dependencies", "devDependencies", "peerDependencies"):
        deps = data.get(section)
        if not isinstance(deps, dict):
            continue
        for name in deps:
            info = _lookup_package(name)
            if info is not None:
                findings.append(_dep_finding(_normalize_pkg(name), info, rel_path, 0))
    return findings


def _scan_manifest(name: str, text: str, rel_path: str) -> List[Finding]:
    if fnmatch(name, "requirements*.txt"):
        return _scan_requirements(text, rel_path)
    if name == "Pipfile":
        return _scan_pipfile(text, rel_path)
    if name == "pyproject.toml":
        return _scan_pyproject(text, rel_path)
    if name == "package.json":
        return _scan_package_json(text, rel_path)
    return []


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _classify_key(matched: str) -> Optional[str]:
    for prefix, provider in _AI_KEY_PREFIXES:
        if matched.startswith(prefix):
            return provider
    return None


def _scan_source(text: str, rel_path: str, detector: PIIDetector) -> List[Finding]:
    findings: List[Finding] = []
    for pattern, provider in _URL_PROVIDERS:
        m = pattern.search(text)
        if m is not None:
            findings.append(
                Finding(
                    kind="api_url",
                    provider=provider,
                    evidence=m.group(0),
                    file=rel_path,
                    line=_line_of(text, m.start()),
                    data_flow="external",
                )
            )
    seen_env = set()
    for m in _ENV_KEY_RE.finditer(text):
        name = m.group(1)
        if name in seen_env:
            continue
        seen_env.add(name)
        findings.append(
            Finding(
                kind="env_read",
                provider=AI_ENV_KEYS[name],
                evidence=name,
                file=rel_path,
                line=_line_of(text, m.start()),
                data_flow="external",
            )
        )
    key_spans = set()
    matches = [m for m in detector.detect(text) if m.type == "api_key"]
    for m in matches:
        provider = _classify_key(m.text)
        if provider is None:
            continue
        key_spans.add(m.start)
        findings.append(
            Finding(
                kind="hardcoded_key",
                provider=provider,
                evidence=f"api_key ({provider})",  # type only, never the value
                file=rel_path,
                line=_line_of(text, m.start),
                data_flow="external",
            )
        )
    for m in _EXTRA_KEY_RE.finditer(text):
        provider = _classify_key(m.group(0))
        if provider is None or m.start() in key_spans:
            continue
        findings.append(
            Finding(
                kind="hardcoded_key",
                provider=provider,
                evidence=f"api_key ({provider})",
                file=rel_path,
                line=_line_of(text, m.start()),
                data_flow="external",
            )
        )
    return findings


def scan_project(
    path: Union[str, Path],
    include_env: bool = False,
    scan_js: bool = True,
    skip_dirs: Optional[frozenset] = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
) -> InventoryReport:
    """Statically scan ``path`` for AI usage; returns an :class:`InventoryReport`.

    ``include_env=True`` additionally reports which known AI env keys are set
    in the current environment (names only, never values).
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"scan_project path is not a directory: {path}")
    skip = DEFAULT_SKIP_DIRS if skip_dirs is None else skip_dirs
    source_exts = _SOURCE_EXTS_PY | (_SOURCE_EXTS_JS if scan_js else frozenset())
    report = InventoryReport(root=str(root))
    detector = PIIDetector()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip)
        for name in sorted(filenames):
            fpath = Path(dirpath) / name
            is_manifest = fnmatch(name, "requirements*.txt") or name in _MANIFEST_NAMES
            is_source = fpath.suffix.lower() in source_exts
            if not (is_manifest or is_source):
                continue
            if report.scanned_files >= max_files:
                report.skipped_files += 1
                continue
            try:
                if fpath.stat().st_size > max_file_bytes:
                    report.skipped_files += 1
                    continue
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                report.skipped_files += 1
                continue
            report.scanned_files += 1
            rel_path = str(fpath.relative_to(root))
            if is_manifest:
                report.findings.extend(_scan_manifest(name, text, rel_path))
            if is_source:
                report.findings.extend(_scan_source(text, rel_path, detector))
    if include_env:
        for key, provider in AI_ENV_KEYS.items():
            if os.environ.get(key):
                report.findings.append(
                    Finding(kind="env_set", provider=provider, evidence=key, data_flow="external")
                )
    return report
