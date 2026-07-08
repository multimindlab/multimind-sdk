"""Anthropic Model Context Protocol (MCP) server for the MultiMind compliance toolkit.

Exposes PII scanning/redaction, grounding checks, policy gating, and audit
logging as MCP tools over stdio (``python -m multimind.mcp_server``). The
``mcp`` package is imported lazily so this module stays importable without it.
Not to be confused with :mod:`multimind.mcp` (the internal Model Composition
Protocol).
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from multimind.compliance.guard import AuditLog, PIIDetector, PIIMatch
from multimind.evaluation.hallucination import HallucinationDetector

SERVER_NAME = "multimind-compliance"
DEFAULT_AUDIT_LOG = "multimind_audit.jsonl"

_detector = PIIDetector()
_hallucination = HallucinationDetector()


def _spans(matches: List[PIIMatch]) -> List[Dict[str, Any]]:
    return [{"type": m.type, "start": m.start, "end": m.end} for m in matches]


def _counts(matches: List[PIIMatch]) -> Dict[str, int]:
    return dict(Counter(m.type for m in matches))


def scan_text(text: str) -> Dict[str, Any]:
    """Scan text for PII; returns finding types, character spans, and per-type counts."""
    matches = _detector.detect(text)
    return {"findings": _spans(matches), "counts": _counts(matches), "total": len(matches)}


def redact_text(text: str, strategy: str = "mask") -> Dict[str, Any]:
    """Redact PII from text using strategy "mask", "hash", or "remove"; returns redacted text and a findings summary."""
    redacted, matches = _detector.redact(text, strategy)
    return {
        "redacted": redacted,
        "strategy": strategy,
        "counts": _counts(matches),
        "total": len(matches),
    }


def check_grounding(response: str, sources: List[str]) -> Dict[str, Any]:
    """Check how well each sentence of a response is grounded in the given source texts (offline, no LLM)."""
    report = _hallucination.check_grounding(response, sources)
    return {
        "score": report.score,
        "unsupported_count": report.unsupported_count,
        "summary": report.summary(),
        "sentences": [
            {
                "sentence": s.sentence,
                "verdict": s.verdict,
                "score": s.score,
                "best_source_snippet": s.best_source_snippet,
            }
            for s in report.sentences
        ],
    }


def _audit_destination() -> Path:
    return Path(os.environ.get("MULTIMIND_AUDIT_LOG", DEFAULT_AUDIT_LOG))


def audit_log(event: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Append a caller-declared compliance event to the JSONL audit trail (PII is masked before writing)."""
    event_text, matches = _detector.redact(event, "mask")
    clean_metadata: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, str):
            value, value_matches = _detector.redact(value, "mask")
            matches.extend(value_matches)
        clean_metadata[key] = value
    destination = _audit_destination()
    record: Dict[str, Any] = {
        "source": "mcp",
        "event": event_text,
        "pii_types": _counts(matches),
    }
    if clean_metadata:
        record["metadata"] = clean_metadata
    AuditLog(destination).write(record)
    return {"ok": True, "path": str(destination)}


def check_policy(text: str, block_on: List[str]) -> Dict[str, Any]:
    """Pre-send policy gate: report whether text is free of the blocked PII types."""
    blocked = {t.lower() for t in block_on}
    counts = Counter(m.type for m in _detector.detect(text) if m.type in blocked)
    violations = [{"type": t, "count": n} for t, n in sorted(counts.items())]
    return {"allowed": not violations, "violations": violations}


_TOOLS = (scan_text, redact_text, check_grounding, audit_log, check_policy)


def _load_fastmcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "The MultiMind MCP server requires the 'mcp' package. "
            "Install it with: pip install 'multimind-sdk[mcp]'"
        ) from e
    return FastMCP


def create_server():
    """Build the FastMCP server with all compliance tools registered."""
    fastmcp = _load_fastmcp()
    server = fastmcp(
        SERVER_NAME,
        instructions=(
            "MultiMind SDK compliance toolkit: scan or redact PII, check "
            "response grounding against sources, gate outbound content on "
            "blocked PII types, and append audit-trail events."
        ),
    )
    for tool in _TOOLS:
        server.tool()(tool)
    return server


def main() -> None:
    create_server().run(transport="stdio")
