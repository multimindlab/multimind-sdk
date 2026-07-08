"""Anthropic MCP server ("multimind-compliance") exposing MultiMind compliance tools.

Distinct from :mod:`multimind.mcp`, the internal Model Composition Protocol.
"""

from .server import (
    SERVER_NAME,
    audit_log,
    check_grounding,
    check_policy,
    create_server,
    main,
    redact_text,
    scan_text,
)

__all__ = [
    "SERVER_NAME",
    "audit_log",
    "check_grounding",
    "check_policy",
    "create_server",
    "main",
    "redact_text",
    "scan_text",
]
