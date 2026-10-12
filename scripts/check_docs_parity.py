#!/usr/bin/env python3
"""Check that every CLI command and REST route is mentioned in docs/.

Usage: .venv/bin/python scripts/check_docs_parity.py

CLI commands are the leaf commands of the ``multimind`` Click group (plus the
``convert`` and ``context-transfer`` commands dispatched before Click) and must
appear in some docs/**/*.md as ``multimind <command>``. Routes come from
``create_platform_app()`` and must appear as their literal path; a ``{name}``
placeholder in the route matches any ``{...}`` placeholder in the docs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

EXTRA_CLI_COMMANDS = ("convert", "context-transfer")

# FastAPI's own documentation endpoints; not part of the product surface.
IGNORED_ROUTES = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}

# TEMPORARY allowlist: these belong in pages another change is editing right
# now (docs/dashboard.md, docs/api_reference/gateway.md). Remove entries once
# those pages mention them.
# Commands or routes that are deliberately undocumented. Keep empty.
ALLOWLIST: set[str] = set()


def docs_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(DOCS.rglob("*.md")))


def cli_commands() -> List[str]:
    import click

    from multimind.cli import cli as root

    found: List[str] = []

    def walk(cmd, prefix: List[str]) -> None:
        if isinstance(cmd, click.Group):
            for name, sub in sorted(cmd.commands.items()):
                walk(sub, [*prefix, name])
        else:
            found.append(" ".join(prefix))

    walk(root, [])
    found.extend(EXTRA_CLI_COMMANDS)
    return found


def routes() -> List[str] | None:
    """``["GET /path", ...]`` or ``None`` when fastapi is not installed."""
    try:
        import fastapi  # noqa: F401
    except ImportError:
        return None
    from multimind.server.platform import create_platform_app

    out: List[str] = []
    for route in create_platform_app().routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not path or not methods or path in IGNORED_ROUTES:
            continue
        for method in sorted(methods):
            if method != "HEAD":
                out.append(f"{method} {path}")
    return sorted(set(out))


def _route_pattern(path: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in re.split(r"\{[^}]*\}", path)]
    return re.compile(r"\{[^}/]*\}".join(parts) + r"(?![\w/{-])")


def check() -> List[str]:
    """Return the undocumented items (empty when docs are in parity)."""
    text = docs_text()
    gaps: List[str] = []
    for cmd in cli_commands():
        if f"multimind {cmd}" not in text and cmd not in ALLOWLIST:
            gaps.append(f"CLI command not documented: multimind {cmd}")
    found_routes = routes()
    if found_routes is None:
        print("fastapi not installed; skipping route check", file=sys.stderr)
        return gaps
    for entry in found_routes:
        _, path = entry.split(" ", 1)
        if entry in ALLOWLIST or path in ALLOWLIST:
            continue
        if not _route_pattern(path).search(text):
            gaps.append(f"route not documented: {entry}")
    return gaps


def main() -> int:
    gaps = check()
    if gaps:
        print("Docs parity check failed:")
        for gap in gaps:
            print(f"  - {gap}")
        return 1
    print("Docs parity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
