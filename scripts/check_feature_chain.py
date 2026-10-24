#!/usr/bin/env python
"""Prove the feature chain: component -> backend route -> dashboard view -> docs.

For every entry in the ``GET /v1/features`` registry this checks four links:

1. component  the module imports, or reports an honest ``unavailable`` reason
2. api        every route the feature claims exists on ``multimind backend``
3. ui         the dashboard bundle (``app.js``) calls every one of those routes
4. docs       every route is mentioned on a page under ``docs/``

Run: ``python scripts/check_feature_chain.py [--json]``. Exit 1 on any gap.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
APP_JS = REPO / "multimind" / "dashboard" / "static" / "app.js"


def _docs_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(DOCS.rglob("*.md")))


def _route_pattern(path: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in re.split(r"\{[^}]*\}", path)]
    return re.compile(r"\{[^}/]*\}".join(parts) + r"(?![\w/{-])")


def _ui_prefix(path: str) -> str:
    """The literal part of a path before the first ``{param}``; what the JS must contain."""
    return path.split("{", 1)[0].rstrip("/") or path


def check() -> Dict[str, Any]:
    from multimind.gateway.features_api import _FEATURES
    from multimind.server.platform import create_platform_app

    openapi = create_platform_app().openapi()["paths"]
    api_routes = {
        f"{method.upper()} {path}" for path, ops in openapi.items() for method in ops if method
    }
    js = APP_JS.read_text(encoding="utf-8") if APP_JS.exists() else ""
    docs = _docs_text()

    report: List[Dict[str, Any]] = []
    gaps: List[str] = []
    for feature in _FEATURES:
        name = feature["name"]
        row: Dict[str, Any] = {"name": name, "module": feature["module"]}
        try:
            importlib.import_module(feature["module"])
            row["component"] = "available"
        except ImportError as exc:
            # Honest unavailability (torch, optional client) is allowed; the rest must still hold.
            row["component"] = f"unavailable: {exc}"
        row["api"], row["ui"], row["docs"] = [], [], []
        for route in feature["routes"]:
            method, path = route.split(" ", 1)
            if route not in api_routes:
                row["api"].append(route)
                gaps.append(f"{name}: route missing on backend: {route}")
            if _ui_prefix(path) not in js:
                row["ui"].append(route)
                gaps.append(f"{name}: dashboard never calls {path}")
            if not _route_pattern(path).search(docs):
                row["docs"].append(route)
                gaps.append(f"{name}: route not documented: {route}")
        report.append(row)
    return {"features": report, "gaps": gaps}


def main(argv: List[str]) -> int:
    result = check()
    if "--json" in argv:
        print(json.dumps(result, indent=2))
    else:
        for row in result["features"]:
            ok = not (row["api"] or row["ui"] or row["docs"])
            print(f"{'ok ' if ok else 'GAP'} {row['name']:<22} {row['component']}")
        if result["gaps"]:
            print("\nFeature chain check failed:")
            for gap in result["gaps"]:
                print(f"  - {gap}")
        else:
            print(f"\nFeature chain check passed for {len(result['features'])} features.")
    return 1 if result["gaps"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
