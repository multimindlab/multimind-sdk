#!/usr/bin/env python
"""Export the OpenAPI schema of every MultiMind FastAPI app to docs/api/."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "api"


def _apps():
    from multimind.api.multi_model_api import app as multi_model_app
    from multimind.api.unified_api import app as unified_app
    from multimind.gateway.api import app as gateway_app
    from multimind.gateway.rag_api import app as rag_app
    from multimind.server import MultiMindServer

    return {
        "gateway": gateway_app,
        "multi-model": multi_model_app,
        "unified": unified_app,
        "rag": rag_app,
        "server": MultiMindServer().get_app(),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, app in _apps().items():
        schema = app.openapi()
        path = OUTPUT_DIR / f"openapi-{name}.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=False) + "\n")
        print(
            f"wrote {path.relative_to(REPO_ROOT)} ({schema['info']['title']} v{schema['info']['version']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
