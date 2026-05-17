"""MultiMind Gateway Package — unified HTTP interface for MultiMind services.

Requires the ``gateway`` extras (``fastapi``, ``uvicorn``, ``redis``, …):
``pip install 'multimind-sdk[gateway]'``.
"""

__version__ = "1.0.0"

try:
    from .api import MultiMindAPI, app, start
    from .compliance_api import router as compliance_router
    from .models import (
        AnthropicHandler,
        HuggingFaceHandler,
        OllamaHandler,
        OpenAIHandler,
    )
except ImportError as exc:  # pragma: no cover - exercised on minimal installs
    raise ImportError(
        "Gateway features require additional dependencies. "
        "Install with: pip install 'multimind-sdk[gateway]'"
    ) from exc

__all__ = [
    # API
    "MultiMindAPI",
    "app",
    "start",
    "compliance_router",
    
    # Model handlers
    "OpenAIHandler",
    "AnthropicHandler",
    "OllamaHandler",
    "HuggingFaceHandler",
]