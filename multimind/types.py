"""
Shared type definitions for MultiMind (single contract module).

All layers should import these models instead of redefining them:
- API (FastAPI)
- Router
- MoE
- Workflows
- CLI / SDK examples
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModalityInput(BaseModel):
    """Input for a specific modality."""

    content: Any
    modality: str


class ModalityOutput(BaseModel):
    """Output for a specific modality.

    Kept flexible (extra fields allowed) to support different backends.
    """

    content: Any = None
    modality: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class UnifiedRequest(BaseModel):
    """Unified request structure for multi-modal processing."""

    inputs: List[ModalityInput]
    use_moe: bool = Field(default=True, description="Whether to use MoE processing")
    constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Processing constraints (cost, latency, etc.)",
    )
    workflow: Optional[str] = Field(
        default=None,
        description="Optional MCP workflow to use",
    )


class UnifiedResponse(BaseModel):
    """Unified response structure."""

    outputs: Dict[str, Any]
    expert_weights: Optional[Dict[str, float]] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
