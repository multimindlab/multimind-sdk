"""
Lightweight tests for the cost-optimized multi-modal processing example.

These tests are intentionally offline-safe:
- they do not require API keys
- they do not make network calls
"""

import sys
from pathlib import Path
import pytest

from multimind.router.multi_modal_router import MultiModalRouter
from multimind.types import UnifiedRequest, ModalityInput
from multimind.metrics.cost_tracker import CostTracker
from multimind.metrics.performance import PerformanceTracker

# Import the example class under test
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from examples.multi_modal.advanced.cost_optimized_processing import CostOptimizedMultiModalProcessor


@pytest.mark.asyncio
async def test_placeholder_flow_cost_is_zero():
    """
    With no models registered, the MultiModalRouter returns placeholder outputs.
    CostTracker should therefore record 0 cost.
    """
    router = MultiModalRouter()
    cost_tracker = CostTracker()
    performance_tracker = PerformanceTracker()

    processor = CostOptimizedMultiModalProcessor(
        router=router,
        cost_tracker=cost_tracker,
        performance_tracker=performance_tracker,
        budget=1.0,
    )

    request = UnifiedRequest(
        inputs=[
            ModalityInput(modality="text", content="hello"),
        ]
    )

    result = await processor.process_request(request, optimize_cost=True)
    assert result["cost"] == 0.0
    assert "text" in result["results"]

@pytest.mark.asyncio
async def test_placeholder_flow_image_audio_text_cost_is_zero():
    """
    With no models registered, image/audio/text modalities all use placeholder outputs.
    This should still run end-to-end without network calls and record 0 cost.
    """
    router = MultiModalRouter()
    cost_tracker = CostTracker()
    performance_tracker = PerformanceTracker()

    processor = CostOptimizedMultiModalProcessor(
        router=router,
        cost_tracker=cost_tracker,
        performance_tracker=performance_tracker,
        budget=1.0,
    )

    request = UnifiedRequest(
        inputs=[
            ModalityInput(modality="image", content="fake_image_base64"),
            ModalityInput(modality="audio", content="fake_audio_base64"),
            ModalityInput(modality="text", content="hello"),
        ]
    )

    result = await processor.process_request(request, optimize_cost=True)
    assert result["cost"] == 0.0
    assert set(result["results"].keys()) == {"image", "audio", "text"}


@pytest.mark.asyncio
async def test_get_available_models_default_is_present():
    """
    The router should always return at least one model id so callers
    don't crash with an empty list.
    """
    router = MultiModalRouter()
    models = router.get_available_models("text")
    assert isinstance(models, list)
    assert len(models) >= 1

