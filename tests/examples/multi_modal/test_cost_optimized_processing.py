"""
Tests for the cost-optimized multi-modal processing example.
"""

import pytest
pytest.skip("Skipping example test not structured as importable module.", allow_module_level=True)
import asyncio
from pathlib import Path
import sys
import os
import base64

# Add examples directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from examples.multi_modal.advanced.cost_optimized_processing import (
    CostOptimizedMultiModalProcessor
)
from multimind.router.multi_modal_router import MultiModalRouter
from multimind.metrics.cost_tracker import CostTracker
from multimind.metrics.performance import PerformanceTracker
from multimind.types import UnifiedRequest, ModalityInput


def create_test_files():
    """Create test files if they don't exist."""
    data_dir = Path("examples/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test image
    image_path = data_dir / "sample_image.jpg"
    if not image_path.exists():
        with open(image_path, "wb") as f:
            f.write(b"fake image data")
    
    # Create test audio
    audio_path = data_dir / "sample_audio.mp3"
    if not audio_path.exists():
        with open(audio_path, "wb") as f:
            f.write(b"fake audio data")


@pytest.fixture
def setup_test_files():
    """Setup test files before each test."""
    create_test_files()
    yield
    # Cleanup could be added here if needed


@pytest.mark.asyncio
async def test_cost_optimized_processing(setup_test_files):
    """Test the cost-optimized multi-modal processing."""
    
    # Initialize components
    router = MultiModalRouter()
    cost_tracker = CostTracker()
    performance_tracker = PerformanceTracker()
    
    # Create processor
    processor = CostOptimizedMultiModalProcessor(
        router=router,
        cost_tracker=cost_tracker,
        performance_tracker=performance_tracker,
        budget=0.1
    )
    
    # Load test data
    data_dir = Path("examples/data")
    image_path = data_dir / "sample_image.jpg"
    audio_path = data_dir / "sample_audio.mp3"
    
    # Read files
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()
    
    # Create request
    request = UnifiedRequest(
        inputs=[
            ModalityInput(
                modality="image",
                content=image_data
            ),
            ModalityInput(
                modality="audio",
                content=audio_data
            ),
            ModalityInput(
                modality="text",
                content="Test analysis"
            )
        ]
    )
    
    # Process request
    result = await processor.process_request(
        request,
        optimize_cost=True
    )
    
    # Verify results
    assert "results" in result
    assert "cost" in result
    assert "latency" in result
    
    # Verify modalities
    assert "image" in result["results"]
    assert "audio" in result["results"]
    assert "text" in result["results"]
    
    # Verify cost
    assert result["cost"] <= 0.1  # Budget check
    
    # Verify latency
    assert result["latency"] >= 0


@pytest.mark.asyncio
async def test_cost_optimized_processor(setup_test_files):
    """Test the CostOptimizedMultiModalProcessor class."""
    
    # Initialize components
    router = MultiModalRouter()
    cost_tracker = CostTracker()
    performance_tracker = PerformanceTracker()
    
    # Create processor
    processor = CostOptimizedMultiModalProcessor(
        router=router,
        cost_tracker=cost_tracker,
        performance_tracker=performance_tracker,
        budget=0.1
    )
    
    # Test model selection
    model = processor._get_cost_optimized_model("image")
    assert model is not None
    
    # Test budget exceeded
    with pytest.raises(ValueError):
        await processor.process_request(
            UnifiedRequest(
                inputs=[
                    ModalityInput(
                        modality="image",
                        content="large_image_data" * 1000
                    )
                ]
            ),
            optimize_cost=True
        )


def test_environment_variables():
    """Test required environment variables."""
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUGGINGFACE_API_KEY"
    ]
    
    for var in required_vars:
        assert var in os.environ, f"Missing required environment variable: {var}"


def test_data_files():
    """Test required data files."""
    data_dir = Path("examples/data")
    required_files = [
        "sample_image.jpg",
        "sample_audio.mp3"
    ]
    
    for file in required_files:
        assert (data_dir / file).exists(), f"Missing required data file: {file}" 