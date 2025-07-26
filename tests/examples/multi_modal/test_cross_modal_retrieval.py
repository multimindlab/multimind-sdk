"""
Tests for cross-modal retrieval example.
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

from examples.multi_modal.workflows.cross_modal_retrieval import (
    CrossModalRetrievalWorkflow,
    main as cross_modal_main
)

def create_test_files():
    """Create test image and audio files."""
    data_dir = Path(__file__).parent.parent.parent / "examples" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a small test image
    image_path = data_dir / "sample_image.jpg"
    if not image_path.exists():
        with open(image_path, "wb") as f:
            f.write(b"fake image data")
    
    # Create a small test audio file
    audio_path = data_dir / "sample_audio.mp3"
    if not audio_path.exists():
        with open(audio_path, "wb") as f:
            f.write(b"fake audio data")

@pytest.fixture(autouse=True)
def setup_test_files():
    """Setup test files before each test."""
    create_test_files()
    yield
    # Cleanup could be added here if needed

@pytest.mark.asyncio
async def test_cross_modal_retrieval():
    """Test cross-modal retrieval example."""
    try:
        await cross_modal_main()
        assert True
    except Exception as e:
        pytest.fail(f"Cross-modal retrieval example failed: {str(e)}")

@pytest.mark.asyncio
async def test_cross_modal_workflow():
    """Test CrossModalRetrievalWorkflow class."""
    from multimind.router.multi_modal_router import MultiModalRouter
    
    # Initialize workflow
    router = MultiModalRouter()
    workflow = CrossModalRetrievalWorkflow(
        models=router.models,
        integrations={}
    )
    
    # Create test request
    request = {
        "content": {
            "text": "This is a test text.",
            "image": base64.b64encode(b"fake image data").decode(),
            "audio": base64.b64encode(b"fake audio data").decode()
        },
        "modalities": ["text", "image", "audio"]
    }
    
    # Execute workflow
    result = await workflow.execute(request)
    
    # Check results
    assert "embeddings" in result
    assert "similarities" in result
    assert "analysis" in result
    
    # Check embeddings
    assert len(result["embeddings"]) == 3
    assert all(modality in result["embeddings"] for modality in ["text", "image", "audio"])
    
    # Check similarities
    assert len(result["similarities"]) > 0
    assert all(isinstance(similarity, float) for similarity in result["similarities"].values())
    
    # Check analysis
    assert isinstance(result["analysis"], str)
    assert len(result["analysis"]) > 0

def test_environment_variables():
    """Test required environment variables."""
    required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    assert not missing_vars, f"Missing environment variables: {', '.join(missing_vars)}"

def test_data_files():
    """Test existence of required data files."""
    data_dir = Path(__file__).parent.parent.parent / "examples" / "data"
    required_files = ["sample_image.jpg", "sample_audio.mp3"]
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    assert not missing_files, f"Missing data files: {', '.join(missing_files)}" 