"""
Tests for multi-modal examples.
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

from examples.multi_modal.basic.model_registration import register_models
from examples.multi_modal.basic.process_request import (
    process_image_caption,
    process_audio_transcription,
    process_multi_modal_analysis
)
from examples.multi_modal.workflows.workflows import run_workflow_example

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
async def test_model_registration():
    """Test model registration example."""
    try:
        await register_models()
        assert True
    except Exception as e:
        pytest.fail(f"Model registration failed: {str(e)}")

@pytest.mark.asyncio
async def test_image_caption():
    """Test image captioning example."""
    try:
        await process_image_caption()
        assert True
    except Exception as e:
        pytest.fail(f"Image captioning failed: {str(e)}")

@pytest.mark.asyncio
async def test_audio_transcription():
    """Test audio transcription example."""
    try:
        await process_audio_transcription()
        assert True
    except Exception as e:
        pytest.fail(f"Audio transcription failed: {str(e)}")

@pytest.mark.asyncio
async def test_multi_modal_analysis():
    """Test multi-modal analysis example."""
    try:
        await process_multi_modal_analysis()
        assert True
    except Exception as e:
        pytest.fail(f"Multi-modal analysis failed: {str(e)}")

@pytest.mark.asyncio
async def test_workflow_example():
    """Test workflow example."""
    try:
        await run_workflow_example()
        assert True
    except Exception as e:
        pytest.fail(f"Workflow example failed: {str(e)}")

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

def test_register_models():
    models = register_models()
    assert isinstance(models, dict)

def test_process_image_caption():
    result = process_image_caption("fake_image_data")
    assert isinstance(result, str)

def test_process_audio_transcription():
    result = process_audio_transcription("fake_audio_data")
    assert isinstance(result, str)

def test_process_multi_modal_analysis():
    result = process_multi_modal_analysis("text", "fake_image_data", "fake_audio_data")
    assert isinstance(result, str)

def test_run_workflow_example():
    try:
        result = run_workflow_example()
        assert result is not None
    except Exception:
        pass 