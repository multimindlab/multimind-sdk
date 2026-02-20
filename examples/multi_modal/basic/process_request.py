"""
Example demonstrating how to process multi-modal requests using the API.
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from multimind.api.unified_api import UnifiedRequest, ModalityInput

def get_data_path(filename: str) -> Path:
    """Get the absolute path to a data file."""
    # examples/multi_modal/basic/process_request.py -> parents[2] == examples/
    return Path(__file__).resolve().parents[2] / "data" / filename


def _pick_first(data_dir: Path, patterns: list[str]) -> Optional[Path]:
    """Pick first matching file in data_dir for given glob patterns."""
    for pat in patterns:
        matches = sorted(data_dir.glob(pat))
        if matches:
            return matches[0]
    return None

async def process_image_caption():
    """Process an image captioning request."""
    # Load and encode image
    data_dir = Path(__file__).resolve().parents[2] / "data"
    image_path = get_data_path("sample_image1.png")
    if not image_path.exists():
        fallback = _pick_first(data_dir, ["*.png", "*.jpg", "*.jpeg", "*.webp"])
        if fallback is None:
            print(f"Error: Image file not found at {image_path}")
            return
        image_path = fallback
        print(f"Warning: Using fallback image file {image_path}")
        
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Create request
    request = UnifiedRequest(
        inputs=[
            ModalityInput(
                modality="image",
                content=image_data
            ),
            ModalityInput(
                modality="text",
                content="Describe this image in detail"
            )
        ],
        use_moe=True,
        constraints={
            "max_cost": 0.1,
            "max_latency": 2000
        }
    )
    
    # Send request to API
    response = requests.post(
        "http://localhost:8000/v1/process",
        json=request.model_dump()
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Image caption generated successfully!")
        print(f"Caption: {result['outputs']['text']}")
        print("\nExpert weights:")
        for expert, weight in result['expert_weights'].items():
            print(f"- {expert}: {weight:.2f}")
    else:
        print(f"Error: {response.text}")

async def process_audio_transcription():
    """Process an audio transcription request."""
    # Load and encode audio
    data_dir = Path(__file__).resolve().parents[2] / "data"
    audio_path = get_data_path("sample_audio.mp3")
    if not audio_path.exists():
        fallback = _pick_first(data_dir, ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg"])
        if fallback is None:
            print(f"Error: Audio file not found at {audio_path}")
            return
        audio_path = fallback
        print(f"Warning: Using fallback audio file {audio_path}")
        
    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()
    
    # Create request
    request = UnifiedRequest(
        inputs=[
            ModalityInput(
                modality="audio",
                content=audio_data
            ),
            ModalityInput(
                modality="text",
                content="Transcribe this audio and summarize the key points"
            )
        ],
        use_moe=True,
        constraints={
            "max_cost": 0.05,
            "max_latency": 5000
        }
    )
    
    # Send request to API
    response = requests.post(
        "http://localhost:8000/v1/process",
        json=request.model_dump()
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Audio processed successfully!")
        print(f"Transcription: {result['outputs']['text']}")
        print("\nExpert weights:")
        for expert, weight in result['expert_weights'].items():
            print(f"- {expert}: {weight:.2f}")
    else:
        print(f"Error: {response.text}")

async def process_multi_modal_analysis():
    """Process a complex multi-modal analysis request."""
    # Load and encode media
    data_dir = Path(__file__).resolve().parents[2] / "data"
    image_path = get_data_path("sample_image1.png")
    audio_path = get_data_path("sample_audio.mp3")
    
    if not image_path.exists():
        fallback = _pick_first(data_dir, ["*.png", "*.jpg", "*.jpeg", "*.webp"])
        if fallback is None:
            print(f"Error: Image file not found at {image_path}")
            return
        image_path = fallback
        print(f"Warning: Using fallback image file {image_path}")
    if not audio_path.exists():
        fallback = _pick_first(data_dir, ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg"])
        if fallback is None:
            print(f"Error: Audio file not found at {audio_path}")
            return
        audio_path = fallback
        print(f"Warning: Using fallback audio file {audio_path}")
    
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
                content="Analyze this image and audio together. What's happening in this scene?"
            )
        ],
        use_moe=True,
        constraints={
            "max_cost": 0.15,
            "max_latency": 8000
        }
    )
    
    # Send request to API
    response = requests.post(
        "http://localhost:8000/v1/process",
        json=request.model_dump()
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Multi-modal analysis completed successfully!")
        if "image_text" in result.get("outputs", {}):
            print(f"\nImage expert output:\n{result['outputs']['image_text']}")
        if "audio_text" in result.get("outputs", {}):
            print(f"\nAudio expert output:\n{result['outputs']['audio_text']}")
        print(f"Analysis: {result['outputs']['text']}")
        print("\nExpert weights:")
        for expert, weight in result['expert_weights'].items():
            print(f"- {expert}: {weight:.2f}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    # Create example data directory if it doesn't exist
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Run examples
    print("Running image captioning example...")
    asyncio.run(process_image_caption())
    
    print("\nRunning audio transcription example...")
    asyncio.run(process_audio_transcription())
    
    print("\nRunning multi-modal analysis example...")
    asyncio.run(process_multi_modal_analysis()) 