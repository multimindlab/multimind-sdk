"""
Example demonstrating how to process multi-modal requests using the API.
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import Dict, Any
import requests
from multimind.api.unified_api import UnifiedRequest, ModalityInput

def get_data_path(filename: str) -> Path:
    """Get the absolute path to a data file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "..", "data", filename))

async def process_image_caption():
    """Process an image captioning request."""
    # Load and encode image
    image_path = get_data_path("sample_image.jpg")
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return
        
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
        json=request.dict()
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
    audio_path = get_data_path("sample_audio.mp3")
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return
        
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
        json=request.dict()
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
    image_path = get_data_path("sample_image.jpg")
    audio_path = get_data_path("sample_audio.mp3")
    
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return
    
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
        json=request.dict()
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Multi-modal analysis completed successfully!")
        print(f"Analysis: {result['outputs']['text']}")
        print("\nExpert weights:")
        for expert, weight in result['expert_weights'].items():
            print(f"- {expert}: {weight:.2f}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    # Create example data directory if it doesn't exist
    data_dir = Path(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Run examples
    print("Running image captioning example...")
    asyncio.run(process_image_caption())
    
    print("\nRunning audio transcription example...")
    asyncio.run(process_audio_transcription())
    
    print("\nRunning multi-modal analysis example...")
    asyncio.run(process_multi_modal_analysis()) 