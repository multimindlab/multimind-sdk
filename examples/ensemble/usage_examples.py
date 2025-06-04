"""
Examples of using the MultiMind Ensemble system through CLI and API interfaces.
"""

import asyncio
import json
import requests
from pathlib import Path
import subprocess
import sys
from typing import Dict, Any

# Example 1: Using the CLI interface
def run_cli_examples():
    """Run examples using the CLI interface."""
    print("\n=== CLI Examples ===")
    
    # 1. Text Generation
    print("\n1. Text Generation:")
    cmd = [
        "python", "-m", "examples.cli.ensemble_cli", "generate",
        "Explain the concept of ensemble learning in machine learning.",
        "--providers", "openai", "anthropic", "ollama",
        "--method", "weighted_voting"
    ]
    subprocess.run(cmd)
    
    # 2. Code Review
    print("\n2. Code Review:")
    code = """
    def calculate_factorial(n):
        if n < 0:
            return None
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    """
    code_file = Path("temp_code.py")
    code_file.write_text(code)
    
    cmd = [
        "python", "-m", "examples.cli.ensemble_cli", "review",
        str(code_file),
        "--providers", "openai", "anthropic", "ollama"
    ]
    subprocess.run(cmd)
    code_file.unlink()
    
    # 3. Embedding Generation
    print("\n3. Embedding Generation:")
    cmd = [
        "python", "-m", "examples.cli.ensemble_cli", "embed",
        "This is a sample text for embedding generation.",
        "--providers", "openai", "huggingface"
    ]
    subprocess.run(cmd)

# Example 2: Using the API interface
async def run_api_examples():
    """Run examples using the API interface."""
    print("\n=== API Examples ===")
    
    # Start the API server in a separate process
    server_process = subprocess.Popen(
        [sys.executable, "-m", "examples.api.ensemble_api"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for the server to start
    await asyncio.sleep(2)
    
    try:
        # 1. Text Generation
        print("\n1. Text Generation:")
        response = requests.post(
            "http://localhost:8000/generate",
            json={
                "prompt": "Explain the concept of ensemble learning in machine learning.",
                "providers": ["openai", "anthropic", "ollama"],
                "method": "weighted_voting"
            }
        )
        print(json.dumps(response.json(), indent=2))
        
        # 2. Code Review
        print("\n2. Code Review:")
        code = """
        def calculate_factorial(n):
            if n < 0:
                return None
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        """
        response = requests.post(
            "http://localhost:8000/review",
            json={
                "code": code,
                "providers": ["openai", "anthropic", "ollama"]
            }
        )
        print(json.dumps(response.json(), indent=2))
        
        # 3. Embedding Generation
        print("\n3. Embedding Generation:")
        response = requests.post(
            "http://localhost:8000/embed",
            json={
                "text": "This is a sample text for embedding generation.",
                "providers": ["openai", "huggingface"]
            }
        )
        print(json.dumps(response.json(), indent=2))
        
        # 4. Image Analysis (if image file exists)
        image_path = Path("sample_image.jpg")
        if image_path.exists():
            print("\n4. Image Analysis:")
            with open(image_path, "rb") as f:
                files = {"image": f}
                response = requests.post(
                    "http://localhost:8000/analyze-image",
                    files=files,
                    params={"providers": ["openai", "anthropic"]}
                )
            print(json.dumps(response.json(), indent=2))
    
    finally:
        # Stop the server
        server_process.terminate()
        server_process.wait()

async def main():
    """Run all examples."""
    # Run CLI examples
    run_cli_examples()
    
    # Run API examples
    await run_api_examples()

if __name__ == "__main__":
    asyncio.run(main()) 