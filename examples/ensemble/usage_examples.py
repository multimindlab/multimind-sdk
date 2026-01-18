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
        "--providers", "openai", "--providers", "anthropic", "--providers", "ollama",
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
        "--providers", "openai", "--providers", "anthropic", "--providers", "ollama"
    ]
    subprocess.run(cmd)
    code_file.unlink()
    
    # 3. Embedding Generation
    print("\n3. Embedding Generation:")
    cmd = [
        "python", "-m", "examples.cli.ensemble_cli", "embed",
        "This is a sample text for embedding generation.",
        "--providers", "openai", "--providers", "huggingface"
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
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    
    # Wait for the server to start with retries
    max_retries = 15
    retry_count = 0
    server_ready = False
    
    # Check if process is still running
    if server_process.poll() is not None:
        print("Error: Server process failed to start")
        stderr_output = server_process.stderr.read().decode() if server_process.stderr else ""
        print(f"Server error: {stderr_output}")
        return
    
    while retry_count < max_retries and not server_ready:
        await asyncio.sleep(1)
        # Check if process is still running
        if server_process.poll() is not None:
            print("Error: Server process terminated unexpectedly")
            stderr_output = server_process.stderr.read().decode() if server_process.stderr else ""
            print(f"Server error: {stderr_output}")
            return
        try:
            response = requests.get("http://localhost:8000/docs", timeout=2)
            if response.status_code == 200:
                server_ready = True
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            retry_count += 1
    
    if not server_ready:
        print("Warning: Server may not be ready, but continuing with requests...")
        print("Note: Make sure the server is running on http://localhost:8000")
    
    try:
        # 1. Text Generation
        print("\n1. Text Generation:")
        try:
            response = requests.post(
                "http://localhost:8000/generate",
                json={
                    "prompt": "Explain the concept of ensemble learning in machine learning.",
                    "providers": ["openai", "anthropic", "ollama"],
                    "method": "weighted_voting"
                },
                timeout=120  # Increased timeout to 120 seconds
            )
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json().get("detail", str(e))
                print(f"Error: {error_detail}")
            except:
                print(f"Error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        
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
        try:
            response = requests.post(
                "http://localhost:8000/review",
                json={
                    "code": code,
                    "providers": ["openai", "anthropic", "ollama"]
                },
                timeout=120  # Increased timeout to 120 seconds
            )
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json().get("detail", str(e))
                print(f"Error: {error_detail}")
            except:
                print(f"Error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        
        # 3. Embedding Generation
        print("\n3. Embedding Generation:")
        try:
            response = requests.post(
                "http://localhost:8000/embed",
                json={
                    "text": "This is a sample text for embedding generation.",
                    "providers": ["openai", "ollama"]  # Changed from huggingface to ollama
                },
                timeout=120  # Increased timeout to 120 seconds
            )
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json().get("detail", str(e))
                print(f"Error: {error_detail}")
            except:
                print(f"Error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        
        # 4. Image Analysis (if image file exists)
        image_path = Path("sample_image.jpg")
        if image_path.exists():
            print("\n4. Image Analysis:")
            try:
                with open(image_path, "rb") as f:
                    files = {"image": f}
                    response = requests.post(
                        "http://localhost:8000/analyze-image",
                        files=files,
                        params={"providers": ["openai", "anthropic"]},
                        timeout=120  # Increased timeout to 120 seconds
                    )
                    response.raise_for_status()
                    print(json.dumps(response.json(), indent=2))
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
    
    finally:
        # Stop the server
        try:
            if sys.platform == "win32":
                server_process.terminate()
            else:
                server_process.terminate()
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
        except Exception as e:
            print(f"Error stopping server: {e}")

async def main():
    """Run all examples."""
    # Run CLI examples
    run_cli_examples()
    
    # Run API examples
    await run_api_examples()

if __name__ == "__main__":
    asyncio.run(main()) 