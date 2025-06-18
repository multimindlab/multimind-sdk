"""
Example usage of the MultiModelWrapper CLI interface.
"""

import subprocess
import json

def run_cli_examples():
    # Example 1: Generate text
    print("Example 1: Generate text")
    cmd = [
        "python", "-m", "multimind.cli.multi_model_cli", "generate",
        "--primary-model", "openai",
        "--fallback-models", "claude",
        "--model-weights", json.dumps({"openai": 0.7, "claude": 0.3}),
        "--temperature", "0.7",
        "Explain quantum computing in simple terms."
    ]
    subprocess.run(cmd)

    # Example 2: Chat completion
    print("\nExample 2: Chat completion")
    cmd = [
        "python", "-m", "multimind.cli.multi_model_cli", "chat",
        "--primary-model", "openai",
        "--fallback-models", "claude",
        "--system-message", "You are a helpful AI assistant.",
        "--temperature", "0.7",
        "What are the benefits of using multiple AI models?"
    ]
    subprocess.run(cmd)

    # Example 3: Generate embeddings
    print("\nExample 3: Generate embeddings")
    cmd = [
        "python", "-m", "multimind.cli.multi_model_cli", "embeddings",
        "--primary-model", "openai",
        "--fallback-models", "claude",
        "This is a test sentence for embeddings."
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    run_cli_examples() 