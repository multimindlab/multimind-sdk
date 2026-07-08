"""
Example usage of the MultiMind CLI from a script.
"""

import subprocess


def run_cli_examples():
    # Example 1: List providers and which API keys are configured (offline)
    print("Example 1: List providers")
    subprocess.run(["multimind", "models", "list"])

    # Example 2: Scan text for PII (offline; exit code 1 when PII is found)
    print("\nExample 2: PII scan")
    subprocess.run(
        ["multimind", "compliance", "scan-text", "Contact jane.doe@example.com for details."]
    )

    # Example 3: Single-prompt chat (requires OPENAI_API_KEY)
    print("\nExample 3: Chat (needs an API key)")
    subprocess.run(["multimind", "chat", "start", "-m", "openai", "-p", "Say hello in one word."])


# Kept for backward compatibility with older imports of this example
main = run_cli_examples

if __name__ == "__main__":
    run_cli_examples()
