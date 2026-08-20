#!/usr/bin/env python3

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OllamaError(Exception):
    """Custom exception for Ollama-related errors."""
    pass

class OllamaChat:
    def __init__(self, model_name: str = "mistral", history_file: Optional[str] = None):
        """Initialize the Ollama chat interface.

        Args:
            model_name: Name of the Ollama model to use (default: "mistral")
            history_file: Optional path to save chat history

        Raises:
            OllamaError: If Ollama is not running or model is not available
        """
        self.model_name = model_name
        self.history_file = history_file
        self.chat_history: List[Dict] = []
        self._verify_ollama_running()
        self._verify_model_available()
        self._load_history()

    def _verify_ollama_running(self) -> None:
        """Verify that Ollama is running and accessible."""
        try:
            subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5  # Add timeout to prevent hanging
            )
        except subprocess.CalledProcessError as e:
            raise OllamaError(f"Ollama is not running or not accessible: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise OllamaError("Ollama command timed out. Is the service running?")
        except FileNotFoundError:
            raise OllamaError("Ollama command not found. Is Ollama installed?")

    def _verify_model_available(self) -> None:
        """Verify that the specified model is available."""
        available_models = self.get_available_models()
        if not available_models:
            raise OllamaError("No models found. Please install at least one model using 'ollama pull <model_name>'")
        if self.model_name not in available_models:
            raise OllamaError(
                f"Model '{self.model_name}' not found. Available models: {', '.join(available_models)}\n"
                f"To install this model, run: ollama pull {self.model_name}"
            )

    def _load_history(self) -> None:
        """Load chat history from file if it exists."""
        if self.history_file and os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.chat_history = json.load(f)
                logger.info(f"Loaded chat history from {self.history_file}")
            except json.JSONDecodeError:
                logger.warning("Could not load chat history file. Starting fresh.")
                self.chat_history = []
            except Exception as e:
                logger.error(f"Error loading chat history: {e}")
                self.chat_history = []

    def _save_history(self) -> None:
        """Save chat history to file if history_file is set."""
        if self.history_file:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(self.history_file)), exist_ok=True)
                with open(self.history_file, 'w') as f:
                    json.dump(self.chat_history, f, indent=2)
                logger.info(f"Saved chat history to {self.history_file}")
            except Exception as e:
                logger.error(f"Error saving chat history: {e}")

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            # Parse the output to get model names
            models = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting model list: {e.stderr}")
            return []
        except subprocess.TimeoutExpired:
            logger.error("Timeout while getting model list")
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama.

        Args:
            model_name: Name of the model to pull

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Pulling model {model_name}... This may take a while.")
            subprocess.run(
                ["ollama", "pull", model_name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"Successfully pulled model {model_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error pulling model: {e.stderr}")
            return False

    def chat(self, prompt: str, stream: bool = True) -> str:
        """Send a message to the Ollama model and get the response.

        Args:
            prompt: The user's message
            stream: Whether to stream the response (default: True)

        Returns:
            The model's response

        Raises:
            OllamaError: If there's an error communicating with Ollama
        """
        if not prompt.strip():
            return ""

        # Add user message to history
        self.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })

        try:
            # Prepare messages for chat API (include history context)
            messages = []
            # Include recent history (last 10 messages to avoid token limits)
            for msg in self.chat_history[-10:-1]:  # Exclude the just-added user message
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            # Add current user message
            messages.append({
                "role": "user",
                "content": prompt
            })

            # Use Ollama HTTP API
            api_url = "http://localhost:11434/api/chat"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": stream
            }

            if stream:
                # Streaming response
                response = requests.post(
                    api_url,
                    json=payload,
                    stream=True,
                    timeout=300
                )
                response.raise_for_status()

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                print(content, end='', flush=True)
                                full_response += content
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

                print()  # New line after streaming
                full_response = full_response.strip()
            else:
                # Non-streaming response
                response = requests.post(
                    api_url,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
                full_response = result["message"]["content"]
                print(full_response)

            # Add assistant response to history
            self.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat()
            })

            self._save_history()
            return full_response

        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Ollama API. Is Ollama running? (Expected at http://localhost:11434)"
            logger.error(error_msg)
            raise OllamaError(error_msg)
        except requests.exceptions.Timeout:
            error_msg = "Request timed out after 5 minutes"
            logger.error(error_msg)
            raise OllamaError(error_msg)
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error communicating with Ollama: {e}"
            logger.error(error_msg)
            raise OllamaError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            raise OllamaError(error_msg)

    def show_history(self, limit: Optional[int] = None) -> None:
        """Display chat history.

        Args:
            limit: Optional number of messages to show (default: all)
        """
        if not self.chat_history:
            print("No chat history available.")
            return

        messages = self.chat_history[-limit:] if limit else self.chat_history
        for msg in messages:
            role = msg["role"].upper()
            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] {role}:")
            print(msg["content"])

    def clear_history(self) -> None:
        """Clear chat history."""
        self.chat_history = []
        if self.history_file and os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
                logger.info(f"Cleared chat history file: {self.history_file}")
            except Exception as e:
                logger.error(f"Error clearing history file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Interactive chat with Ollama models")
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Ollama model to use (default: mistral)"
    )
    parser.add_argument(
        "--history",
        type=str,
        help="Path to save chat history (optional)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming responses"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Initialize chat interface
        chat = OllamaChat(model_name=args.model, history_file=args.history)

        # Show available models
        print("\nAvailable models:")
        models = chat.get_available_models()
        for model in models:
            print(f"  - {model}")
        print(f"\nUsing model: {chat.model_name}")

        # Show chat history if it exists
        if chat.chat_history:
            print("\nPrevious chat history:")
            chat.show_history(limit=5)  # Show last 5 messages

        print("\nStarting chat (type 'exit' to quit, 'history' to show history, 'models' to list models)")
        print("Special commands:")
        print("  exit     - Exit the chat")
        print("  history  - Show chat history")
        print("  models   - List available models")
        print("  clear    - Clear chat history")
        print("  pull     - Pull a new model (e.g., 'pull llama2')")
        print("=" * 50)

        while True:
            try:
                # Get user input
                prompt = input("\nYou: ").strip()

                # Handle special commands
                if prompt.lower() == 'exit':
                    break
                elif prompt.lower() == 'history':
                    chat.show_history()
                    continue
                elif prompt.lower() == 'models':
                    print("\nAvailable models:")
                    for model in chat.get_available_models():
                        print(f"  - {model}")
                    continue
                elif prompt.lower() == 'clear':
                    chat.clear_history()
                    print("Chat history cleared.")
                    continue
                elif prompt.lower().startswith('pull '):
                    model_to_pull = prompt[5:].strip()
                    if model_to_pull:
                        if chat.pull_model(model_to_pull):
                            print(f"Model {model_to_pull} pulled successfully.")
                            # Update available models
                            models = chat.get_available_models()
                            print("\nUpdated model list:")
                            for model in models:
                                print(f"  - {model}")
                    else:
                        print("Please specify a model to pull (e.g., 'pull llama2')")
                    continue
                elif not prompt:
                    continue

                # Get response from model
                print("\nAssistant: ", end='', flush=True)
                chat.chat(prompt, stream=not args.no_stream)

            except KeyboardInterrupt:
                print("\nExiting chat...")
                break
            except OllamaError as e:
                logger.error(f"Ollama error: {e}")
                print(f"\nError: {e}")
                if "model not found" in str(e).lower():
                    print("Try pulling the model first using: pull <model_name>")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"\nAn unexpected error occurred: {e}")
                continue

    except OllamaError as e:
        logger.error(f"Initialization error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
