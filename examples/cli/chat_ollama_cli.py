#!/usr/bin/env python3
"""Interactive chat with a local Ollama model via ``multimind.models.OllamaModel``.

Usage::

    python examples/cli/chat_ollama_cli.py --model mistral
    python examples/cli/chat_ollama_cli.py --model llama3 --prompt "Hello"   # one-shot

Runs offline: if no Ollama server is reachable it prints how to start one and
exits 0.
"""

import argparse
import asyncio
import json
import os
from typing import Dict, List, Optional

import aiohttp

from multimind.models import OllamaModel

Message = Dict[str, str]


def load_history(path: Optional[str]) -> List[Message]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return [m for m in data if isinstance(m, dict) and "role" in m and "content" in m]
    except (OSError, json.JSONDecodeError):
        return []


def save_history(path: Optional[str], history: List[Message]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)


async def ask(model: OllamaModel, history: List[Message], prompt: str, stream: bool) -> str:
    """Append the user turn, query the model, append and return the assistant turn."""
    history.append({"role": "user", "content": prompt})
    messages = history[-10:]  # keep the context bounded for small local models

    if stream:
        parts: List[str] = []
        async for chunk in model.chat_stream(messages):
            print(chunk, end="", flush=True)
            parts.append(chunk)
        print()
        reply = "".join(parts).strip()
    else:
        reply = await model.chat(messages)
        print(reply)

    history.append({"role": "assistant", "content": reply})
    return reply


def unreachable_message(base_url: str, model_name: str) -> str:
    return (
        f"Cannot reach Ollama at {base_url}. Start it with `ollama serve` and pull a model "
        f"with `ollama pull {model_name}`, or point --base-url at a running server."
    )


async def run(args: argparse.Namespace) -> int:
    model = OllamaModel(model_name=args.model, base_url=args.base_url)
    history = load_history(args.history)
    stream = not args.no_stream

    try:
        if args.prompt:
            print("Assistant: ", end="", flush=True)
            await ask(model, history, args.prompt, stream)
            save_history(args.history, history)
            return 0

        print(f"Chatting with {args.model} at {args.base_url} (type 'exit' to quit)")
        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in {"exit", "quit"}:
                break
            if not prompt:
                continue
            print("Assistant: ", end="", flush=True)
            await ask(model, history, prompt, stream)
            save_history(args.history, history)
        return 0
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        print(unreachable_message(args.base_url, args.model))
        return 0
    finally:
        await model.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive chat with Ollama models")
    parser.add_argument("--model", default="mistral", help="Ollama model name (default: mistral)")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL (default: $OLLAMA_HOST or http://localhost:11434)",
    )
    parser.add_argument("--prompt", help="Send a single prompt and exit instead of chatting")
    parser.add_argument("--history", help="JSON file to load/save the conversation")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
