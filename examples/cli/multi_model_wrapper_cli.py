import argparse
import asyncio
import logging

from multimind import ModelFactory, MultiModelWrapper

logger = logging.getLogger(__name__)

async def main():
    # Initialize factory first to get available models
    factory = ModelFactory()
    available = factory.available_models()

    if not available:
        print("No models available. Please check your API keys and Ollama installation.")
        return

    parser = argparse.ArgumentParser(description="Query various LLM models using CLI")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    parser.add_argument(
        "--model",
        choices=available,  # Use dynamically available models
        required=False,
        help=f"Model to use. Available models: {', '.join(available)}"
    )
    parser.add_argument("--prompt", type=str, required=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=None)

    args = parser.parse_args()

    # If --list-models is specified, show available models and exit
    if args.list_models:
        print("\n=== Available Models ===")
        for i, model in enumerate(available, 1):
            print(f"{i}. {model}")
        print(f"\nTotal: {len(available)} model(s) available")
        print("\nTo use a model, run:")
        print("  python examples\\cli\\multi_model_wrapper_cli.py --model <model_name> --prompt \"your prompt\"")
        return

    # Validate required arguments if not listing models
    if not args.model or not args.prompt:
        parser.error("--model and --prompt are required (unless using --list-models)")

    logger.info(f"Querying {args.model} with prompt: {args.prompt}")

    # Create MultiModelWrapper with selected model as primary
    # Use other available models as fallbacks
    fallback_models = [m for m in available if m != args.model]

    try:
        wrapper = MultiModelWrapper(
            model_factory=factory,
            primary_model=args.model,
            fallback_models=fallback_models
        )

        # Generate response using async method
        response = await wrapper.generate(
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )

        print(f"\n--- {args.model.upper()} Response ---\n")
        print(response)

    except Exception as e:
        logger.error(f"Error querying {args.model}: {str(e)}")
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
