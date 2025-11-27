"""
CLI interface for the MultiMind Ensemble system.
"""

import os
import asyncio
import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import click

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from multimind import (
    AdvancedEnsemble,
    EnsembleMethod,
    Router,
    RoutingStrategy,
    TaskConfig,
    TaskType,
)
from multimind.core.provider import ProviderConfig, ImageAnalysisResult, GenerationResult, EmbeddingResult
from multimind.providers.openai import OpenAIProvider
from multimind.providers.claude import ClaudeProvider
from multimind.providers.ollama import OllamaProvider

if load_dotenv:
    load_dotenv()

# Suppress error logging from metrics since we handle errors gracefully in CLI
multimind_logger = logging.getLogger("multimind")
multimind_logger.setLevel(logging.WARNING)  # Only show WARNING and above, suppress ERROR

PROVIDER_ALIASES: Dict[str, str] = {
    "claude": "anthropic",
}

PROVIDER_REGISTRY: Dict[str, Dict[str, object]] = {
    "openai": {
        "env": ["OPENAI_API_KEY"],
        "adapter": OpenAIProvider,
        "capabilities": {
            TaskType.TEXT_GENERATION,
            TaskType.EMBEDDINGS,
            TaskType.IMAGE_ANALYSIS,
        },
    },
    "anthropic": {
        "env": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "adapter": ClaudeProvider,
        "capabilities": {
            TaskType.TEXT_GENERATION,
            TaskType.IMAGE_ANALYSIS,
        },
    },
    "ollama": {
        "env": [],  # No API key needed for local Ollama
        "adapter": OllamaProvider,
        "capabilities": {
            TaskType.TEXT_GENERATION,
            TaskType.EMBEDDINGS,
            TaskType.IMAGE_ANALYSIS,  # Supported via vision models like llava
        },
    },
}


def _normalize_provider(name: str) -> str:
    return PROVIDER_ALIASES.get(name.lower(), name.lower())


def _get_env_value(env_vars: List[str]) -> Optional[str]:
    for var in env_vars:
        value = os.getenv(var)
        if value:
            return value
    return None


def _get_provider_name(result: Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]) -> str:
    """Extract provider name from a result object."""
    return getattr(result, 'provider_name', getattr(result, 'provider', 'unknown'))


def _configure_task(router: Router, task_type: TaskType, providers: List[str]) -> None:
    if not providers:
        return
    if len(providers) == 1:
        router.configure_task(
            task_type,
            TaskConfig(
                preferred_providers=providers,
                fallback_providers=[],
                routing_strategy=RoutingStrategy.COST_BASED,
            ),
        )
        return
    weight = 1.0 / len(providers)
    router.configure_task(
        task_type,
        TaskConfig(
            preferred_providers=providers,
            fallback_providers=[],
            routing_strategy=RoutingStrategy.ENSEMBLE,
            ensemble_config={
                "method": "weighted_voting",
                "weights": {provider: weight for provider in providers},
            },
        ),
    )


def _configure_default_tasks(router: Router, providers: List[str]) -> None:
    _configure_task(router, TaskType.TEXT_GENERATION, providers)
    embedding_providers = [
        p
        for p in providers
        if TaskType.EMBEDDINGS in PROVIDER_REGISTRY.get(p, {}).get("capabilities", set())
    ]
    _configure_task(router, TaskType.EMBEDDINGS, embedding_providers)
    image_providers = [
        p
        for p in providers
        if TaskType.IMAGE_ANALYSIS in PROVIDER_REGISTRY.get(p, {}).get("capabilities", set())
    ]
    _configure_task(router, TaskType.IMAGE_ANALYSIS, image_providers)


def _prepare_router(requested: Sequence[str]) -> Tuple[Router, List[str]]:
    normalized = [_normalize_provider(p) for p in requested]
    if not normalized:
        normalized = ["openai"]
    router = Router()
    registered: List[str] = []
    seen = set()
    missing_messages: List[str] = []
    for name in normalized:
        if name in seen:
            continue
        seen.add(name)
        spec = PROVIDER_REGISTRY.get(name)
        if not spec:
            missing_messages.append(f"{name} (not supported)")
            continue
        env_vars = spec["env"]  # type: ignore[index]
        api_key = _get_env_value(env_vars) if env_vars else None
        # For Ollama (no API key needed), we still need to check if it's available
        if name == "ollama":
            # Try to verify Ollama is running by checking if we can connect
            # We'll let the actual API call fail if Ollama isn't running
            pass
        elif not api_key:
            env_list = " or ".join(env_vars)
            missing_messages.append(f"{name} (set {env_list})")
            continue
        adapter_cls = spec["adapter"]  # type: ignore[index]
        # For Ollama, use api_base from config or default, and set longer timeout
        if name == "ollama":
            import os
            ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # Set longer timeout for local models (600s = 10 minutes)
            # Can be overridden with OLLAMA_TIMEOUT environment variable
            ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
            provider_adapter = adapter_cls(ProviderConfig(api_key=None, api_base=ollama_base, timeout=ollama_timeout))
        else:
            provider_adapter = adapter_cls(ProviderConfig(api_key=api_key))
        router.register_provider(name, provider_adapter)
        registered.append(name)
    if missing_messages:
        for msg in missing_messages:
            click.echo(f"Skipping provider: {msg}", err=True)
    if not registered:
        raise click.ClickException(
            "No supported providers were configured. Set the required API keys first."
        )
    _configure_default_tasks(router, registered)
    # Disable fallback notifications since we handle errors gracefully in CLI
    router.fallback_policy.notify_user = False
    return router, registered


def _providers_for_task(registered: List[str], task_type: TaskType) -> List[str]:
    return [
        provider
        for provider in registered
        if task_type in PROVIDER_REGISTRY.get(provider, {}).get("capabilities", set())
    ]

@click.group()
def ensemble():
    """MultiMind Ensemble CLI commands."""
    pass

@ensemble.command()
@click.argument('prompt')
@click.option('--providers', '-p', multiple=True, default=['openai', 'anthropic', 'ollama'],
              help='List of providers to use')
@click.option('--method', '-m', type=click.Choice([m.value for m in EnsembleMethod]),
              default=EnsembleMethod.WEIGHTED_VOTING.value,
              help='Ensemble method to use')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def generate(prompt: str, providers: List[str], method: str, output: Optional[str]):
    """Generate text using ensemble of models."""
    router, configured_providers = _prepare_router(providers)
    ensemble = AdvancedEnsemble(router)
    text_providers = _providers_for_task(configured_providers, TaskType.TEXT_GENERATION)
    if not text_providers:
        raise click.ClickException("No configured providers support text generation.")

    async def run():
        # Get results from all providers
        async def get_result(provider: str):
            try:
                model = "gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "mistral"
                return await router.route(
                    TaskType.TEXT_GENERATION,
                    prompt,
                    provider=provider,
                    model=model
                )
            except Exception as e:
                error_msg = str(e)
                if provider == "ollama":
                    if "not found" in error_msg.lower():
                        click.echo(
                            f"Skipping provider: ollama (model '{model}' not found). "
                            f"Run 'ollama pull {model}' to install the model.",
                            err=True,
                        )
                    elif "timeout" in error_msg.lower():
                        click.echo(
                            f"Skipping provider: ollama (request timeout). Model may be too slow on CPU.",
                            err=True,
                        )
                    else:
                        click.echo(
                            f"Skipping provider: ollama ({error_msg})",
                            err=True,
                        )
                else:
                    click.echo(
                        f"Skipping provider: {provider} ({error_msg})",
                        err=True,
                    )
                return None
        
        results = await asyncio.gather(*[get_result(provider) for provider in text_providers])
        results = [r for r in results if r is not None]  # Filter out None results
        
        if not results:
            raise click.ClickException(
                "Text generation failed: all providers returned errors."
            )
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod(method),
            task_type=TaskType.TEXT_GENERATION
        )
        
        # Format output
        result_obj = combined_result.result
        if hasattr(result_obj, 'text'):
            result_text = result_obj.text
        else:
            result_text = str(result_obj)
        
        output_data = {
            "result": result_text,
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
        else:
            click.echo(json.dumps(output_data, indent=2))
    
    asyncio.run(run())

@ensemble.command()
@click.argument('code', type=click.Path(exists=True))
@click.option('--providers', '-p', multiple=True, default=['openai', 'anthropic', 'ollama'],
              help='List of providers to use')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def review(code: str, providers: List[str], output: Optional[str]):
    """Review code using ensemble of models."""
    router, configured_providers = _prepare_router(providers)
    ensemble = AdvancedEnsemble(router)
    review_providers = _providers_for_task(configured_providers, TaskType.TEXT_GENERATION)
    if not review_providers:
        raise click.ClickException("No configured providers support code review.")

    async def run():
        # Read code file
        with open(code, 'r') as f:
            code_content = f.read()
        
        # Prepare prompt
        prompt = f"""Please review the following code and provide feedback on:
1. Code quality
2. Potential bugs
3. Security issues
4. Performance improvements
5. Best practices

Code:
{code_content}"""
        
        evaluation_models: Dict[str, str] = {}

        # Get reviews from all providers
        async def get_review(provider: str):
            try:
                model = "gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "mistral"
                # Track which model to use as the evaluator/judge for this provider
                if provider == "openai":
                    evaluation_models[provider] = "gpt-4"
                elif provider == "anthropic":
                    evaluation_models[provider] = "claude-3-sonnet"
                elif provider == "ollama":
                    evaluation_models[provider] = os.getenv("OLLAMA_TEXT_MODEL", "mistral")
                else:
                    evaluation_models.setdefault(provider, "gpt-4")

                return await router.route(
                    TaskType.TEXT_GENERATION,
                    prompt,
                    provider=provider,
                    model=model
                )
            except Exception as e:
                error_msg = str(e)
                if provider == "ollama":
                    if "not found" in error_msg.lower():
                        click.echo(
                            f"Skipping provider: ollama (model '{model}' not found). "
                            f"Run 'ollama pull {model}' to install the model.",
                            err=True,
                        )
                    elif "timeout" in error_msg.lower():
                        click.echo(
                            f"Skipping provider: ollama (request timeout). Model may be too slow on CPU.",
                            err=True,
                        )
                    else:
                        click.echo(
                            f"Skipping provider: ollama ({error_msg})",
                            err=True,
                        )
                else:
                    click.echo(
                        f"Skipping provider: {provider} ({error_msg})",
                        err=True,
                    )
                return None
        
        results = await asyncio.gather(*[get_review(provider) for provider in review_providers])
        results = [r for r in results if r is not None]  # Filter out None results
        
        if not results:
            raise click.ClickException(
                "Code review failed: all providers returned errors."
            )
        
        # Combine results using the confidence cascade (same scoring used for image analysis)
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.CONFIDENCE_CASCADE,
            task_type=TaskType.TEXT_GENERATION,
            confidence_threshold=0.7,
            evaluation_models=evaluation_models
        )
        
        # Format output
        result_obj = combined_result.result
        if hasattr(result_obj, 'text'):
            review_text = result_obj.text
        else:
            review_text = str(result_obj)
        
        output_data = {
            "review": review_text,
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
        else:
            click.echo(json.dumps(output_data, indent=2))
    
    asyncio.run(run())

@ensemble.command()
@click.argument('image', type=click.Path(exists=True), required=True)
@click.option('--providers', '-p', multiple=True, default=['openai', 'anthropic', 'ollama'],
              help='List of providers to use')
@click.option('--prompt', '-q', 'analysis_prompt',
              default='Describe this image in detail, extracting key objects, text, and context.',
              help='Instruction sent to vision models.')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def analyze_image(image: str, providers: List[str], analysis_prompt: str, output: Optional[str]):
    """Analyze image using ensemble of models.
    
    IMAGE: Path to the image file to analyze (required)
    
    Example:
        ensemble_cli.py analyze-image path/to/image.jpg --prompt "Summarize the slide"
    """
    router, configured_providers = _prepare_router(providers)
    ensemble = AdvancedEnsemble(router)
    vision_providers = _providers_for_task(configured_providers, TaskType.IMAGE_ANALYSIS)
    if not vision_providers:
        raise click.ClickException("No configured providers support image analysis.")

    async def run():
        # Read image file
        with open(image, 'rb') as f:
            image_data = f.read()
        
        async def get_model_for_provider(provider: str) -> Optional[str]:
            """Select the best available model for a provider."""
            if provider == "openai":
                return "gpt-4o-mini"
            if provider == "anthropic":
                return "claude-3-sonnet"
            if provider == "ollama":
                return os.getenv("OLLAMA_VISION_MODEL", "llava-phi3:latest")
            return "default"
        
        results = []
        evaluation_models: Dict[str, str] = {}
        for provider in vision_providers:
            model_name = None
            try:
                model_name = await get_model_for_provider(provider)
                if not model_name:
                    continue
                if provider == "openai":
                    evaluation_models[provider] = "gpt-4"
                elif provider == "anthropic":
                    evaluation_models[provider] = "claude-3-sonnet"
                elif provider == "ollama":
                    evaluation_models[provider] = os.getenv("OLLAMA_TEXT_MODEL", "mistral")
                
                result = await router.route(
                    TaskType.IMAGE_ANALYSIS,
                    image_data,
                    provider=provider,
                    model=model_name,
                    prompt=analysis_prompt
                )
                results.append(result)
            except Exception as e:
                error_msg = str(e)
                if provider == "ollama":
                    # Check if it's a model not found error
                    if "not found" in error_msg.lower():
                        click.echo(
                            f"Skipping provider: ollama (model '{model_name or 'unknown'}' not found). "
                            f"Run 'ollama pull {model_name or 'llava-phi3:latest'}' to install the model.",
                            err=True,
                        )
                    elif "timeout" in error_msg.lower():
                        click.echo(
                            f"Skipping provider: ollama (request timeout). Model may be too slow on CPU.",
                            err=True,
                        )
                    else:
                        click.echo(
                            f"Skipping provider: ollama ({error_msg})",
                            err=True,
                        )
                else:
                    click.echo(
                        f"Skipping provider: {provider} ({error_msg})",
                        err=True,
                    )
        
        if not results:
            raise click.ClickException(
                "Image analysis failed: all providers returned errors."
            )
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.CONFIDENCE_CASCADE,
            task_type=TaskType.IMAGE_ANALYSIS,
            confidence_threshold=0.7,
            evaluation_models=evaluation_models
        )
        
        # Format output
        result_obj = combined_result.result
        if isinstance(result_obj, ImageAnalysisResult):
            analysis_text = result_obj.text or (result_obj.captions[0] if result_obj.captions else "No analysis available")
        else:
            analysis_text = getattr(result_obj, 'text', str(result_obj))
        
        output_data = {
            "analysis": analysis_text,
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
        else:
            click.echo(json.dumps(output_data, indent=2))
    
    asyncio.run(run())

@ensemble.command()
@click.argument('text')
@click.option('--providers', '-p', multiple=True, default=['openai', 'ollama', 'huggingface'],
              help='List of providers to use')
@click.option('--model', '-m', type=str, help='Model to use (provider-specific). For Ollama, defaults to "mistral" if available, otherwise tries "llama2"')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def embed(text: str, providers: List[str], model: Optional[str], output: Optional[str]):
    """Generate embeddings using ensemble of models."""
    router, configured_providers = _prepare_router(providers)
    ensemble = AdvancedEnsemble(router)
    embedding_providers = _providers_for_task(configured_providers, TaskType.EMBEDDINGS)
    if not embedding_providers:
        raise click.ClickException("No configured providers support embeddings.")

    async def run():
        async def get_model_for_provider(provider: str) -> str:
            """Get the appropriate model for a provider."""
            if model:
                # User specified a model, use it for all providers
                return model
            
            # Provider-specific defaults
            if provider == "openai":
                return "text-embedding-ada-002"
            elif provider == "ollama":
                # Try to auto-detect available models
                try:
                    ollama_adapter = router.providers.get("ollama")
                    if ollama_adapter:
                        available_models = await ollama_adapter.list_models()
                        # Prefer models that commonly support embeddings
                        preferred_models = ["mistral", "llama2", "nomic-embed-text", "all-minilm"]
                        for pref in preferred_models:
                            if pref in available_models:
                                return pref
                        # If no preferred model found, use first available
                        if available_models:
                            return available_models[0]
                except Exception:
                    pass
                # Fallback to mistral (most common)
                return "mistral"
            else:
                return "default"
        
        # Get embeddings from all providers with proper model selection
        results = []
        for provider in embedding_providers:
            try:
                model_name = await get_model_for_provider(provider)
                result = await router.route(
                    TaskType.EMBEDDINGS,
                    text,
                    provider=provider,
                    model=model_name
                )
                results.append(result)
            except Exception as e:
                # If model not found, try to suggest available models
                if provider == "ollama" and "not found" in str(e).lower():
                    try:
                        ollama_adapter = router.providers.get("ollama")
                        if ollama_adapter:
                            available = await ollama_adapter.list_models()
                            if available:
                                click.echo(f"Warning: Model not found. Available Ollama models: {', '.join(available)}", err=True)
                                click.echo(f"Try: --model <model_name> or pull the model with: ollama pull <model_name>", err=True)
                    except Exception:
                        pass
                # Re-raise the error
                raise
        
        # Combine results
        if not results:
            raise click.ClickException("No providers succeeded in generating embeddings.")
        
        weight = 1.0 / len(results)
        # Map results to their providers (we need to track which provider succeeded)
        # For now, use equal weights for all successful results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.EMBEDDINGS,
            weights={_get_provider_name(r): weight for r in results}
        )
        
        # Format output
        result_obj = combined_result.result
        if hasattr(result_obj, 'embedding'):
            embedding_data = result_obj.embedding
        else:
            embedding_data = []
        
        output_data = {
            "embedding": embedding_data,
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
        else:
            click.echo(json.dumps(output_data, indent=2))
    
    asyncio.run(run())

if __name__ == '__main__':
    ensemble() 