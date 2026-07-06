"""
CLI interface for the MultiModelWrapper.
"""

import asyncio
import json
from typing import List, Optional

import click

from ..models.factory import ModelFactory
from ..models.multi_model import MultiModelWrapper


@click.group()
def cli():
    """Multi-model CLI interface with config/feedback commands."""
    pass


@cli.command()
@click.option("--primary-model", default="openai", help="Primary model to use")
@click.option("--fallback-models", multiple=True, help="Fallback models to use")
@click.option("--model-weights", help="JSON string of model weights")
@click.option("--temperature", default=0.7, help="Temperature for generation")
@click.option("--max-tokens", type=int, help="Maximum tokens to generate")
@click.argument("prompt")
def generate(
    primary_model: str,
    fallback_models: List[str],
    model_weights: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    prompt: str,
):
    """Generate text using the multi-model wrapper."""

    async def run():
        factory = ModelFactory()
        weights = json.loads(model_weights) if model_weights else None

        multi_model = MultiModelWrapper(
            model_factory=factory,
            primary_model=primary_model,
            fallback_models=list(fallback_models),
            model_weights=weights,
        )

        response = await multi_model.generate(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens
        )
        click.echo(response)

    asyncio.run(run())


@cli.command()
@click.option("--primary-model", default="openai", help="Primary model to use")
@click.option("--fallback-models", multiple=True, help="Fallback models to use")
@click.option("--model-weights", help="JSON string of model weights")
@click.option("--temperature", default=0.7, help="Temperature for generation")
@click.option("--max-tokens", type=int, help="Maximum tokens to generate")
@click.option("--system-message", default="You are a helpful AI assistant.", help="System message")
@click.argument("user_message")
def chat(
    primary_model: str,
    fallback_models: List[str],
    model_weights: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    system_message: str,
    user_message: str,
):
    """Generate chat completion using the multi-model wrapper."""

    async def run():
        factory = ModelFactory()
        weights = json.loads(model_weights) if model_weights else None

        multi_model = MultiModelWrapper(
            model_factory=factory,
            primary_model=primary_model,
            fallback_models=list(fallback_models),
            model_weights=weights,
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response = await multi_model.chat(
            messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        click.echo(response)

    asyncio.run(run())


@cli.command()
@click.option("--primary-model", default="openai", help="Primary model to use")
@click.option("--fallback-models", multiple=True, help="Fallback models to use")
@click.option("--model-weights", help="JSON string of model weights")
@click.argument("text")
def embeddings(
    primary_model: str, fallback_models: List[str], model_weights: Optional[str], text: str
):
    """Generate embeddings using the multi-model wrapper."""

    async def run():
        factory = ModelFactory()
        weights = json.loads(model_weights) if model_weights else None

        multi_model = MultiModelWrapper(
            model_factory=factory,
            primary_model=primary_model,
            fallback_models=list(fallback_models),
            model_weights=weights,
        )

        embeddings = await multi_model.embeddings(text)
        click.echo(json.dumps(embeddings))

    asyncio.run(run())


@cli.command()
def list_strategies():
    """List available ensemble, fusion, and router strategies."""
    click.echo(
        "Ensemble: weighted_voting, confidence_cascade, parallel_voting, majority_voting, rank_based"
    )
    click.echo(
        "Fusion: weighted_sum, neural_fusion, multi_layer_fusion, attention_fusion, transformer_fusion"
    )
    click.echo("Router: cost, latency, hybrid, pareto, learning, deep_rl")


@cli.command()
@click.option("--strategy-type", type=click.Choice(["ensemble", "fusion", "router"]), required=True)
@click.option("--strategy", required=True, help="Strategy name to set")
def set_strategy(strategy_type, strategy):
    """Set the active strategy for ensemble, fusion, or router."""
    raise click.ClickException("set-strategy is not implemented yet")


@cli.command()
def show_config():
    """Show current configuration for ensemble, fusion, router, and memory."""
    raise click.ClickException("show-config is not implemented yet")


@cli.command()
@click.option("--strategy-type", type=click.Choice(["ensemble", "fusion", "router"]), required=True)
@click.option("--param", required=True, help="Parameter name (e.g., weight, threshold)")
@click.option("--value", required=True, help="Parameter value (JSON or string)")
def set_param(strategy_type, param, value):
    """Set a parameter for a strategy (e.g., weight, threshold)."""
    raise click.ClickException("set-param is not implemented yet")


@cli.command()
@click.option("--strategy-type", type=click.Choice(["ensemble", "fusion", "router"]), required=True)
@click.option("--feedback", required=True, help="Feedback value (e.g., success, fail, numeric)")
def submit_feedback(strategy_type, feedback):
    """Submit feedback for a strategy (e.g., after a request)."""
    raise click.ClickException("submit-feedback is not implemented yet")


@cli.command()
def visualize_feedback():
    """Visualize feedback and adaptation stats."""
    raise click.ClickException("visualize-feedback is not implemented yet")


if __name__ == "__main__":
    cli()
