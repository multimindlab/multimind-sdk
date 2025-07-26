"""
CLI interface for the MultiModelWrapper.
"""

import click
import asyncio
import json
from typing import Optional, List
from ..models.factory import ModelFactory
from ..models.multi_model import MultiModelWrapper

@click.group()
def cli():
    """Multi-model CLI interface with config/feedback commands."""
    pass

@cli.command()
@click.option('--primary-model', default='openai', help='Primary model to use')
@click.option('--fallback-models', multiple=True, help='Fallback models to use')
@click.option('--model-weights', help='JSON string of model weights')
@click.option('--temperature', default=0.7, help='Temperature for generation')
@click.option('--max-tokens', type=int, help='Maximum tokens to generate')
@click.argument('prompt')
def generate(primary_model: str, fallback_models: List[str], model_weights: Optional[str],
            temperature: float, max_tokens: Optional[int], prompt: str):
    """Generate text using the multi-model wrapper."""
    async def run():
        factory = ModelFactory()
        weights = json.loads(model_weights) if model_weights else None
        
        multi_model = MultiModelWrapper(
            model_factory=factory,
            primary_model=primary_model,
            fallback_models=list(fallback_models),
            model_weights=weights
        )
        
        response = await multi_model.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        click.echo(response)
    
    asyncio.run(run())

@cli.command()
@click.option('--primary-model', default='openai', help='Primary model to use')
@click.option('--fallback-models', multiple=True, help='Fallback models to use')
@click.option('--model-weights', help='JSON string of model weights')
@click.option('--temperature', default=0.7, help='Temperature for generation')
@click.option('--max-tokens', type=int, help='Maximum tokens to generate')
@click.option('--system-message', default='You are a helpful AI assistant.', help='System message')
@click.argument('user_message')
def chat(primary_model: str, fallback_models: List[str], model_weights: Optional[str],
         temperature: float, max_tokens: Optional[int], system_message: str, user_message: str):
    """Generate chat completion using the multi-model wrapper."""
    async def run():
        factory = ModelFactory()
        weights = json.loads(model_weights) if model_weights else None
        
        multi_model = MultiModelWrapper(
            model_factory=factory,
            primary_model=primary_model,
            fallback_models=list(fallback_models),
            model_weights=weights
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        response = await multi_model.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        click.echo(response)
    
    asyncio.run(run())

@cli.command()
@click.option('--primary-model', default='openai', help='Primary model to use')
@click.option('--fallback-models', multiple=True, help='Fallback models to use')
@click.option('--model-weights', help='JSON string of model weights')
@click.argument('text')
def embeddings(primary_model: str, fallback_models: List[str], model_weights: Optional[str], text: str):
    """Generate embeddings using the multi-model wrapper."""
    async def run():
        factory = ModelFactory()
        weights = json.loads(model_weights) if model_weights else None
        
        multi_model = MultiModelWrapper(
            model_factory=factory,
            primary_model=primary_model,
            fallback_models=list(fallback_models),
            model_weights=weights
        )
        
        embeddings = await multi_model.embeddings(text)
        click.echo(json.dumps(embeddings))
    
    asyncio.run(run())

@cli.command()
def list_strategies():
    """List available ensemble, fusion, and router strategies."""
    click.echo("Ensemble: weighted_voting, confidence_cascade, parallel_voting, majority_voting, rank_based")
    click.echo("Fusion: weighted_sum, neural_fusion, multi_layer_fusion, attention_fusion, transformer_fusion")
    click.echo("Router: cost, latency, hybrid, pareto, learning, deep_rl")

@cli.command()
@click.option('--strategy-type', type=click.Choice(['ensemble', 'fusion', 'router']), required=True)
@click.option('--strategy', required=True, help='Strategy name to set')
def set_strategy(strategy_type, strategy):
    """Set the active strategy for ensemble, fusion, or router."""
    # This is a placeholder; in a real system, this would update config files or a running service
    click.echo(f"Set {strategy_type} strategy to: {strategy}")

@cli.command()
def show_config():
    """Show current configuration for ensemble, fusion, router, and memory."""
    # Placeholder: In a real system, would load/display actual config
    click.echo("Current configuration:")
    click.echo("Ensemble: weighted_voting (adaptive)")
    click.echo("Fusion: weighted_sum (dynamic)")
    click.echo("Router: hybrid (feedback)")
    click.echo("Memory: summary (LLM-based compression)")

@cli.command()
@click.option('--strategy-type', type=click.Choice(['ensemble', 'fusion', 'router']), required=True)
@click.option('--param', required=True, help='Parameter name (e.g., weight, threshold)')
@click.option('--value', required=True, help='Parameter value (JSON or string)')
def set_param(strategy_type, param, value):
    """Set a parameter for a strategy (e.g., weight, threshold)."""
    click.echo(f"Set {strategy_type} parameter {param} to {value}")

@cli.command()
@click.option('--strategy-type', type=click.Choice(['ensemble', 'fusion', 'router']), required=True)
@click.option('--feedback', required=True, help='Feedback value (e.g., success, fail, numeric)')
def submit_feedback(strategy_type, feedback):
    """Submit feedback for a strategy (e.g., after a request)."""
    click.echo(f"Feedback for {strategy_type}: {feedback}")

@cli.command()
def visualize_feedback():
    """Visualize feedback and adaptation stats (placeholder)."""
    click.echo("Feedback stats: (placeholder)")
    click.echo("Ensemble: success=10, fail=2, avg_weight=0.7")
    click.echo("Fusion: avg_attention=0.5")
    click.echo("Router: success=8, fail=1, avg_reward=0.8")

if __name__ == '__main__':
    cli() 