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
    """Multi-model CLI interface."""
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

if __name__ == '__main__':
    cli() 