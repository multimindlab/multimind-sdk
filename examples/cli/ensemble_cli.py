"""
CLI interface for the MultiMind Ensemble system.
"""

import click
import asyncio
import json
from pathlib import Path
from typing import List, Optional

from multimind import Router, TaskType, AdvancedEnsemble, EnsembleMethod

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
    async def run():
        router = Router()
        ensemble = AdvancedEnsemble(router)
        
        # Get results from all providers
        results = await asyncio.gather(*[
            router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider=provider,
                model="gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "mistral"
            )
            for provider in providers
        ])
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod(method),
            task_type=TaskType.TEXT_GENERATION
        )
        
        # Format output
        output_data = {
            "result": combined_result.result.result,
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
    async def run():
        router = Router()
        ensemble = AdvancedEnsemble(router)
        
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
        
        # Get reviews from all providers
        results = await asyncio.gather(*[
            router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider=provider,
                model="gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "codellama"
            )
            for provider in providers
        ])
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.RANK_BASED,
            task_type=TaskType.TEXT_GENERATION
        )
        
        # Format output
        output_data = {
            "review": combined_result.result.result,
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
@click.argument('image', type=click.Path(exists=True))
@click.option('--providers', '-p', multiple=True, default=['openai', 'anthropic'],
              help='List of providers to use')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def analyze_image(image: str, providers: List[str], output: Optional[str]):
    """Analyze image using ensemble of models."""
    async def run():
        router = Router()
        ensemble = AdvancedEnsemble(router)
        
        # Read image file
        with open(image, 'rb') as f:
            image_data = f.read()
        
        # Get analysis from all providers
        results = await asyncio.gather(*[
            router.route(
                TaskType.IMAGE_ANALYSIS,
                image_data,
                provider=provider,
                model="gpt-4-vision-preview" if provider == "openai" else "claude-3-sonnet"
            )
            for provider in providers
        ])
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.CONFIDENCE_CASCADE,
            task_type=TaskType.IMAGE_ANALYSIS,
            confidence_threshold=0.7
        )
        
        # Format output
        output_data = {
            "analysis": combined_result.result.result,
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
@click.option('--providers', '-p', multiple=True, default=['openai', 'huggingface'],
              help='List of providers to use')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def embed(text: str, providers: List[str], output: Optional[str]):
    """Generate embeddings using ensemble of models."""
    async def run():
        router = Router()
        ensemble = AdvancedEnsemble(router)
        
        # Get embeddings from all providers
        results = await asyncio.gather(*[
            router.route(
                TaskType.EMBEDDINGS,
                text,
                provider=provider,
                model="text-embedding-ada-002" if provider == "openai" else "sentence-transformers/all-MiniLM-L6-v2"
            )
            for provider in providers
        ])
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.EMBEDDINGS,
            weights={
                "openai": 0.6,
                "huggingface": 0.4
            }
        )
        
        # Format output
        output_data = {
            "embedding": combined_result.result.result.tolist(),
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