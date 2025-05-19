"""
Command-line interface for the MultiMind Gateway
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
import rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

from .config import config
from .models import ModelResponse, get_model_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()

class MultiMindCLI:
    """Command-line interface for MultiMind Gateway"""
    
    def __init__(self):
        self.console = Console()
        self.chat_history: List[Dict] = []
    
    def validate_config(self) -> None:
        """Validate the configuration and show status"""
        status = config.validate()
        
        table = Table(title="Model Configuration Status")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("API Key/Base", style="yellow")
        
        for model, is_valid in status.items():
            model_config = config.get_model_config(model)
            status_str = "✅" if is_valid else "❌"
            key_info = "Configured" if is_valid else "Missing"
            table.add_row(model, status_str, key_info)
        
        self.console.print(table)
        
        if not any(status.values()):
            self.console.print("[red]No models are properly configured![/red]")
            self.console.print("Please set up your API keys in the .env file.")
            sys.exit(1)
    
    async def chat(self, model_name: str, prompt: Optional[str] = None) -> None:
        """Interactive chat session with a model"""
        try:
            handler = get_model_handler(model_name)
            
            if prompt:
                # Single message mode
                response = await handler.generate(prompt)
                self.console.print(Panel(response.content, title=f"{model_name} Response"))
                return
            
            # Interactive chat mode
            self.console.print(f"[bold green]Starting chat with {model_name}[/bold green]")
            self.console.print("Type 'exit' to quit, 'clear' to clear history")
            
            while True:
                try:
                    user_input = click.prompt("\nYou", type=str)
                    
                    if user_input.lower() == 'exit':
                        break
                    elif user_input.lower() == 'clear':
                        self.chat_history = []
                        self.console.print("[yellow]Chat history cleared[/yellow]")
                        continue
                    
                    with Progress() as progress:
                        task = progress.add_task("[cyan]Thinking...", total=None)
                        response = await handler.chat(
                            [{"role": "user", "content": user_input}]
                        )
                        progress.update(task, completed=True)
                    
                    self.chat_history.append({
                        "role": "user",
                        "content": user_input,
                        "model": model_name
                    })
                    self.chat_history.append({
                        "role": "assistant",
                        "content": response.content,
                        "model": model_name
                    })
                    
                    self.console.print(Panel(
                        response.content,
                        title=f"{model_name} Response",
                        border_style="green"
                    ))
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error during chat: {str(e)}")
                    self.console.print(f"[red]Error: {str(e)}[/red]")
        
        except Exception as e:
            logger.error(f"Error initializing chat: {str(e)}")
            self.console.print(f"[red]Error: {str(e)}[/red]")
    
    async def compare(self, prompt: str, models: List[str]) -> None:
        """Compare responses from multiple models"""
        try:
            responses: Dict[str, ModelResponse] = {}
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Comparing models...", total=len(models))
                
                for model in models:
                    try:
                        handler = get_model_handler(model)
                        response = await handler.generate(prompt)
                        responses[model] = response
                    except Exception as e:
                        logger.error(f"Error with {model}: {str(e)}")
                        responses[model] = ModelResponse(
                            content=f"Error: {str(e)}",
                            model=model
                        )
                    progress.update(task, advance=1)
            
            # Display results
            for model, response in responses.items():
                self.console.print(Panel(
                    response.content,
                    title=f"{model} Response",
                    border_style="green"
                ))
                
                if response.usage:
                    usage_table = Table(title=f"{model} Usage")
                    for key, value in response.usage.items():
                        usage_table.add_row(key, str(value))
                    self.console.print(usage_table)
        
        except Exception as e:
            logger.error(f"Error during comparison: {str(e)}")
            self.console.print(f"[red]Error: {str(e)}[/red]")

@click.group()
def cli():
    """MultiMind Gateway CLI - Unified interface for multiple AI models"""
    pass

@cli.command()
@click.option("--model", "-m", default=config.default_model, help="Model to use")
@click.option("--prompt", "-p", help="Single prompt to send (optional)")
def chat(model: str, prompt: Optional[str]):
    """Start an interactive chat session with a model"""
    cli = MultiMindCLI()
    cli.validate_config()
    asyncio.run(cli.chat(model, prompt))

@cli.command()
@click.argument("prompt")
@click.option("--models", "-m", multiple=True, help="Models to compare")
def compare(prompt: str, models: List[str]):
    """Compare responses from multiple models"""
    if not models:
        models = ["openai", "anthropic", "ollama"]
    
    cli = MultiMindCLI()
    cli.validate_config()
    asyncio.run(cli.compare(prompt, models))

@cli.command()
def status():
    """Show configuration status for all models"""
    cli = MultiMindCLI()
    cli.validate_config()

if __name__ == "__main__":
    cli() 