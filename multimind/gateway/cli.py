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
from .monitoring import monitor
from .chat import chat_manager, ChatSession

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
        self.current_session: Optional[ChatSession] = None
    
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
    
    async def show_metrics(self, model: Optional[str] = None) -> None:
        """Display metrics and health status for models"""
        try:
            metrics = await monitor.get_metrics(model)
            
            # Create metrics table
            metrics_table = Table(title="Model Metrics")
            metrics_table.add_column("Model", style="cyan")
            metrics_table.add_column("Requests", style="green")
            metrics_table.add_column("Success Rate", style="green")
            metrics_table.add_column("Avg Response Time", style="yellow")
            metrics_table.add_column("Total Tokens", style="blue")
            metrics_table.add_column("Total Cost", style="red")
            
            for model_name, data in metrics.items():
                m = data["metrics"]
                success_rate = (m.successful_requests / m.total_requests * 100 
                              if m.total_requests > 0 else 0)
                
                metrics_table.add_row(
                    model_name,
                    str(m.total_requests),
                    f"{success_rate:.1f}%",
                    f"{m.avg_response_time:.2f}s",
                    str(m.total_tokens),
                    f"${m.total_cost:.4f}"
                )
            
            self.console.print(metrics_table)
            
            # Create health table
            health_table = Table(title="Model Health")
            health_table.add_column("Model", style="cyan")
            health_table.add_column("Status", style="green")
            health_table.add_column("Latency", style="yellow")
            health_table.add_column("Last Check", style="blue")
            
            for model_name, health in monitor.health.items():
                status = "✅" if health.is_healthy else "❌"
                latency = f"{health.latency_ms:.0f}ms" if health.latency_ms else "N/A"
                
                health_table.add_row(
                    model_name,
                    status,
                    latency,
                    health.last_check.strftime("%Y-%m-%d %H:%M:%S")
                )
            
            self.console.print(health_table)
            
        except Exception as e:
            logger.error(f"Error showing metrics: {e}")
            self.console.print(f"[red]Error: {str(e)}[/red]")
    
    async def check_health(self, model: Optional[str] = None) -> None:
        """Check health of models"""
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Checking model health...", total=None)
                
                if model:
                    handler = get_model_handler(model)
                    health = await monitor.check_health(model, handler)
                    status = {model: health}
                else:
                    # Check all configured models
                    status = {}
                    for model_name in config.validate().keys():
                        if config.validate()[model_name]:
                            handler = get_model_handler(model_name)
                            health = await monitor.check_health(model_name, handler)
                            status[model_name] = health
                
                progress.update(task, completed=True)
            
            # Display results
            for model_name, health in status.items():
                status_str = "✅" if health.is_healthy else "❌"
                latency = f"{health.latency_ms:.0f}ms" if health.latency_ms else "N/A"
                
                self.console.print(Panel(
                    f"Status: {status_str}\n"
                    f"Latency: {latency}\n"
                    f"Last Check: {health.last_check.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Error: {health.error_message or 'None'}",
                    title=f"{model_name} Health Check",
                    border_style="green" if health.is_healthy else "red"
                ))
        
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            self.console.print(f"[red]Error: {str(e)}[/red]")
    
    async def list_sessions(self) -> None:
        """List all chat sessions"""
        try:
            sessions = chat_manager.list_sessions()
            
            if not sessions:
                self.console.print("[yellow]No active sessions found[/yellow]")
                return
            
            table = Table(title="Chat Sessions")
            table.add_column("Session ID", style="cyan")
            table.add_column("Model", style="green")
            table.add_column("Created", style="blue")
            table.add_column("Updated", style="blue")
            table.add_column("Messages", style="yellow")
            
            for session in sessions:
                table.add_row(
                    session["session_id"],
                    session["model"],
                    session["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
                    session["updated_at"].strftime("%Y-%m-%d %H:%M:%S"),
                    str(session["message_count"])
                )
            
            self.console.print(table)
        
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            self.console.print(f"[red]Error: {str(e)}[/red]")
    
    async def load_session(self, session_id: str) -> None:
        """Load a chat session"""
        try:
            session = chat_manager.get_session(session_id)
            if not session:
                session = chat_manager.load_session(session_id)
            if not session:
                self.console.print(f"[red]Session {session_id} not found[/red]")
                return
            
            self.current_session = session
            self.console.print(f"[green]Loaded session {session_id}[/green]")
            self.console.print(f"Model: {session.model}")
            self.console.print(f"Messages: {len(session.messages)}")
            
            # Show recent messages
            if session.messages:
                self.console.print("\n[bold]Recent Messages:[/bold]")
                for msg in session.messages[-5:]:
                    self.console.print(Panel(
                        msg.content,
                        title=f"{msg.role} ({msg.model})",
                        border_style="blue"
                    ))
        
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            self.console.print(f"[red]Error: {str(e)}[/red]")
    
    async def save_session(self, session_id: Optional[str] = None) -> None:
        """Save current chat session"""
        try:
            if not session_id and self.current_session:
                session_id = self.current_session.session_id
            
            if not session_id:
                self.console.print("[red]No session to save[/red]")
                return
            
            if chat_manager.save_session(session_id):
                self.console.print(f"[green]Saved session {session_id}[/green]")
            else:
                self.console.print(f"[red]Failed to save session {session_id}[/red]")
        
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            self.console.print(f"[red]Error: {str(e)}[/red]")
    
    async def delete_session(self, session_id: str) -> None:
        """Delete a chat session"""
        try:
            if chat_manager.delete_session(session_id):
                self.console.print(f"[green]Deleted session {session_id}[/green]")
                if self.current_session and self.current_session.session_id == session_id:
                    self.current_session = None
            else:
                self.console.print(f"[red]Session {session_id} not found[/red]")
        
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
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

@cli.command()
@click.option("--model", "-m", help="Specific model to show metrics for")
def metrics(model: Optional[str]):
    """Show metrics and health status for models"""
    cli = MultiMindCLI()
    cli.validate_config()
    asyncio.run(cli.show_metrics(model))

@cli.command()
@click.option("--model", "-m", help="Specific model to check")
def health(model: Optional[str]):
    """Check health of models"""
    cli = MultiMindCLI()
    cli.validate_config()
    asyncio.run(cli.check_health(model))

@cli.command()
def sessions():
    """List all chat sessions"""
    cli = MultiMindCLI()
    cli.validate_config()
    asyncio.run(cli.list_sessions())

@cli.command()
@click.argument("session_id")
def load(session_id: str):
    """Load a chat session"""
    cli = MultiMindCLI()
    cli.validate_config()
    asyncio.run(cli.load_session(session_id))

@cli.command()
@click.argument("session_id")
def save(session_id: str):
    """Save a chat session"""
    cli = MultiMindCLI()
    cli.validate_config()
    asyncio.run(cli.save_session(session_id))

@cli.command()
@click.argument("session_id")
def delete(session_id: str):
    """Delete a chat session"""
    cli = MultiMindCLI()
    cli.validate_config()
    asyncio.run(cli.delete_session(session_id))

if __name__ == "__main__":
    cli() 