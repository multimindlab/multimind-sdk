"""
Model management commands for MultiMind CLI
"""

import asyncio
import click
import os
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

from ..core.models import ModelResponse
from ..gateway.models import get_model_handler
from ..gateway.monitoring import monitor
from ..gateway.config import config

console = Console()

@click.group()
def models():
    """Model management commands"""
    pass

@models.command()
@click.argument("prompt")
@click.option("--models", "-m", multiple=True, help="Models to compare")
def compare(prompt: str, models: List[str]):
    """Compare responses from multiple models"""
    if not models:
        models = ["openai", "anthropic", "ollama"]

    try:
        responses = {}

        with Progress() as progress:
            task = progress.add_task("[cyan]Comparing models...", total=len(models))

            for model in models:
                try:
                    handler = get_model_handler(model)
                    response = asyncio.run(handler.generate(prompt))
                    responses[model] = response
                except Exception as e:
                    console.print(f"[red]Error with {model}: {str(e)}[/red]")
                progress.update(task, advance=1)

        # Display results
        for model, response in responses.items():
            console.print(Panel(
                response.content,
                title=f"{model} Response",
                border_style="green"
            ))

            if response.usage:
                usage_table = Table(title=f"{model} Usage")
                for key, value in response.usage.items():
                    usage_table.add_row(key, str(value))
                console.print(usage_table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@models.command()
@click.option("--model", "-m", help="Specific model to show metrics for")
def metrics(model: Optional[str]):
    """Show metrics and health status for models"""
    try:
        metrics = asyncio.run(monitor.get_metrics(model))

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

        console.print(metrics_table)

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

        console.print(health_table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@models.command()
@click.option("--model", "-m", help="Specific model to check")
def health(model: Optional[str]):
    """Check health of models"""
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Checking model health...", total=None)

            if model:
                handler = get_model_handler(model)
                health = asyncio.run(monitor.check_health(model, handler))
                status = {model: health}
            else:
                # Check all configured models
                status = {}
                for model_name in config.validate().keys():
                    if config.validate()[model_name]:
                        handler = get_model_handler(model_name)
                        health = asyncio.run(monitor.check_health(model_name, handler))
                        status[model_name] = health

            progress.update(task, completed=True)

        # Display results
        for model_name, health in status.items():
            status_str = "✅" if health.is_healthy else "❌"
            latency = f"{health.latency_ms:.0f}ms" if health.latency_ms else "N/A"

            console.print(Panel(
                f"Status: {status_str}\n"
                f"Latency: {latency}\n"
                f"Last Check: {health.last_check.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Error: {health.error_message or 'None'}",
                title=f"{model_name} Health Check",
                border_style="green" if health.is_healthy else "red"
            ))

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@models.command()
@click.option('--output-dir', type=click.Path(), default='./output', help='Directory where models are saved.')
def list(output_dir):
    """List available or fine-tuned models"""
    try:
        if not os.path.exists(output_dir):
            console.print(f"[yellow]No models found in {output_dir}[/yellow]")
            return
        models = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not models:
            console.print(f"[yellow]No models found in {output_dir}[/yellow]")
        else:
            console.print("[bold]Available models:[/bold]")
            for m in models:
                console.print(f"- {m}")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@models.command()
@click.option('--model', '-m', type=str, help='Model name to download (e.g., bert-base-uncased).')
def download(model):
    """Download a pretrained or fine-tuned model"""
    if not model:
        model = click.prompt('Model name to download')
    try:
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained(model)
        console.print(f"[green]Downloaded model: {model}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@models.command()
@click.option('--model', '-m', type=click.Path(exists=True), help='Path to model to export.')
@click.option('--format', '-f', type=click.Choice(['onnx', 'torchscript'], case_sensitive=False), help='Export format.')
@click.option('--output', '-o', type=click.Path(), help='Output path for exported model.')
def export(model, format, output):
    """Export a model to ONNX or TorchScript format"""
    if not model:
        model = click.prompt('Model path', type=click.Path(exists=True))
    if not format:
        format = click.prompt('Export format (onnx/torchscript)', type=click.Choice(['onnx', 'torchscript']))
    if not output:
        output = click.prompt('Output path', type=click.Path())
    try:
        from transformers import AutoModelForCausalLM
        model_obj = AutoModelForCausalLM.from_pretrained(model)
        if format == 'onnx':
            import torch
            dummy_input = torch.randint(0, 100, (1, 16))
            torch.onnx.export(model_obj, dummy_input, output)
        elif format == 'torchscript':
            import torch
            scripted = torch.jit.script(model_obj)
            scripted.save(output)
        console.print(f"[green]Exported {model} to {format} at {output}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@models.command()
@click.option('--model', '-m', type=click.Path(), help='Path to model to delete.')
def delete(model):
    """Delete a local fine-tuned model"""
    if not model:
        model = click.prompt('Model path to delete', type=click.Path())
    if click.confirm(f'Are you sure you want to delete {model}?'):
        try:
            if os.path.isdir(model):
                import shutil
                shutil.rmtree(model)
            else:
                os.remove(model)
            console.print(f"[green]Deleted model: {model}[/green]")
        except Exception as e:
            console.print(f"[red]Error deleting model: {str(e)}[/red]")
    else:
        console.print("[yellow]Aborted.[/yellow]") 