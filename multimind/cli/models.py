"""
Model management commands for MultiMind CLI
"""

import asyncio
import os
import sys
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table


def _gateway():
    # Gateway needs the [gateway] extras; import at command time so core
    # installs can still run `multimind models list`
    try:
        from ..gateway.models import get_model_handler
        from ..gateway.monitoring import monitor
    except ImportError as exc:
        raise click.ClickException(
            "This command requires the gateway extras. "
            "Install with: pip install 'multimind-sdk[gateway]'"
        ) from exc
    return get_model_handler, monitor


console = Console()

# Env vars required per gateway model handler (empty tuple = local, no key needed)
_ENV_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "ollama": (),
    "huggingface": (),
}

# Env vars per ModelFactory provider (empty tuple = local, no key needed)
_PROVIDER_ENV_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "claude": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "ollama": (),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
}


def _missing_api_key(model: str) -> Optional[str]:
    """Return the missing env var name for a model, or None if configured."""
    env_vars = _ENV_KEYS.get(model.lower())
    if not env_vars:
        return None
    if any(os.getenv(v) for v in env_vars):
        return None
    return env_vars[0]


def _require_api_key(model: str) -> None:
    missing = _missing_api_key(model)
    if missing:
        console.print(f"[red]{missing} not set. Export it to use the '{model}' model.[/red]")
        sys.exit(1)


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

    responses = {}

    get_model_handler, _ = _gateway()
    with Progress() as progress:
        task = progress.add_task("[cyan]Comparing models...", total=len(models))

        for model in models:
            missing = _missing_api_key(model)
            if missing:
                console.print(f"[yellow]Skipping {model}: {missing} not set[/yellow]")
                progress.update(task, advance=1)
                continue
            try:
                handler = get_model_handler(model)
                response = asyncio.run(handler.generate(prompt))
                responses[model] = response
            except Exception as e:
                console.print(f"[red]Error with {model}: {str(e)}[/red]")
            progress.update(task, advance=1)

    if not responses:
        console.print("[red]No model produced a response.[/red]")
        sys.exit(1)

    # Display results
    for model, response in responses.items():
        console.print(Panel(response.content, title=f"{model} Response", border_style="green"))

        if response.usage:
            usage_table = Table(title=f"{model} Usage")
            for key, value in response.usage.items():
                usage_table.add_row(key, str(value))
            console.print(usage_table)


@models.command()
@click.option("--model", "-m", help="Specific model to show metrics for")
def metrics(model: Optional[str]):
    """Show metrics and health status for models"""
    try:
        _, monitor = _gateway()
        metrics = asyncio.run(monitor.get_metrics(model))
        if model:
            # get_metrics returns {"metrics": ..., "health": ...} for a single model
            metrics = {model: metrics}

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
            success_rate = (
                m.successful_requests / m.total_requests * 100 if m.total_requests > 0 else 0
            )

            metrics_table.add_row(
                model_name,
                str(m.total_requests),
                f"{success_rate:.1f}%",
                f"{m.avg_response_time:.2f}s",
                str(m.total_tokens),
                f"${m.total_cost:.4f}",
            )

        console.print(metrics_table)

        # Create health table
        health_table = Table(title="Model Health")
        health_table.add_column("Model", style="cyan")
        health_table.add_column("Status", style="green")
        health_table.add_column("Latency", style="yellow")
        health_table.add_column("Last Check", style="blue")

        _, monitor = _gateway()
        for model_name, health in monitor.health.items():
            status = "✅" if health.is_healthy else "❌"
            latency = f"{health.latency_ms:.0f}ms" if health.latency_ms else "N/A"

            health_table.add_row(
                model_name, status, latency, health.last_check.strftime("%Y-%m-%d %H:%M:%S")
            )

        console.print(health_table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@models.command()
@click.option("--model", "-m", help="Specific model to check")
def health(model: Optional[str]):
    """Check health of models"""
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Checking model health...", total=None)

            if model:
                missing = _missing_api_key(model)
                if missing:
                    console.print(
                        f"[red]{missing} not set. Export it to check '{model}' health.[/red]"
                    )
                    sys.exit(1)
                get_model_handler, monitor = _gateway()
                handler = get_model_handler(model)
                health = asyncio.run(monitor.check_health(model, handler))
                status = {model: health}
            else:
                # Check every gateway model whose API key is configured;
                # local models (ollama, huggingface) must be checked explicitly with --model
                status = {}
                for model_name, env_vars in _ENV_KEYS.items():
                    if not env_vars or not any(os.getenv(v) for v in env_vars):
                        continue
                    try:
                        get_model_handler, monitor = _gateway()
                        handler = get_model_handler(model_name)
                        health = asyncio.run(monitor.check_health(model_name, handler))
                        status[model_name] = health
                    except Exception as e:
                        console.print(f"[red]Error checking {model_name}: {str(e)}[/red]")

            progress.update(task, completed=True)

        if not status:
            console.print(
                "[yellow]No models with configured API keys to check. "
                "Use --model to check a specific model.[/yellow]"
            )
            return

        # Display results
        for model_name, health in status.items():
            status_str = "✅" if health.is_healthy else "❌"
            latency = f"{health.latency_ms:.0f}ms" if health.latency_ms else "N/A"

            console.print(
                Panel(
                    f"Status: {status_str}\n"
                    f"Latency: {latency}\n"
                    f"Last Check: {health.last_check.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Error: {health.error_message or 'None'}",
                    title=f"{model_name} Health Check",
                    border_style="green" if health.is_healthy else "red",
                )
            )

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@models.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="List local fine-tuned models in this directory instead of registered providers.",
)
def list(output_dir):
    """List registered model providers or local fine-tuned models"""
    if output_dir:
        if not os.path.exists(output_dir):
            console.print(f"[yellow]No models found in {output_dir}[/yellow]")
            return
        local = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not local:
            console.print(f"[yellow]No models found in {output_dir}[/yellow]")
        else:
            console.print("[bold]Available models:[/bold]")
            for m in local:
                console.print(f"- {m}")
        return

    from ..models.factory import ModelFactory

    factory = ModelFactory()
    table = Table(title="Registered Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("API Key", style="green")
    table.add_column("Env Var", style="blue")

    for provider in sorted(factory._model_classes):
        env_vars = _PROVIDER_ENV_KEYS.get(provider, (f"{provider.upper()}_API_KEY",))
        if not env_vars:
            key_status = "not required (local)"
            env_display = "-"
        else:
            configured = next((v for v in env_vars if os.getenv(v)), None)
            key_status = "configured" if configured else "not set"
            env_display = configured or " or ".join(env_vars)
        table.add_row(provider, key_status, env_display)

    console.print(table)


@models.command()
@click.option("--model", "-m", type=str, help="Model name to download (e.g., bert-base-uncased).")
def download(model):
    """Download a pretrained or fine-tuned model"""
    if not model:
        model = click.prompt("Model name to download")
    try:
        from transformers import AutoModelForCausalLM

        AutoModelForCausalLM.from_pretrained(model)
        console.print(f"[green]Downloaded model: {model}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@models.command()
@click.option("--model", "-m", type=click.Path(exists=True), help="Path to model to export.")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["onnx", "torchscript"], case_sensitive=False),
    help="Export format.",
)
@click.option("--output", "-o", type=click.Path(), help="Output path for exported model.")
def export(model, format, output):
    """Export a model to ONNX or TorchScript format"""
    if not model:
        model = click.prompt("Model path", type=click.Path(exists=True))
    if not format:
        format = click.prompt(
            "Export format (onnx/torchscript)", type=click.Choice(["onnx", "torchscript"])
        )
    if not output:
        output = click.prompt("Output path", type=click.Path())
    try:
        from transformers import AutoModelForCausalLM

        model_obj = AutoModelForCausalLM.from_pretrained(model)
        if format == "onnx":
            import torch

            dummy_input = torch.randint(0, 100, (1, 16))
            torch.onnx.export(model_obj, dummy_input, output)
        elif format == "torchscript":
            import torch

            scripted = torch.jit.script(model_obj)
            scripted.save(output)
        console.print(f"[green]Exported {model} to {format} at {output}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@models.command()
@click.option("--model", "-m", type=click.Path(), help="Path to model to delete.")
def delete(model):
    """Delete a local fine-tuned model"""
    if not model:
        model = click.prompt("Model path to delete", type=click.Path())
    if click.confirm(f"Are you sure you want to delete {model}?"):
        try:
            if os.path.isdir(model):
                import shutil

                shutil.rmtree(model)
            else:
                os.remove(model)
            console.print(f"[green]Deleted model: {model}[/green]")
        except Exception as e:
            console.print(f"[red]Error deleting model: {str(e)}[/red]")
            sys.exit(1)
    else:
        console.print("[yellow]Aborted.[/yellow]")
