"""`multimind serve` — one-command OpenAI-compatible compliance proxy."""

from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


def _proxy():
    # Proxy needs the [gateway] extras; import at command time so `multimind
    # --help` and torch-free/core installs keep working
    try:
        import uvicorn

        from ..gateway.guard_proxy import ProxySettings, create_app
    except ImportError as exc:
        raise click.ClickException(
            "The serve command requires the gateway extras. "
            "Install with: pip install 'multimind-sdk[gateway]'"
        ) from exc
    return uvicorn, ProxySettings, create_app


@click.command()
@click.option("--host", default=None, help="Bind host (default 127.0.0.1)")
@click.option("--port", "-p", type=int, default=None, help="Bind port (default 8400)")
@click.option(
    "--upstream",
    default=None,
    help="Named upstream provider: openai, groq, mistral, gemini, deepseek, ollama",
)
@click.option(
    "--upstream-base-url",
    default=None,
    help="Custom OpenAI-compatible upstream URL (overrides --upstream)",
)
@click.option(
    "--strategy",
    type=click.Choice(["mask", "hash", "remove"]),
    default=None,
    help="PII redaction strategy (default mask)",
)
@click.option(
    "--block-on",
    default=None,
    help="Comma-separated PII types that reject the request, e.g. ssn,credit_card",
)
@click.option("--budget", type=float, default=None, help="Max session spend in USD")
@click.option("--audit-log", default=None, help="Path for the JSONL audit trail")
@click.option(
    "--scan-output/--no-scan-output",
    default=None,
    help="Also redact PII in model output (default on)",
)
@click.option(
    "--config", default=None, help="Guardrails JSON file authored via `multimind dashboard`"
)
def serve(
    host: Optional[str],
    port: Optional[int],
    upstream: Optional[str],
    upstream_base_url: Optional[str],
    strategy: Optional[str],
    block_on: Optional[str],
    budget: Optional[float],
    audit_log: Optional[str],
    scan_output: Optional[bool],
    config: Optional[str],
):
    """Start the OpenAI-compatible compliance proxy.

    Point any existing OpenAI client's base_url at this proxy to get PII
    redaction, budget enforcement, and an audit trail with zero code changes.
    """
    uvicorn, ProxySettings, create_app = _proxy()
    try:
        kwargs = dict(
            host=host,
            port=port,
            upstream=upstream,
            upstream_base_url=upstream_base_url,
            strategy=strategy,
            block_on=block_on,
            budget=budget,
            audit_log=audit_log,
            scan_output=scan_output,
        )
        if config:
            settings = ProxySettings.from_file(config, **kwargs)
        else:
            settings = ProxySettings.from_env(**kwargs)
        app = create_app(settings)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    state = app.state.proxy
    base_url = f"http://{settings.host}:{settings.port}/v1"
    lines = [
        f"[bold]Upstream:[/bold]        {state.provider} ({state.base_url})",
        f"[bold]PII redaction:[/bold]   {settings.strategy} "
        f"(output scan {'on' if settings.scan_output else 'off'})",
        f"[bold]Block on:[/bold]        "
        f"{', '.join(settings.block_on) if settings.block_on else 'none'}",
        f"[bold]Budget:[/bold]          "
        f"{f'${settings.budget:.2f}' if settings.budget else 'unlimited'}",
        f"[bold]Audit log:[/bold]       {settings.audit_log or 'disabled'}",
        "",
        "Point your existing OpenAI client here (no other code changes):",
        f'[green]client = OpenAI(base_url="{base_url}", api_key="unused")[/green]',
    ]
    console.print(Panel("\n".join(lines), title="MultiMind Guard Proxy", border_style="cyan"))
    uvicorn.run(app, host=settings.host, port=settings.port)
