"""`multimind backend` — run every REST API from one process/port."""

from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _backend():
    # Backend needs the [gateway] extras; import at command time so
    # `multimind --help` and torch-free/core installs keep working
    try:
        import uvicorn

        from ..backend.app import BackendSettings, create_backend_app
    except ImportError as exc:
        raise click.ClickException(
            "The backend command requires the gateway extras. "
            "Install with: pip install 'multimind-sdk[gateway]'"
        ) from exc
    return uvicorn, BackendSettings, create_backend_app


@click.command()
@click.option("--host", default=None, help="Bind host (default 127.0.0.1)")
@click.option("--port", "-p", type=int, default=None, help="Bind port (default 8080)")
def backend(host: Optional[str], port: Optional[int]):
    """Serve every MultiMind REST API from one process and port.

    Mounts the API Gateway, RAG API, Unified multi-modal API, and Multi-model
    API as sub-applications, each keeping its own Swagger UI. The governance
    dashboard (`multimind dashboard`) and guard proxy (`multimind serve`)
    remain separate processes.
    """
    uvicorn, BackendSettings, create_backend_app = _backend()
    settings = BackendSettings.from_env(host=host, port=port)
    app = create_backend_app(settings)

    base = f"http://{settings.host}:{settings.port}"
    table = Table(show_header=True, header_style="bold")
    table.add_column("Service")
    table.add_column("Swagger")
    for service in app.state.services:
        table.add_row(service["label"], f"{base}{service['docs']}")
    for item in app.state.unavailable:
        console.print(f"[yellow]Not mounted:[/yellow] {item['label']} — {item['reason']}")

    console.print(
        Panel.fit(
            f"[bold]Backend:[/bold] [green]{base}[/green]\n"
            f"[bold]Index:[/bold]   {base}/\n\n"
            "Dashboard (`multimind dashboard`) and guard proxy (`multimind "
            "serve`) run as separate processes.",
            title="MultiMind Backend",
            border_style="cyan",
        )
    )
    console.print(table)
    uvicorn.run(app, host=settings.host, port=settings.port)
