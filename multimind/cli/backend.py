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

        from ..backend.app import SERVICE_MOUNTS, BackendSettings, create_backend_app
    except ImportError as exc:
        raise click.ClickException(
            "The backend command requires the gateway extras. "
            "Install with: pip install 'multimind-sdk[gateway]'"
        ) from exc
    return uvicorn, BackendSettings, create_backend_app, SERVICE_MOUNTS


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
    uvicorn, BackendSettings, create_backend_app, SERVICE_MOUNTS = _backend()
    settings = BackendSettings.from_env(host=host, port=port)
    app = create_backend_app(settings)

    base = f"http://{settings.host}:{settings.port}"
    table = Table(show_header=True, header_style="bold")
    table.add_column("Service")
    table.add_column("Swagger")
    for mount_path, _module_path, _attr, label in SERVICE_MOUNTS:
        table.add_row(label, f"{base}{mount_path}/docs")

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
