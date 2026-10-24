"""`multimind backend` — run every REST API plus the dashboard UI from one process/port."""

from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


def _backend():
    # Backend needs the [gateway] extras; import at command time so
    # `multimind --help` and torch-free/core installs keep working
    try:
        import uvicorn

        from ..server.platform import create_platform_app
    except ImportError as exc:
        raise click.ClickException(
            "The backend command requires the gateway extras. "
            "Install with: pip install 'multimind-sdk[gateway]'"
        ) from exc
    return uvicorn, create_platform_app


@click.command()
@click.option("--host", default=None, help="Bind host (default 127.0.0.1)")
@click.option("--port", "-p", type=int, default=None, help="Bind port (default 8080)")
def backend(host: Optional[str], port: Optional[int]):
    """Serve every supported MultiMind REST route and the governance dashboard UI from one port.

    The dashboard page is served at `/` (same origin as `/v1/*` and `/api/*`,
    so no CORS or backend URL setup is needed); the stable API docs are at
    `GET /docs`. The older four-app mount aggregator remains available as
    multimind.backend.app.create_backend_app.
    """
    import os

    uvicorn, create_platform_app = _backend()
    bind_host = host or os.getenv("MULTIMIND_BACKEND_HOST", "127.0.0.1")
    bind_port = port or int(os.getenv("MULTIMIND_BACKEND_PORT", "8080"))
    app = create_platform_app()
    base = f"http://{bind_host}:{bind_port}"
    console.print(
        Panel.fit(
            f"[bold]API:[/bold]     [green]{base}[/green]\n"
            f"[bold]UI:[/bold]      {base}/\n"
            f"[bold]Swagger:[/bold] {base}/docs\n"
            f"[bold]Health:[/bold]  {base}/health",
            title="MultiMind API",
            border_style="cyan",
        )
    )
    uvicorn.run(app, host=bind_host, port=bind_port)
