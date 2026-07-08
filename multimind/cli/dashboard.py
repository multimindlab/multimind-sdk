"""`multimind dashboard` — local AI governance dashboard."""

from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


def _dashboard():
    # Dashboard needs the [gateway] extras; import at command time so
    # `multimind --help` and torch-free/core installs keep working
    try:
        import uvicorn

        from ..dashboard.server import DashboardSettings, create_dashboard_app
    except ImportError as exc:
        raise click.ClickException(
            "The dashboard command requires the gateway extras. "
            "Install with: pip install 'multimind-sdk[gateway]'"
        ) from exc
    return uvicorn, DashboardSettings, create_dashboard_app


@click.command()
@click.option("--host", default=None, help="Bind host (default 127.0.0.1)")
@click.option("--port", "-p", type=int, default=None, help="Bind port (default 8501)")
@click.option(
    "--audit-log", default=None, help="Guard/proxy JSONL audit trail (default audit.jsonl)"
)
@click.option("--costs-log", default=None, help="Cost tracker JSONL log (default costs.jsonl)")
@click.option(
    "--project",
    "project_path",
    default=None,
    help="Project directory for the shadow-AI inventory scan (default .)",
)
@click.option(
    "--guardrails",
    "guardrails_path",
    default=None,
    help="Guardrails JSON file to read/author (default guardrails.json)",
)
def dashboard(
    host: Optional[str],
    port: Optional[int],
    audit_log: Optional[str],
    costs_log: Optional[str],
    project_path: Optional[str],
    guardrails_path: Optional[str],
):
    """Serve the local AI governance dashboard.

    A read-only web UI over local MultiMind artifacts: PII audit trail,
    spend and chargeback, shadow-AI inventory, and no-code guardrail
    authoring for `multimind serve --config`. Nothing leaves the machine.
    """
    uvicorn, DashboardSettings, create_dashboard_app = _dashboard()
    settings = DashboardSettings.from_env(
        host=host,
        port=port,
        audit_log=audit_log,
        costs_log=costs_log,
        project_path=project_path,
        guardrails_path=guardrails_path,
    )
    app = create_dashboard_app(settings)
    lines = [
        f"[bold]Dashboard:[/bold]   [green]http://{settings.host}:{settings.port}[/green]",
        f"[bold]Audit log:[/bold]   {settings.audit_log}",
        f"[bold]Costs log:[/bold]   {settings.costs_log}",
        f"[bold]Project:[/bold]     {settings.project_path}",
        f"[bold]Guardrails:[/bold]  {settings.guardrails_path}",
        "",
        "Read-only over local artifacts; no auth. Keep it on localhost or",
        "put it behind a reverse proxy for team access.",
    ]
    console.print(
        Panel("\n".join(lines), title="MultiMind Governance Dashboard", border_style="cyan")
    )
    uvicorn.run(app, host=settings.host, port=settings.port)
