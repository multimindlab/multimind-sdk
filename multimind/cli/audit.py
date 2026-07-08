"""
Command-line interface for MultiMind AI usage audits.
"""

import json
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def audit():
    """MultiMind AI usage audit commands."""
    pass


@audit.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False), default=".")
@click.option(
    "--include-env",
    is_flag=True,
    help="Also report which known AI env keys are set (names only, never values)",
)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON output")
def scan(path: str, include_env: bool, as_json: bool):
    """Scan a project directory for AI usage (shadow-AI inventory).

    Exits with code 1 if hardcoded AI API keys are found.
    """
    from ..observability.ai_inventory import scan_project

    report = scan_project(path, include_env=include_env)

    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        if not report.findings:
            console.print("[green]No AI usage found.[/green]")
        else:
            table = Table(title=f"AI inventory ({len(report.findings)} findings)")
            table.add_column("Provider", style="cyan")
            table.add_column("Kind")
            table.add_column("Evidence")
            table.add_column("Data flow")
            table.add_column("Location")
            for f in sorted(report.findings, key=lambda f: (f.provider, f.kind, f.file or "")):
                location = f"{f.file}:{f.line}" if f.file and f.line else (f.file or "-")
                table.add_row(f.provider, f.kind, f.evidence, f.data_flow or "-", location)
            console.print(table)

        risks = report.risks()
        if risks["external_data_flow_providers"]:
            console.print(
                "[yellow]External data flow providers:[/yellow] "
                + ", ".join(risks["external_data_flow_providers"])
            )
        if report.hardcoded_keys:
            console.print(
                f"[red]Hardcoded API key(s) found: {len(report.hardcoded_keys)} "
                "(key values are never recorded)[/red]"
            )
        if report.skipped_files:
            console.print(
                f"[yellow]Skipped {report.skipped_files} file(s) "
                "(size/count caps or read errors); results may be incomplete.[/yellow]"
            )
        console.print(
            f"Scanned {report.scanned_files} file(s) under {report.root}",
        )

    if report.hardcoded_keys:
        sys.exit(1)


@audit.command()
@click.option(
    "--log",
    "log_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Rebuild the tracker from a persisted JSONL cost log (e.g. costs.jsonl)",
)
@click.option("--period", help="Filter by ISO timestamp prefix, e.g. 2026-07")
def costs(log_path: str, period: str):
    """Per-tag chargeback report from the session tracker or a JSONL log."""
    from ..observability.cost_tracker import get_default_tracker, load_tracker

    tracker = load_tracker(log_path) if log_path else get_default_tracker()
    click.echo(tracker.report_chargeback(period=period))


def main():
    """Main entry point for CLI."""
    audit()
