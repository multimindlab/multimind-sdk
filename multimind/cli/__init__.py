"""
MultiMind CLI - Command Line Interface for MultiMind SDK
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .audit import audit
from .backend import backend
from .chat import chat
from .compliance import compliance
from .config import config
from .context_transfer import main as context_transfer_main
from .dashboard import dashboard
from .models import models
from .serve import serve

console = Console()


@click.group()
def cli():
    """MultiMind CLI - Command Line Interface for MultiMind SDK"""
    pass


# Register command groups
cli.add_command(compliance)
cli.add_command(chat)
cli.add_command(models)
cli.add_command(config)
cli.add_command(serve)
cli.add_command(audit)
cli.add_command(dashboard)
cli.add_command(backend)


def _run_convert():
    """Run the model conversion CLI; imported lazily because it needs torch."""
    import sys

    try:
        from .model_conversion_cli import main as convert_main
    except ImportError:
        print(
            "The 'convert' command requires the optional model-conversion "
            'dependencies (torch). Install them with: pip install "multimind-sdk[finetune]"',
            file=sys.stderr,
        )
        return 1
    return convert_main()


def main():
    """Main entry point for the CLI."""
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "convert":
            sys.argv.pop(1)  # Remove 'convert' from arguments
            sys.exit(_run_convert())
        elif sys.argv[1] == "context-transfer":
            sys.argv.pop(1)  # Remove 'context-transfer' from arguments
            sys.exit(context_transfer_main())

    # Everything else (chat, models, compliance, config, --help, ...) is
    # handled by the Click group.
    cli()


def __getattr__(name):
    # Backward-compatible lazy access; keeps torch out of plain CLI imports
    if name == "convert_main":
        from .model_conversion_cli import main as convert_main

        return convert_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Export main CLI functions
__all__ = [
    "cli",
    "main",
    "compliance",
    "chat",
    "models",
    "config",
    "serve",
    "audit",
    "dashboard",
    "convert_main",
    "context_transfer_main",
]

if __name__ == "__main__":
    main()
