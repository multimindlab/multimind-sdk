"""
MultiMind CLI - Command Line Interface for MultiMind SDK
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .compliance import compliance
from .chat import chat
from .models import models
from .config import config
from .model_conversion_cli import main as convert_main

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

def main():
    """Main entry point for the CLI."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        sys.argv.pop(1)  # Remove 'convert' from arguments
        sys.exit(convert_main())
    else:
        print("Usage: multimind convert [options]")
        print("Run 'multimind convert --help' for more information")
        sys.exit(1)

if __name__ == "__main__":
    main() 