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
from .context_transfer import main as context_transfer_main

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
    if len(sys.argv) > 1:
        if sys.argv[1] == "convert":
            sys.argv.pop(1)  # Remove 'convert' from arguments
            sys.exit(convert_main())
        elif sys.argv[1] == "context-transfer":
            sys.argv.pop(1)  # Remove 'context-transfer' from arguments
            sys.exit(context_transfer_main())
        else:
            print("Usage: multimind [convert|context-transfer] [options]")
            print("Run 'multimind convert --help' or 'multimind context-transfer --help' for more information")
            sys.exit(1)
    else:
        print("Usage: multimind [convert|context-transfer] [options]")
        print("Run 'multimind convert --help' or 'multimind context-transfer --help' for more information")
        sys.exit(1)

# Export main CLI functions
__all__ = [
    "cli",
    "main",
    "compliance",
    "chat", 
    "models",
    "config",
    "convert_main",
    "context_transfer_main"
]

if __name__ == "__main__":
    main() 