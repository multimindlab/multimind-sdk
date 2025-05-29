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

if __name__ == "__main__":
    cli() 