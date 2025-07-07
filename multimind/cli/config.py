"""
Configuration management commands for MultiMind CLI
"""

import click
import os
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

@click.group()
def config():
    """Configuration management commands"""
    pass

@config.command()
@click.option('--set', 'set_', nargs=2, type=str, help='Set a config key and value.')
@click.option('--get', 'get_', type=str, help='Get a config value by key.')
def manage(set_, get_):
    """View or set global CLI configuration"""
    config_path = os.path.expanduser('~/.multimind_cli_config')
    
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({}, f)
            
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        
    if set_:
        key, value = set_
        cfg[key] = value
        with open(config_path, 'w') as f:
            json.dump(cfg, f)
        console.print(f"[green]Set {key} = {value}[/green]")
    elif get_:
        value = cfg.get(get_)
        console.print(f"[cyan]{get_} = {value}[/cyan]")
    else:
        # Display all config in a table
        table = Table(title="CLI Configuration")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in cfg.items():
            table.add_row(key, str(value))
            
        console.print(table)

@config.command()
def info():
    """Show environment and configuration info"""
    console.print("[bold]MultiMind SDK environment info:[/bold]")
    
    try:
        import torch
        console.print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        console.print("[yellow]PyTorch not installed[/yellow]")
        
    try:
        import transformers
        console.print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        console.print("[yellow]Transformers not installed[/yellow]")
        
    console.print(f"Python version: {sys.version}")
    console.print(f"Platform: {sys.platform}")
    
    # Show config file location
    config_path = os.path.expanduser('~/.multimind_cli_config')
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"Config file: {config_path}")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            table = Table(title="Current Configuration")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in cfg.items():
                table.add_row(key, str(value))
                
            console.print(table)
    else:
        console.print("[yellow]No configuration file found[/yellow]")

@config.command()
@click.argument('shell', required=False, type=click.Choice(['bash', 'zsh', 'fish', 'powershell'], case_sensitive=False))
def completion(shell):
    """Generate shell completion script"""
    import sys
    import importlib
    if not shell:
        shell = click.prompt('Shell type (bash/zsh/fish/powershell)', type=click.Choice(['bash', 'zsh', 'fish', 'powershell']))
    console.print(f"[bold]Shell Completion for {shell}[/bold]")
    console.print("To enable completion, run:")
    console.print(f"[cyan]eval \"$(multimind completion {shell})\"[/cyan]")

    # Output the actual completion script for the shell
    # Find the main multimind CLI group
    multimind_cli = None
    try:
        multimind_cli = importlib.import_module('multimind.cli.__main__').cli
    except Exception:
        try:
            multimind_cli = importlib.import_module('multimind.cli').cli
        except Exception:
            console.print("[red]Could not import multimind CLI main group for completion.[/red]")
            sys.exit(1)
    script = click.shell_completion._get_completion_script(
        cli=multimind_cli,
        prog_name='multimind',
        shell=shell
    )
    click.echo(script) 