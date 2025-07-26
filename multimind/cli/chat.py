"""
Chat management commands for MultiMind CLI
"""

import asyncio
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from typing import Optional

from ..gateway.chat import chat_manager, ChatSession
from ..gateway.models import get_model_handler

console = Console()

@click.group()
def chat():
    """Chat management commands"""
    pass

@chat.command()
@click.option("--model", "-m", required=True, help="Model to use")
@click.option("--prompt", "-p", help="Single prompt to send (optional)")
def start(model: str, prompt: Optional[str]):
    """Start an interactive chat session with a model"""
    try:
        handler = get_model_handler(model)

        if prompt:
            # Single message mode
            response = asyncio.run(handler.generate(prompt))
            console.print(Panel(response.content, title=f"{model} Response"))
            return

        # Interactive chat mode
        console.print(f"[bold green]Starting chat with {model}[/bold green]")
        console.print("Type 'exit' to quit, 'clear' to clear history")

        chat_history = []
        while True:
            try:
                user_input = click.prompt("\nYou", type=str)

                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'clear':
                    chat_history = []
                    console.print("[yellow]Chat history cleared[/yellow]")
                    continue

                with Progress() as progress:
                    task = progress.add_task("[cyan]Thinking...", total=None)
                    response = asyncio.run(handler.chat(
                        [{"role": "user", "content": user_input}]
                    ))
                    progress.update(task, completed=True)

                chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "model": model
                })
                chat_history.append({
                    "role": "assistant",
                    "content": response.content,
                    "model": model
                })

                console.print(Panel(
                    response.content,
                    title=f"{model} Response",
                    border_style="green"
                ))

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@chat.command()
def list_sessions():
    """List all chat sessions"""
    try:
        sessions = chat_manager.list_sessions()

        if not sessions:
            console.print("[yellow]No active sessions found[/yellow]")
            return

        table = Table(title="Chat Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Created", style="blue")
        table.add_column("Updated", style="blue")
        table.add_column("Messages", style="yellow")

        for session in sessions:
            table.add_row(
                session["session_id"],
                session["model"],
                session["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
                session["updated_at"].strftime("%Y-%m-%d %H:%M:%S"),
                str(session["message_count"])
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@chat.command()
@click.argument("session_id")
def load(session_id: str):
    """Load a chat session"""
    try:
        session = chat_manager.get_session(session_id)
        if not session:
            session = chat_manager.load_session(session_id)
        if not session:
            console.print(f"[red]Session {session_id} not found[/red]")
            return

        console.print(f"[green]Loaded session {session_id}[/green]")
        console.print(f"Model: {session.model}")
        console.print(f"Messages: {len(session.messages)}")

        # Show recent messages
        if session.messages:
            console.print("\n[bold]Recent Messages:[/bold]")
            for msg in session.messages[-5:]:
                console.print(Panel(
                    msg.content,
                    title=f"{msg.role} ({msg.model})",
                    border_style="blue"
                ))

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@chat.command()
@click.argument("session_id")
def save(session_id: str):
    """Save a chat session"""
    try:
        if chat_manager.save_session(session_id):
            console.print(f"[green]Saved session {session_id}[/green]")
        else:
            console.print(f"[red]Failed to save session {session_id}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@chat.command()
@click.argument("session_id")
def delete(session_id: str):
    """Delete a chat session"""
    try:
        if chat_manager.delete_session(session_id):
            console.print(f"[green]Deleted session {session_id}[/green]")
        else:
            console.print(f"[red]Session {session_id} not found[/red]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")