"""
Metrics collection and telemetry system for MultimindSDK.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import logging
import json
import os
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

class Metric(BaseModel):
    """Base class for metrics."""
    timestamp: datetime
    provider: str
    task_type: str
    model: str
    value: float
    metadata: Dict[str, Any] = {}

class LatencyMetric(Metric):
    """Latency metric."""
    pass

class CostMetric(Metric):
    """Cost metric."""
    pass

class TokenMetric(Metric):
    """Token usage metric."""
    pass

class ErrorMetric(Metric):
    """Error metric."""
    error_type: str
    error_message: str

class MetricsCollector:
    """Collects and manages metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the metrics collector."""
        self.metrics: List[Metric] = []
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("multimind")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / "multimind.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        self.logger.addHandler(console_handler)
    
    def record_latency(self, provider: str, task_type: str, model: str, 
                      latency_ms: float, metadata: Dict[str, Any] = None):
        """Record a latency metric."""
        metric = LatencyMetric(
            timestamp=datetime.now(),
            provider=provider,
            task_type=task_type,
            model=model,
            value=latency_ms,
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        self.logger.info(f"Latency: {provider} {task_type} {model} {latency_ms}ms")
    
    def record_cost(self, provider: str, task_type: str, model: str,
                   cost: float, metadata: Dict[str, Any] = None):
        """Record a cost metric."""
        metric = CostMetric(
            timestamp=datetime.now(),
            provider=provider,
            task_type=task_type,
            model=model,
            value=cost,
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        self.logger.info(f"Cost: {provider} {task_type} {model} ${cost:.6f}")
    
    def record_tokens(self, provider: str, task_type: str, model: str,
                     tokens: int, metadata: Dict[str, Any] = None):
        """Record a token usage metric."""
        metric = TokenMetric(
            timestamp=datetime.now(),
            provider=provider,
            task_type=task_type,
            model=model,
            value=tokens,
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        self.logger.info(f"Tokens: {provider} {task_type} {model} {tokens}")
    
    def record_error(self, provider: str, task_type: str, model: str,
                    error_type: str, error_message: str, metadata: Dict[str, Any] = None):
        """Record an error metric."""
        metric = ErrorMetric(
            timestamp=datetime.now(),
            provider=provider,
            task_type=task_type,
            model=model,
            value=1.0,  # Error count
            error_type=error_type,
            error_message=error_message,
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        self.logger.error(f"Error: {provider} {task_type} {model} - {error_type}: {error_message}")
    
    def get_metrics(self, metric_type: Optional[str] = None,
                   provider: Optional[str] = None,
                   task_type: Optional[str] = None,
                   model: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Metric]:
        """Get filtered metrics."""
        filtered = self.metrics
        
        if metric_type:
            filtered = [m for m in filtered if m.__class__.__name__ == f"{metric_type}Metric"]
        if provider:
            filtered = [m for m in filtered if m.provider == provider]
        if task_type:
            filtered = [m for m in filtered if m.task_type == task_type]
        if model:
            filtered = [m for m in filtered if m.model == model]
        if start_time:
            filtered = [m for m in filtered if m.timestamp >= start_time]
        if end_time:
            filtered = [m for m in filtered if m.timestamp <= end_time]
        
        return filtered
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {
            "total_requests": len(self.metrics),
            "total_cost": sum(m.value for m in self.metrics if isinstance(m, CostMetric)),
            "total_tokens": sum(m.value for m in self.metrics if isinstance(m, TokenMetric)),
            "avg_latency": sum(m.value for m in self.metrics if isinstance(m, LatencyMetric)) / 
                         len([m for m in self.metrics if isinstance(m, LatencyMetric)]) 
                         if any(isinstance(m, LatencyMetric) for m in self.metrics) else 0,
            "error_count": len([m for m in self.metrics if isinstance(m, ErrorMetric)]),
            "providers": list(set(m.provider for m in self.metrics)),
            "task_types": list(set(m.task_type for m in self.metrics)),
            "models": list(set(m.model for m in self.metrics))
        }
        return summary
    
    def save_metrics(self, filepath: Optional[str] = None):
        """Save metrics to a JSON file."""
        if filepath is None:
            filepath = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        metrics_data = [m.dict() for m in self.metrics]
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics saved to {filepath}")

# CLI Commands
@click.group()
def cli():
    """Multimind metrics CLI."""
    pass

@cli.command()
@click.option('--metric-type', help='Type of metric to show')
@click.option('--provider', help='Filter by provider')
@click.option('--task-type', help='Filter by task type')
@click.option('--model', help='Filter by model')
@click.option('--start-time', help='Start time (YYYY-MM-DD HH:MM:SS)')
@click.option('--end-time', help='End time (YYYY-MM-DD HH:MM:SS)')
def show_metrics(metric_type, provider, task_type, model, start_time, end_time):
    """Show metrics in a table format."""
    collector = MetricsCollector()
    
    # Convert time strings to datetime objects
    start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') if start_time else None
    end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') if end_time else None
    
    metrics = collector.get_metrics(
        metric_type=metric_type,
        provider=provider,
        task_type=task_type,
        model=model,
        start_time=start,
        end_time=end
    )
    
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Timestamp")
    table.add_column("Provider")
    table.add_column("Task Type")
    table.add_column("Model")
    table.add_column("Value")
    table.add_column("Type")
    
    for metric in metrics:
        table.add_row(
            metric.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            metric.provider,
            metric.task_type,
            metric.model,
            str(metric.value),
            metric.__class__.__name__.replace('Metric', '')
        )
    
    console.print(table)

@cli.command()
def show_summary():
    """Show metrics summary."""
    collector = MetricsCollector()
    summary = collector.get_summary()
    
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value")
    
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            value = f"{value:,.2f}"
        elif isinstance(value, list):
            value = ", ".join(value)
        table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(table)

@cli.command()
@click.option('--filepath', help='Path to save metrics file')
def save_metrics(filepath):
    """Save metrics to a JSON file."""
    collector = MetricsCollector()
    collector.save_metrics(filepath)

if __name__ == '__main__':
    cli() 