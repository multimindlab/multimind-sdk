"""
Privacy compliance management commands for MultiMind CLI
"""

import asyncio
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..compliance.privacy import (
    PrivacyCompliance,
    GovernanceConfig,
    DataCategory,
    NotificationType,
    AuditAction
)

console = Console()

@click.group()
def compliance():
    """Privacy compliance management commands"""
    pass

@compliance.command()
@click.option('--org-id', required=True, help='Organization ID')
@click.option('--jurisdiction', default='global', help='Jurisdiction')
@click.option('--regulations', multiple=True, help='Regulations to monitor')
def init(org_id: str, jurisdiction: str, regulations: List[str]):
    """Initialize privacy compliance manager"""
    config = GovernanceConfig(
        organization_id=org_id,
        jurisdiction=jurisdiction,
        regulations=list(regulations)
    )
    privacy_manager = PrivacyCompliance(config=config)
    console.print(f"Initialized privacy compliance manager for {org_id}")

@compliance.command()
@click.option('--purpose-id', required=True, help='Purpose ID')
@click.option('--name', required=True, help='Purpose name')
@click.option('--description', required=True, help='Purpose description')
@click.option('--legal-basis', required=True, help='Legal basis')
@click.option('--retention-period', required=True, type=int, help='Retention period in days')
@click.option('--categories', multiple=True, help='Data categories')
def add_purpose(purpose_id: str, name: str, description: str, legal_basis: str, 
                retention_period: int, categories: List[str]):
    """Add a new data purpose"""
    data_categories = {DataCategory[cat.upper()] for cat in categories}
    
    result = asyncio.run(privacy_manager.add_data_purpose(
        purpose_id=purpose_id,
        name=name,
        description=description,
        legal_basis=legal_basis,
        retention_period=retention_period,
        data_categories=data_categories
    ))
    console.print(f"Added data purpose: {name}")

@compliance.command()
@click.option('--entity-id', required=True, help='Entity ID')
@click.option('--entity-type', default='system', help='Entity type')
def calculate_risk(entity_id: str, entity_type: str):
    """Calculate risk score for an entity"""
    risk_score = asyncio.run(privacy_manager.calculate_risk_score(
        entity_id=entity_id,
        entity_type=entity_type
    ))
    console.print(f"Risk score for {entity_id}: {risk_score.score} ({risk_score.level})")

@compliance.command()
@click.option('--dashboard-id', required=True, help='Dashboard ID')
@click.option('--name', required=True, help='Dashboard name')
@click.option('--description', required=True, help='Dashboard description')
@click.option('--refresh-interval', default=3600, help='Refresh interval in seconds')
def create_dashboard(dashboard_id: str, name: str, description: str, refresh_interval: int):
    """Create a new compliance dashboard"""
    dashboard = asyncio.run(privacy_manager.create_compliance_dashboard(
        dashboard_id=dashboard_id,
        name=name,
        description=description,
        refresh_interval=refresh_interval
    ))
    console.print(f"Created dashboard: {name}")

@compliance.command()
@click.option('--template-id', required=True, help='Template ID')
@click.option('--name', required=True, help='Template name')
@click.option('--regulation', required=True, help='Regulation')
@click.option('--jurisdiction', required=True, help='Jurisdiction')
def create_report_template(template_id: str, name: str, regulation: str, jurisdiction: str):
    """Create a new compliance report template"""
    template = asyncio.run(privacy_manager.create_report_template(
        template_id=template_id,
        name=name,
        description=f"Compliance report for {regulation}",
        regulation=regulation,
        jurisdiction=jurisdiction,
        sections=[
            {
                "id": "compliance_status",
                "type": "compliance_status",
                "title": "Compliance Status"
            },
            {
                "id": "risk_assessment",
                "type": "risk_assessment",
                "title": "Risk Assessment"
            }
        ]
    ))
    console.print(f"Created report template: {name}")

@compliance.command()
@click.option('--training-id', required=True, help='Training ID')
@click.option('--title', required=True, help='Training title')
@click.option('--description', required=True, help='Training description')
@click.option('--duration', required=True, type=int, help='Duration in minutes')
def create_training(training_id: str, title: str, description: str, duration: int):
    """Create a new compliance training"""
    training = asyncio.run(privacy_manager.create_compliance_training(
        training_id=training_id,
        title=title,
        description=description,
        modules=[
            {
                "id": "module_1",
                "title": "Overview",
                "duration": duration // 2
            },
            {
                "id": "module_2",
                "title": "Best Practices",
                "duration": duration // 2
            }
        ],
        target_audience=["employees"],
        duration=duration,
        completion_criteria={
            "required_modules": ["module_1", "module_2"],
            "minimum_percentage": 80
        }
    ))
    console.print(f"Created training: {title}")

@compliance.command()
def detect_anomalies():
    """Detect anomalies in system behavior"""
    anomalies = asyncio.run(privacy_manager.detect_anomalies())
    for anomaly in anomalies:
        console.print(f"Anomaly detected: {anomaly.description} (Severity: {anomaly.severity})") 