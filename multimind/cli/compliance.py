"""
Command-line interface for MultiMind compliance features.
"""

import click
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta

from ..compliance.model_training import ComplianceTrainer
from ..compliance.governance import GovernanceConfig, Regulation
from ..gateway.compliance_api import (
    run_compliance_monitoring,
    generate_compliance_report,
    get_dashboard_metrics,
    get_compliance_alerts,
    save_alert_rules
)

@click.group()
def compliance():
    """MultiMind compliance management commands."""
    pass

@compliance.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to compliance configuration file')
@click.option('--output', '-o', type=click.Path(), help='Path to save results')
def run_compliance(config: str, output: str):
    """Run compliance monitoring."""
    asyncio.run(_run_compliance(config, output))

async def _run_compliance(config_path: str, output_path: str):
    """Run compliance monitoring with configuration."""
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Initialize governance config
    governance_config = GovernanceConfig(
        organization_id=config["organization_id"],
        organization_name=config["organization_name"],
        dpo_email=config["dpo_email"],
        enabled_regulations=[Regulation[r] for r in config["enabled_regulations"]]
    )
    
    # Run compliance monitoring
    results = await run_compliance_monitoring(config)
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print results
    print("\nCompliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

@compliance.command()
@click.option('--type', '-t', type=click.Choice(['healthcare', 'general']), required=True, help='Type of compliance monitoring')
@click.option('--use-case', '-u', type=str, help='Specific use case for healthcare compliance')
@click.option('--output', '-o', type=click.Path(), help='Path to save results')
def run_example(type: str, use_case: str, output: str):
    """Run compliance example."""
    asyncio.run(_run_example(type, use_case, output))

async def _run_example(type: str, use_case: str, output: str):
    """Run compliance example."""
    if type == 'healthcare':
        from examples.compliance.healthcare_compliance_example import main as run_healthcare
        results = await run_healthcare()
    else:
        from examples.compliance.compliance_training_example import main as run_general
        results = await run_general()
    
    # Save results
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print results
    print("\nCompliance Evaluation Results:")
    print(json.dumps(results["final_evaluation"], indent=2))
    
    print("\nRecommendations:")
    for rec in results["final_evaluation"]["recommendations"]:
        print(f"- {rec['action']} (Priority: {rec['priority']})")

@compliance.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to compliance configuration file')
@click.option('--output', '-o', type=click.Path(), help='Path to save report')
def generate_report(config: str, output: str):
    """Generate compliance report."""
    asyncio.run(_generate_report(config, output))

async def _generate_report(config_path: str, output_path: str):
    """Generate compliance report."""
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Generate report
    report = await generate_compliance_report(config)
    
    # Save report
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    # Print report
    print("\nCompliance Report:")
    print(json.dumps(report, indent=2))

@compliance.command()
@click.option('--organization-id', '-o', required=True, help='Organization ID')
@click.option('--time-range', '-t', default='7d', help='Time range (e.g., 7d, 24h)')
@click.option('--use-case', '-u', help='Specific use case')
@click.option('--output', '-o', type=click.Path(), help='Path to save dashboard data')
def dashboard(organization_id: str, time_range: str, use_case: str, output: str):
    """Show compliance dashboard."""
    asyncio.run(_show_dashboard(organization_id, time_range, use_case, output))

async def _show_dashboard(organization_id: str, time_range: str, use_case: str, output: str):
    """Show compliance dashboard."""
    # Get dashboard metrics
    metrics = await get_dashboard_metrics(
        organization_id=organization_id,
        time_range=time_range,
        use_case=use_case
    )
    
    # Save metrics if output path provided
    if output:
        with open(output, 'w') as f:
            json.dump(metrics.dict(), f, indent=2)
    
    # Print dashboard
    print("\nCompliance Dashboard")
    print("===================")
    print(f"Organization: {organization_id}")
    print(f"Time Range: {time_range}")
    if use_case:
        print(f"Use Case: {use_case}")
    
    print("\nCompliance Overview")
    print(f"Total Checks: {metrics.total_checks}")
    print(f"Passed Checks: {metrics.passed_checks}")
    print(f"Failed Checks: {metrics.failed_checks}")
    print(f"Compliance Score: {metrics.compliance_score:.2%}")
    
    print("\nRecent Issues")
    for issue in metrics.recent_issues:
        print(f"- {issue['description']} (Severity: {issue['severity']})")
    
    print("\nActive Alerts")
    for alert in metrics.alerts:
        print(f"- {alert['description']} (Severity: {alert['severity']})")

@compliance.command()
@click.option('--organization-id', '-o', required=True, help='Organization ID')
@click.option('--status', '-s', default='active', help='Alert status (active/resolved)')
@click.option('--severity', '-v', help='Alert severity (high/medium/low)')
@click.option('--output', '-o', type=click.Path(), help='Path to save alerts')
def alerts(organization_id: str, status: str, severity: str, output: str):
    """Show compliance alerts."""
    asyncio.run(_show_alerts(organization_id, status, severity, output))

async def _show_alerts(organization_id: str, status: str, severity: str, output: str):
    """Show compliance alerts."""
    # Get alerts
    alerts = await get_compliance_alerts(
        organization_id=organization_id,
        status=status,
        severity=severity
    )
    
    # Save alerts if output path provided
    if output:
        with open(output, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    # Print alerts
    print("\nCompliance Alerts")
    print("================")
    print(f"Organization: {organization_id}")
    print(f"Status: {status}")
    if severity:
        print(f"Severity: {severity}")
    
    for alert in alerts:
        print(f"\n- {alert['description']}")
        print(f"  Severity: {alert['severity']}")
        print(f"  Created: {alert['created_at']}")
        if alert.get('resolved_at'):
            print(f"  Resolved: {alert['resolved_at']}")

@compliance.command()
@click.option('--organization-id', '-o', required=True, help='Organization ID')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to alert rules configuration')
def configure_alerts(organization_id: str, config: str):
    """Configure compliance alert rules."""
    asyncio.run(_configure_alerts(organization_id, config))

async def _configure_alerts(organization_id: str, config_path: str):
    """Configure compliance alert rules."""
    # Load alert rules
    with open(config_path) as f:
        alert_rules = json.load(f)
    
    # Configure alerts
    await save_alert_rules(organization_id, alert_rules)
    print("Alert rules configured successfully")

def main():
    """Main entry point for CLI."""
    compliance() 