"""
API for MultiMind compliance features.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta

from .model_training import ComplianceTrainer
from . import GovernanceConfig, Regulation

app = FastAPI(
    title="MultiMind Compliance API",
    description="API for managing compliance monitoring and evaluation",
    version="1.0.0"
)

class ComplianceConfig(BaseModel):
    """Compliance configuration model."""
    organization_id: str
    organization_name: str
    dpo_email: str
    enabled_regulations: List[str]
    compliance_rules: Dict[str, Any]
    metadata: Dict[str, Any]

class ComplianceResult(BaseModel):
    """Compliance result model."""
    final_evaluation: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    metrics: Dict[str, float]

class DashboardMetrics(BaseModel):
    """Dashboard metrics model."""
    total_checks: int
    passed_checks: int
    failed_checks: int
    compliance_score: float
    recent_issues: List[Dict[str, Any]]
    trend_data: Dict[str, List[float]]
    alerts: List[Dict[str, Any]]

@app.post("/compliance/monitor", response_model=ComplianceResult)
async def monitor_compliance(config: ComplianceConfig):
    """Run compliance monitoring."""
    try:
        # Initialize governance config
        governance_config = GovernanceConfig(
            organization_id=config.organization_id,
            organization_name=config.organization_name,
            dpo_email=config.dpo_email,
            enabled_regulations=[Regulation[r] for r in config.enabled_regulations]
        )
        
        # Run compliance monitoring
        results = await run_compliance_monitoring(config.dict())
        return ComplianceResult(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compliance/example/{type}", response_model=ComplianceResult)
async def run_example(type: str, use_case: Optional[str] = None):
    """Run compliance example."""
    try:
        if type == 'healthcare':
            from examples.compliance.healthcare_compliance_example import main as run_healthcare
            results = await run_healthcare()
        else:
            from examples.compliance.compliance_training_example import main as run_general
            results = await run_general()
        
        return ComplianceResult(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compliance/report", response_model=Dict[str, Any])
async def generate_report(config: ComplianceConfig):
    """Generate compliance report."""
    try:
        report = await generate_compliance_report(config.dict())
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compliance/regulations", response_model=List[str])
async def list_regulations():
    """List available regulations."""
    return [r.name for r in Regulation]

@app.get("/compliance/healthcare/use-cases", response_model=List[str])
async def list_healthcare_use_cases():
    """List available healthcare use cases."""
    return [
        "medical_diagnosis",
        "patient_monitoring",
        "medical_imaging",
        "clinical_trial",
        "ehr",
        "medical_device",
        "medical_research",
        "telemedicine",
        "mental_health",
        "medical_imaging_analysis",
        "drug_discovery",
        "fraud_detection"
    ]

@app.get("/compliance/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    organization_id: str,
    time_range: Optional[str] = "7d",
    use_case: Optional[str] = None
):
    """Get compliance dashboard metrics."""
    try:
        # Parse time range
        if time_range.endswith('d'):
            days = int(time_range[:-1])
        elif time_range.endswith('h'):
            days = int(time_range[:-1]) / 24
        else:
            days = 7  # Default to 7 days
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get compliance history
        history = await get_compliance_history(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            use_case=use_case
        )
        
        # Calculate metrics
        total_checks = len(history)
        passed_checks = sum(1 for check in history if check["status"] == "passed")
        failed_checks = total_checks - passed_checks
        compliance_score = passed_checks / total_checks if total_checks > 0 else 0
        
        # Get recent issues
        recent_issues = [
            check for check in history 
            if check["status"] == "failed"
        ][-5:]  # Last 5 issues
        
        # Calculate trend data
        trend_data = {
            "compliance_score": [],
            "privacy_score": [],
            "fairness_score": [],
            "transparency_score": []
        }
        
        for check in history:
            trend_data["compliance_score"].append(check["metrics"]["overall_score"])
            trend_data["privacy_score"].append(check["metrics"]["privacy_score"])
            trend_data["fairness_score"].append(check["metrics"]["fairness_score"])
            trend_data["transparency_score"].append(check["metrics"]["transparency_score"])
        
        # Get active alerts
        alerts = await get_active_alerts(organization_id, use_case)
        
        return DashboardMetrics(
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            compliance_score=compliance_score,
            recent_issues=recent_issues,
            trend_data=trend_data,
            alerts=alerts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compliance/alerts/configure")
async def configure_alerts(
    organization_id: str,
    alert_rules: Dict[str, Any]
):
    """Configure compliance alert rules."""
    try:
        await save_alert_rules(organization_id, alert_rules)
        return {"status": "success", "message": "Alert rules configured successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compliance/alerts")
async def get_alerts(
    organization_id: str,
    status: Optional[str] = "active",
    severity: Optional[str] = None
):
    """Get compliance alerts."""
    try:
        alerts = await get_compliance_alerts(
            organization_id=organization_id,
            status=status,
            severity=severity
        )
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 