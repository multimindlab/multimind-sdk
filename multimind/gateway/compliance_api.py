"""
Compliance API Gateway for MultiMind.
Provides endpoints for compliance monitoring, evaluation, and reporting.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from ..compliance.model_training import ComplianceTrainer
from ..compliance.governance import GovernanceConfig, Regulation
from ..compliance.advanced import (
    ComplianceShard,
    SelfHealingCompliance,
    ExplainableDTO,
    ModelWatermarking,
    AdaptivePrivacy,
    RegulatoryChangeDetector,
    FederatedCompliance,
    ComplianceLevel
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router for compliance endpoints
router = APIRouter(
    prefix="/v1/compliance",
    tags=["compliance"],
    responses={404: {"description": "Not found"}},
)

# Pydantic models for request/response
class ComplianceConfig(BaseModel):
    """Compliance configuration model."""
    organization_id: str = Field(..., description="Organization identifier")
    organization_name: str = Field(..., description="Organization name")
    dpo_email: str = Field(..., description="Data Protection Officer email")
    enabled_regulations: List[str] = Field(..., description="List of enabled regulations")
    compliance_rules: Dict[str, Any] = Field(..., description="Compliance rules configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ComplianceResult(BaseModel):
    """Compliance result model."""
    final_evaluation: Dict[str, Any] = Field(..., description="Final compliance evaluation")
    recommendations: List[Dict[str, Any]] = Field(..., description="Compliance recommendations")
    metrics: Dict[str, float] = Field(..., description="Compliance metrics")

class DashboardMetrics(BaseModel):
    """Dashboard metrics model."""
    total_checks: int = Field(..., description="Total compliance checks")
    passed_checks: int = Field(..., description="Number of passed checks")
    failed_checks: int = Field(..., description="Number of failed checks")
    compliance_score: float = Field(..., description="Overall compliance score")
    recent_issues: List[Dict[str, Any]] = Field(..., description="Recent compliance issues")
    trend_data: Dict[str, List[float]] = Field(..., description="Compliance trend data")
    alerts: List[Dict[str, Any]] = Field(..., description="Active compliance alerts")

@router.post("/monitor", response_model=ComplianceResult)
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
        logger.error(f"Error in compliance monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/example/{type}", response_model=ComplianceResult)
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
        logger.error(f"Error running compliance example: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/report", response_model=Dict[str, Any])
async def generate_report(config: ComplianceConfig):
    """Generate compliance report."""
    try:
        report = await generate_compliance_report(config.dict())
        return report
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/regulations", response_model=List[str])
async def list_regulations():
    """List available regulations."""
    return [r.name for r in Regulation]

@router.get("/healthcare/use-cases", response_model=List[str])
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

@router.get("/dashboard", response_model=DashboardMetrics)
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
        logger.error(f"Error getting dashboard metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/configure")
async def configure_alerts(
    organization_id: str,
    alert_rules: Dict[str, Any]
):
    """Configure compliance alert rules."""
    try:
        await save_alert_rules(organization_id, alert_rules)
        return {"status": "success", "message": "Alert rules configured successfully"}
    except Exception as e:
        logger.error(f"Error configuring alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
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
        logger.error(f"Error getting alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions for compliance operations
async def run_compliance_monitoring(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run compliance monitoring with the given configuration."""
    try:
        # Initialize compliance components with proper parameters
        shard = ComplianceShard(
            shard_id=f"shard_{config['organization_id']}",
            jurisdiction=config.get('jurisdiction', 'global'),
            config=config
        )
        self_healing = SelfHealingCompliance(config)
        explainable = ExplainableDTO(config)
        watermarking = ModelWatermarking(config)
        privacy = AdaptivePrivacy(config)
        detector = RegulatoryChangeDetector(config)
        federated = FederatedCompliance(config)

        # Run compliance checks
        evaluation = await shard.verify_compliance(config)
        healing_result = await self_healing.check_and_heal(config)
        explanation = await explainable.explain_decision(evaluation)
        watermark = await watermarking.watermark_model(config.get('model', None))
        privacy_result = await privacy.adapt_privacy(config)
        regulatory_changes = await detector.detect_changes()
        federated_result = await federated.verify_global_compliance(config)

        # Combine results
        return {
            "final_evaluation": evaluation,
            "recommendations": [
                healing_result.get("recommendations", []),
                privacy_result.get("recommendations", []),
                federated_result.get("recommendations", [])
            ],
            "metrics": {
                "compliance_score": evaluation.get("score", 0.0),
                "privacy_score": privacy_result.get("score", 0.0),
                "fairness_score": evaluation.get("fairness_score", 0.0),
                "transparency_score": evaluation.get("transparency_score", 0.0)
            }
        }
    except Exception as e:
        logger.error(f"Error in compliance monitoring: {str(e)}")
        raise

async def generate_compliance_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive compliance report."""
    try:
        # Get compliance monitoring results
        monitoring_results = await run_compliance_monitoring(config)
        
        # Generate report sections
        report = {
            "organization": {
                "id": config["organization_id"],
                "name": config["organization_name"],
                "dpo_email": config["dpo_email"]
            },
            "compliance_summary": {
                "overall_score": monitoring_results["metrics"]["compliance_score"],
                "status": "compliant" if monitoring_results["metrics"]["compliance_score"] >= 0.8 else "non-compliant",
                "last_updated": datetime.now().isoformat()
            },
            "detailed_metrics": monitoring_results["metrics"],
            "recommendations": monitoring_results["recommendations"],
            "regulations": {
                reg: {
                    "status": "enabled" if reg in config["enabled_regulations"] else "disabled",
                    "compliance_score": monitoring_results["final_evaluation"].get(reg, {}).get("score", 0.0)
                }
                for reg in [r.name for r in Regulation]
            }
        }
        
        return report
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise

async def get_compliance_history(
    organization_id: str,
    start_date: datetime,
    end_date: datetime,
    use_case: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get compliance check history for an organization."""
    try:
        # Initialize compliance components with proper parameters
        shard = ComplianceShard(
            shard_id=f"shard_{organization_id}",
            jurisdiction="global",
            config={"organization_id": organization_id, "use_case": use_case}
        )
        
        # Get history from shard
        history = await shard.get_compliance_history(
            start_date=start_date,
            end_date=end_date,
            use_case=use_case
        )
        
        return history
    except Exception as e:
        logger.error(f"Error getting compliance history: {str(e)}")
        raise

async def get_active_alerts(
    organization_id: str,
    use_case: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get active compliance alerts for an organization."""
    try:
        # Initialize compliance components with proper parameters
        shard = ComplianceShard(
            shard_id=f"shard_{organization_id}",
            jurisdiction="global",
            config={"organization_id": organization_id, "use_case": use_case}
        )
        
        # Get alerts from shard
        alerts = await shard.get_active_alerts(use_case=use_case)
        
        return alerts
    except Exception as e:
        logger.error(f"Error getting active alerts: {str(e)}")
        raise

async def save_alert_rules(
    organization_id: str,
    alert_rules: Dict[str, Any]
) -> None:
    """Save alert rules for an organization."""
    try:
        # Initialize compliance components with proper parameters
        shard = ComplianceShard(
            shard_id=f"shard_{organization_id}",
            jurisdiction="global",
            config={"organization_id": organization_id, "alert_rules": alert_rules}
        )
        
        # Save rules to shard
        await shard.configure_alerts(alert_rules)
    except Exception as e:
        logger.error(f"Error saving alert rules: {str(e)}")
        raise

async def get_compliance_alerts(
    organization_id: str,
    status: Optional[str] = "active",
    severity: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get compliance alerts with optional filtering."""
    try:
        # Initialize compliance components with proper parameters
        shard = ComplianceShard(
            shard_id=f"shard_{organization_id}",
            jurisdiction="global",
            config={"organization_id": organization_id, "status": status, "severity": severity}
        )
        
        # Get alerts from shard with filters
        alerts = await shard.get_alerts(
            status=status,
            severity=severity
        )
        
        return alerts
    except Exception as e:
        logger.error(f"Error getting compliance alerts: {str(e)}")
        raise

def init_app(app):
    """Initialize the compliance API routes."""
    app.include_router(router) 