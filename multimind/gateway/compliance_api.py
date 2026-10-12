"""
Compliance API Gateway for MultiMind.
Provides endpoints for compliance monitoring, evaluation, and reporting.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..compliance.advanced import ComplianceShard
from ..compliance.governance import Regulation

# Configure logging
logger = logging.getLogger(__name__)

# Create router for compliance endpoints
router = APIRouter(
    prefix="/v1/compliance",
    tags=["compliance"],
    responses={404: {"description": "Not found"}},
)


# One shard per organization for the life of the process, so dashboard and
# alerts reflect the checks this process recorded.
_SHARDS: Dict[str, ComplianceShard] = {}


def _get_shard(organization_id: str) -> ComplianceShard:
    shard = _SHARDS.get(organization_id)
    if shard is None:
        shard = ComplianceShard(
            shard_id=f"shard_{organization_id}",
            jurisdiction="global",
            config={"organization_id": organization_id},
        )
        _SHARDS[organization_id] = shard
    return shard


def reset_compliance_state() -> None:
    """Forget every recorded check and alert in this process."""
    _SHARDS.clear()


class ComplianceConfig(BaseModel):
    """Compliance configuration model."""

    organization_id: str = Field(..., description="Organization identifier")
    organization_name: str = Field(..., description="Organization name")
    dpo_email: str = Field(..., description="Data Protection Officer email")
    enabled_regulations: List[str] = Field(..., description="List of enabled regulations")
    compliance_rules: Dict[str, Any] = Field(..., description="Compliance rules configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "organization_id": "org-123",
                    "organization_name": "Acme Corp",
                    "dpo_email": "dpo@acme.example",
                    "enabled_regulations": ["GDPR"],
                    "compliance_rules": {"data_retention_days": 90},
                    "metadata": {},
                }
            ]
        }
    }


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
    unverifiable_checks: int = Field(
        0, description="Checks that had no runtime evidence; neither passed nor failed"
    )
    compliance_score: float = Field(
        ..., description="passed / (passed + failed); 0 when nothing was recorded"
    )
    recent_issues: List[Dict[str, Any]] = Field(..., description="Recent compliance issues")
    trend_data: Dict[str, List[float]] = Field(..., description="Compliance trend data")
    alerts: List[Dict[str, Any]] = Field(..., description="Active compliance alerts")


def _parse_regulations(names: List[str]) -> List[Regulation]:
    try:
        return [Regulation[r] for r in names]
    except KeyError as e:
        valid = ", ".join(r.name for r in Regulation)
        raise HTTPException(
            status_code=400, detail=f"Unknown regulation {e}. Valid regulations: {valid}"
        )


@router.post(
    "/monitor",
    response_model=ComplianceResult,
    responses={400: {"description": "Unknown regulation"}, 500: {"description": "Internal error"}},
)
async def monitor_compliance(config: ComplianceConfig):
    """Run compliance monitoring against implemented checkers only."""
    _parse_regulations(config.enabled_regulations)
    try:
        results = await run_compliance_monitoring(config.model_dump())
        return ComplianceResult(**results)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in compliance monitoring")
        raise HTTPException(status_code=500, detail="Compliance monitoring failed")


@router.post("/report", response_model=Dict[str, Any])
async def generate_report(config: ComplianceConfig):
    """Generate compliance report."""
    _parse_regulations(config.enabled_regulations)
    try:
        report = await generate_compliance_report(config.model_dump())
        return report
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error generating compliance report")
        raise HTTPException(status_code=500, detail="Report generation failed")


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
        "fraud_detection",
    ]


@router.get("/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    organization_id: str, time_range: Optional[str] = "7d", use_case: Optional[str] = None
):
    """Get compliance dashboard metrics."""
    try:
        # Parse time range
        if time_range.endswith("d"):
            days = int(time_range[:-1])
        elif time_range.endswith("h"):
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
            use_case=use_case,
        )

        total_checks = len(history)
        passed_checks = sum(1 for check in history if check["status"] == "passed")
        failed_checks = sum(1 for check in history if check["status"] == "failed")
        unverifiable = total_checks - passed_checks - failed_checks
        decided = passed_checks + failed_checks
        compliance_score = passed_checks / decided if decided else 0

        recent_issues = [check for check in history if check["status"] == "failed"][-5:]

        # Only the pass/fail outcome is real evidence; no other score is invented.
        trend_data = {
            "compliance_score": [
                1.0 if check["status"] == "passed" else 0.0
                for check in history
                if check["status"] in ("passed", "failed")
            ]
        }

        alerts = await get_active_alerts(organization_id, use_case)

        return DashboardMetrics(
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            unverifiable_checks=unverifiable,
            compliance_score=compliance_score,
            recent_issues=recent_issues,
            trend_data=trend_data,
            alerts=alerts,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/alerts/configure")
async def configure_alerts(organization_id: str, alert_rules: Dict[str, Any]):
    """Configure compliance alert rules."""
    try:
        await save_alert_rules(organization_id, alert_rules)
        return {"status": "success", "message": "Alert rules configured successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/alerts")
async def get_alerts(
    organization_id: str, status: Optional[str] = "active", severity: Optional[str] = None
):
    """Get compliance alerts."""
    try:
        alerts = await get_compliance_alerts(
            organization_id=organization_id, status=status, severity=severity
        )
        return alerts
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Helper functions for compliance operations


async def run_compliance_monitoring(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check implemented regulations. Missing evidence is unverifiable, never a pass."""
    checks: List[Dict[str, Any]] = []
    recommendations: List[Dict[str, Any]] = []
    metadata = config.get("metadata") or {}
    text = metadata.get("text")
    if isinstance(text, str) and text.strip():
        from ..compliance.guard import PIIDetector

        matches = PIIDetector().detect(text)
        types = sorted({m.type for m in matches})
        checks.append(
            {
                "check": "pii_scan",
                "status": "findings" if matches else "clear",
                "types": types,
                "count": len(matches),
            }
        )
        if matches:
            recommendations.append(
                {
                    "action": "redact_or_block",
                    "detail": f"PII types detected: {', '.join(types)}",
                }
            )

    for name in config.get("enabled_regulations") or []:
        if name == "GDPR" and metadata.get("purpose"):
            from ..compliance.gdpr import GDPRPolicy, PurposeRule
            from ..compliance.governance import GovernanceConfig

            retention = (config.get("compliance_rules") or {}).get("data_retention_days", 365)
            policy = GDPRPolicy(
                config=GovernanceConfig(
                    organization_id=config["organization_id"],
                    organization_name=config["organization_name"],
                    dpo_email=config["dpo_email"],
                    data_retention_days=int(retention),
                )
            )
            for rule in metadata.get("purpose_rules") or []:
                policy.register_purpose(PurposeRule(**rule) if isinstance(rule, dict) else rule)
            decision = policy.check_processing(
                purpose=metadata["purpose"],
                data_categories=metadata.get("data_categories") or [],
                consent=bool(metadata.get("consent")),
                retention_days=metadata.get("retention_days"),
            )
            checks.append(
                {
                    "regulation": "GDPR",
                    "status": decision.status,
                    "allowed": decision.allowed,
                    "reasons": decision.reasons,
                }
            )
        else:
            checks.append(
                {
                    "regulation": name,
                    "status": "unverifiable",
                    "detail": (
                        f"No runtime evidence was supplied for {name}. "
                        "A missing checker is not treated as compliant."
                    ),
                }
            )

    statuses = [c.get("status") for c in checks]
    if not checks or any(s == "unverifiable" for s in statuses):
        overall = "unverifiable"
    elif any(s in ("denied", "findings", "non_compliant") for s in statuses):
        overall = "attention"
    else:
        overall = "checked"
    _record_check(config, overall, checks)
    return {
        "final_evaluation": {"status": overall, "checks": checks},
        "recommendations": recommendations,
        "metrics": {},
    }


def _record_check(config: Dict[str, Any], overall: str, checks: List[Dict[str, Any]]) -> None:
    """Append the outcome to the organization's shard so dashboard and alerts see it."""
    shard = _get_shard(config["organization_id"])
    now = datetime.now().isoformat()
    status = {"checked": "passed", "attention": "failed"}.get(overall, "unverifiable")
    use_case = (config.get("metadata") or {}).get("use_case")
    problems = [c for c in checks if c.get("status") in ("denied", "findings", "non_compliant")]
    description = (
        "; ".join(f"{c.get('regulation') or c.get('check')}: {c.get('status')}" for c in problems)
        or f"{overall}: {len(checks)} check(s)"
    )
    severity = "high" if any(c.get("check") == "pii_scan" for c in problems) else "medium"
    shard.history.append(
        {
            "timestamp": now,
            "status": status,
            "use_case": use_case,
            "description": description,
            "severity": severity if problems else "low",
            "checks": checks,
        }
    )
    if status == "failed" and shard.alert_rules.get("enabled", True):
        shard.alerts.append(
            {
                "id": f"alert-{len(shard.alerts) + 1}",
                "description": description,
                "severity": severity,
                "status": "active",
                "created_at": now,
                "resolved_at": None,
                "use_case": use_case,
            }
        )


async def generate_compliance_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a report from implemented checks. Does not invent a compliance score."""
    monitoring_results = await run_compliance_monitoring(config)
    checks = monitoring_results["final_evaluation"]["checks"]
    enabled = set(config.get("enabled_regulations") or [])
    return {
        "organization": {
            "id": config["organization_id"],
            "name": config["organization_name"],
            "dpo_email": config["dpo_email"],
        },
        "compliance_summary": {
            "status": monitoring_results["final_evaluation"]["status"],
            "last_updated": datetime.now().isoformat(),
        },
        "checks": checks,
        "recommendations": monitoring_results["recommendations"],
        "regulations": {
            reg: {"status": "enabled" if reg in enabled else "disabled"}
            for reg in [r.name for r in Regulation]
        },
    }


async def get_compliance_history(
    organization_id: str, start_date: datetime, end_date: datetime, use_case: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return history recorded on this process's compliance shard."""
    try:
        shard = _get_shard(organization_id)
        return await shard.get_compliance_history(
            start_date=start_date, end_date=end_date, use_case=use_case
        )
    except Exception as e:
        logger.error(f"Error getting compliance history: {str(e)}")
        raise


async def get_active_alerts(
    organization_id: str, use_case: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return active alerts recorded on this process's compliance shard."""
    try:
        shard = _get_shard(organization_id)
        return await shard.get_active_alerts(use_case=use_case)
    except Exception as e:
        logger.error(f"Error getting active alerts: {str(e)}")
        raise


async def save_alert_rules(organization_id: str, alert_rules: Dict[str, Any]) -> None:
    """Store alert rules on this process's compliance shard."""
    try:
        shard = _get_shard(organization_id)
        await shard.configure_alerts(alert_rules)
    except Exception as e:
        logger.error(f"Error saving alert rules: {str(e)}")
        raise


async def get_compliance_alerts(
    organization_id: str, status: Optional[str] = "active", severity: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return alerts recorded on this process's compliance shard."""
    try:
        shard = _get_shard(organization_id)
        return await shard.get_alerts(status=status, severity=severity)
    except Exception as e:
        logger.error(f"Error getting compliance alerts: {str(e)}")
        raise


def init_app(app):
    """Initialize the compliance API routes."""
    app.include_router(router)
