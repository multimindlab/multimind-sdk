"""
Configuration for MultiMind compliance features.
"""

from typing import Any, Dict, List

from pydantic import BaseModel

from .governance import Regulation


class ComplianceRule(BaseModel):
    """Compliance rule configuration."""

    name: str
    description: str
    threshold: float
    enabled: bool = True
    metadata: Dict[str, Any] = {}


class ComplianceConfig(BaseModel):
    """Compliance configuration."""

    organization_id: str
    organization_name: str
    dpo_email: str
    enabled_regulations: List[Regulation]
    compliance_rules: List[ComplianceRule]
    metadata: Dict[str, Any] = {}


class HealthcareConfig(ComplianceConfig):
    """Healthcare-specific compliance configuration."""

    use_case: str
    data_categories: List[str]
    hipaa_covered: bool = True
    sensitive_data: bool = True
    explainability_required: bool = True


class ComplianceMetrics(BaseModel):
    """Compliance metrics configuration."""

    privacy_score: float
    fairness_score: float
    transparency_score: float
    bias_score: float
    overall_score: float


class ComplianceReport(BaseModel):
    """Compliance report configuration."""

    evaluation_results: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    metrics: ComplianceMetrics
    metadata: Dict[str, Any] = {}


# Default compliance rules
DEFAULT_COMPLIANCE_RULES = [
    ComplianceRule(
        name="privacy_threshold", description="Minimum privacy compliance score", threshold=0.9
    ),
    ComplianceRule(
        name="fairness_threshold", description="Minimum fairness compliance score", threshold=0.9
    ),
    ComplianceRule(
        name="transparency_threshold",
        description="Minimum transparency compliance score",
        threshold=0.9,
    ),
    ComplianceRule(name="bias_threshold", description="Maximum allowed bias score", threshold=0.1),
]

# Default healthcare compliance rules
DEFAULT_HEALTHCARE_RULES = DEFAULT_COMPLIANCE_RULES + [
    ComplianceRule(
        name="hipaa_compliance", description="HIPAA compliance requirements", threshold=1.0
    ),
    ComplianceRule(
        name="data_minimization", description="Data minimization requirements", threshold=0.9
    ),
    ComplianceRule(name="audit_trail", description="Audit trail requirements", threshold=1.0),
    ComplianceRule(
        name="explainability", description="Model explainability requirements", threshold=0.9
    ),
]


def load_config(config_path: str) -> ComplianceConfig:
    """Load compliance configuration from file."""
    import json

    from pydantic import ValidationError

    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
        return ComplianceConfig(**config_data)
    except FileNotFoundError as e:
        raise RuntimeError(f"Compliance config file not found: {config_path}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in compliance config: {config_path}") from e
    except ValidationError as e:
        raise RuntimeError(f"Invalid compliance config schema: {config_path}") from e
    except OSError as e:
        raise RuntimeError(f"Failed to read compliance config: {config_path}") from e


def save_config(config: ComplianceConfig, config_path: str):
    """Save compliance configuration to file."""
    import json
    import os

    try:
        parent = os.path.dirname(config_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        data = config.model_dump() if hasattr(config, "model_dump") else config.dict()
        tmp_path = f"{config_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, config_path)
    except OSError as e:
        raise RuntimeError(f"Failed to write compliance config: {config_path}") from e
