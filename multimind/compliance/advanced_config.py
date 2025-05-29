"""
Configuration for advanced compliance features.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class PrivacyLevel(str, Enum):
    """Privacy protection levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    MAXIMAL = "maximal"

class WatermarkType(str, Enum):
    """Types of model watermarks."""
    VISIBLE = "visible"
    INVISIBLE = "invisible"
    DYNAMIC = "dynamic"

class ComplianceLevel(str, Enum):
    """Compliance verification levels."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    CRITICAL = "critical"

class ConsensusMethod(str, Enum):
    """Methods for reaching consensus in federated compliance."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    BYZANTINE = "byzantine"
    PROOF_OF_COMPLIANCE = "proof_of_compliance"

class ComplianceShardConfig(BaseModel):
    """Enhanced configuration for compliance shards."""
    shard_id: str
    jurisdiction: str
    epsilon: float = 1.0
    rules: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    compliance_level: ComplianceLevel = ComplianceLevel.STANDARD
    encryption_enabled: bool = True
    metrics_tracking: bool = True
    resource_limits: Dict[str, float] = Field(default_factory=lambda: {
        "cpu": 1.0,
        "memory": 1024.0,
        "network": 100.0
    })

class SelfHealingConfig(BaseModel):
    """Enhanced configuration for self-healing compliance."""
    auto_patch: bool = True
    rollback_enabled: bool = True
    notification_channels: List[str]
    vulnerability_threshold: float = 0.8
    patch_history_size: int = 100
    effectiveness_tracking: bool = True
    rollback_points: int = 10
    patch_validation: bool = True
    impact_analysis: bool = True

class ExplainableDTOConfig(BaseModel):
    """Enhanced configuration for explainable DTOs."""
    model_version: str
    confidence_threshold: float = 0.8
    explanation_depth: int = 3
    include_metadata: bool = True
    uncertainty_estimation: bool = True
    factor_importance: bool = True
    explanation_history: bool = True
    visualization_enabled: bool = True

class ModelWatermarkingConfig(BaseModel):
    """Enhanced configuration for model watermarking."""
    watermark_type: WatermarkType
    fingerprint_size: int = 256
    tracking_enabled: bool = True
    verification_threshold: float = 0.9
    tamper_detection: bool = True
    version_tracking: bool = True
    verification_history: bool = True
    security_level: str = "high"

class AdaptivePrivacyConfig(BaseModel):
    """Enhanced configuration for adaptive privacy."""
    initial_epsilon: float = 1.0
    min_epsilon: float = 0.1
    max_epsilon: float = 10.0
    adaptation_rate: float = 0.1
    feedback_window: int = 100
    adaptation_strategy: str = "dynamic"
    privacy_metrics: bool = True
    validation_enabled: bool = True
    guarantees_verification: bool = True

class RegulatoryChangeConfig(BaseModel):
    """Enhanced configuration for regulatory change detection."""
    sources: List[Dict[str, str]]
    check_interval: int = 3600  # seconds
    auto_patch: bool = True
    notification_channels: List[str]
    impact_analysis: bool = True
    patch_validation: bool = True
    patch_testing: bool = True
    change_history: bool = True

class FederatedComplianceConfig(BaseModel):
    """Enhanced configuration for federated compliance."""
    shards: List[ComplianceShardConfig]
    coordinator: Dict[str, Any]
    aggregation_method: str = "weighted"
    proof_generation: bool = True
    consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED
    load_balancing: bool = True
    verification_history: bool = True
    security_level: str = "high"

# Default configurations
DEFAULT_SHARD_CONFIG = ComplianceShardConfig(
    shard_id="default",
    jurisdiction="global",
    epsilon=1.0,
    rules=[],
    metadata={},
    compliance_level=ComplianceLevel.STANDARD,
    encryption_enabled=True,
    metrics_tracking=True
)

DEFAULT_SELF_HEALING_CONFIG = SelfHealingConfig(
    auto_patch=True,
    rollback_enabled=True,
    notification_channels=["email", "slack"],
    vulnerability_threshold=0.8,
    patch_history_size=100,
    effectiveness_tracking=True,
    rollback_points=10,
    patch_validation=True,
    impact_analysis=True
)

DEFAULT_EXPLAINABLE_DTO_CONFIG = ExplainableDTOConfig(
    model_version="1.0.0",
    confidence_threshold=0.8,
    explanation_depth=3,
    include_metadata=True,
    uncertainty_estimation=True,
    factor_importance=True,
    explanation_history=True,
    visualization_enabled=True
)

DEFAULT_WATERMARKING_CONFIG = ModelWatermarkingConfig(
    watermark_type=WatermarkType.INVISIBLE,
    fingerprint_size=256,
    tracking_enabled=True,
    verification_threshold=0.9,
    tamper_detection=True,
    version_tracking=True,
    verification_history=True,
    security_level="high"
)

DEFAULT_ADAPTIVE_PRIVACY_CONFIG = AdaptivePrivacyConfig(
    initial_epsilon=1.0,
    min_epsilon=0.1,
    max_epsilon=10.0,
    adaptation_rate=0.1,
    feedback_window=100,
    adaptation_strategy="dynamic",
    privacy_metrics=True,
    validation_enabled=True,
    guarantees_verification=True
)

DEFAULT_REGULATORY_CONFIG = RegulatoryChangeConfig(
    sources=[
        {"name": "EU", "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/RSS/"},
        {"name": "US", "url": "https://www.federalregister.gov/api/v1/documents.rss"}
    ],
    check_interval=3600,
    auto_patch=True,
    notification_channels=["email", "slack"],
    impact_analysis=True,
    patch_validation=True,
    patch_testing=True,
    change_history=True
)

DEFAULT_FEDERATED_CONFIG = FederatedComplianceConfig(
    shards=[DEFAULT_SHARD_CONFIG],
    coordinator={"type": "centralized"},
    aggregation_method="weighted",
    proof_generation=True,
    consensus_method=ConsensusMethod.WEIGHTED,
    load_balancing=True,
    verification_history=True,
    security_level="high"
)

def load_advanced_config(config_path: str) -> Dict[str, Any]:
    """Load advanced compliance configuration from file."""
    import json
    with open(config_path) as f:
        config_data = json.load(f)
    return config_data

def save_advanced_config(config: Dict[str, Any], config_path: str):
    """Save advanced compliance configuration to file."""
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2) 