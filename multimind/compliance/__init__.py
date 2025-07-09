"""
MultiMind Compliance Module

This module provides comprehensive compliance monitoring and evaluation capabilities,
including advanced features for privacy, security, and regulatory compliance.
"""

from .advanced_config import (
    ComplianceShardConfig,
    SelfHealingConfig,
    ExplainableDTOConfig,
    ModelWatermarkingConfig,
    AdaptivePrivacyConfig,
    RegulatoryChangeConfig,
    FederatedComplianceConfig,
    load_advanced_config,
    save_advanced_config
)

from .advanced import (
    ComplianceShard,
    SelfHealingCompliance,
    ExplainableDTO,
    ModelWatermarking,
    AdaptivePrivacy,
    RegulatoryChangeDetector,
    FederatedCompliance,
    ComplianceLevel,
    ComplianceMetrics
)

from .governance import GovernanceConfig, Regulation
from .model_training import ComplianceTrainer

__all__ = [
    # Advanced Features
    'ComplianceShard',
    'SelfHealingCompliance',
    'ExplainableDTO',
    'ModelWatermarking',
    'AdaptivePrivacy',
    'RegulatoryChangeDetector',
    'FederatedCompliance',
    'ComplianceLevel',
    'ComplianceMetrics',
    # Advanced Configurations
    'ComplianceShardConfig',
    'SelfHealingConfig',
    'ExplainableDTOConfig',
    'ModelWatermarkingConfig',
    'AdaptivePrivacyConfig',
    'RegulatoryChangeConfig',
    'FederatedComplianceConfig',
    'load_advanced_config',
    'save_advanced_config',
    # Governance
    'GovernanceConfig',
    'Regulation',
    # Training
    'ComplianceTrainer',
]

# Backward compatibility: import legacy CLI and API functions if available
try:
    from .cli import (
        run_compliance,
        run_example,
        generate_report,
        show_dashboard,
        show_alerts,
        configure_alerts
    )
    __all__.extend([
        'run_compliance',
        'run_example',
        'generate_report',
        'show_dashboard',
        'show_alerts',
        'configure_alerts',
    ])
except ImportError:
    import warnings
    warnings.warn(
        "multimind.compliance.cli legacy interface not found. If you rely on these functions, please update your code.",
        DeprecationWarning
    )

try:
    from .api import *
except ImportError:
    import warnings
    warnings.warn(
        "multimind.compliance.api legacy interface not found. If you rely on these functions, please update your code.",
        DeprecationWarning
    )

__version__ = '1.0.0' 