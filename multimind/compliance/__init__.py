"""
MultiMind Compliance Module

This module provides comprehensive compliance monitoring and evaluation capabilities,
including advanced features for privacy, security, and regulatory compliance.
"""

import os
import warnings
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
from .privacy import PrivacyCompliance, DataCategory, NotificationType, AuditAction, ComplianceStatus
from multimind.cli.compliance import run_compliance

def _log_legacy_warning(message: str) -> None:
    """Log legacy warning only if explicitly enabled."""
    show_warnings = os.getenv('MULTIMIND_SHOW_LEGACY_WARNINGS', 'false').lower() == 'true'
    if show_warnings:
        warnings.warn(message)

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
    # Privacy
    'PrivacyCompliance',
    'DataCategory',
    'NotificationType',
    'AuditAction',
    'ComplianceStatus',
    # Training
    'ComplianceTrainer',
    # CLI
    'run_compliance',
]

# Backward compatibility: import legacy CLI and API functions if available
try:
    from .cli import (
        run_example,
        generate_report,
        show_dashboard,
        show_alerts,
        configure_alerts
    )
    __all__.extend([
        'run_example',
        'generate_report',
        'show_dashboard',
        'show_alerts',
        'configure_alerts',
    ])
except ImportError:
    _log_legacy_warning("multimind.compliance.cli legacy interface not found. If you rely on these functions, please update your code.")

try:
    from .api import *
except ImportError:
    _log_legacy_warning(
        "multimind.compliance.api legacy interface not found. If you rely on these functions, please update your code."
    )

__version__ = '1.0.0'