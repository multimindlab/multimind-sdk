"""MultiMind Compliance Module.

Comprehensive compliance monitoring and evaluation: privacy, security, and
regulatory features (GDPR, HIPAA, NIS2, …).

Requires the ``compliance`` extras (``cryptography``, ``bcrypt``, ``pycryptodome``):
``pip install 'multimind-sdk[compliance]'``.
"""

import os
import warnings

try:
    from .advanced_config import (
        ComplianceShardConfig,
        SelfHealingConfig,
        ExplainableDTOConfig,
        ModelWatermarkingConfig,
        AdaptivePrivacyConfig,
        RegulatoryChangeConfig,
        FederatedComplianceConfig,
        load_advanced_config,
        save_advanced_config,
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
        ComplianceMetrics,
    )
    from .governance import GovernanceConfig, Regulation
    from .model_training import ComplianceTrainer
    from .privacy import (
        PrivacyCompliance,
        DataCategory,
        NotificationType,
        AuditAction,
        ComplianceStatus,
    )
except ImportError as exc:  # pragma: no cover - exercised on minimal installs
    raise ImportError(
        "Compliance features require additional dependencies. "
        "Install with: pip install 'multimind-sdk[compliance]'"
    ) from exc

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