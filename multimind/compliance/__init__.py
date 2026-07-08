"""MultiMind Compliance Module.

Comprehensive compliance monitoring and evaluation: privacy, security, and
regulatory features (GDPR, HIPAA, NIS2, …).

Requires the ``compliance`` extras (``cryptography``, ``bcrypt``, ``pycryptodome``):
``pip install 'multimind-sdk[compliance]'``.
"""

import os
import warnings

# Runtime guard and evidence reporting are stdlib-only, and regulatory
# watching only needs httpx (a core dep), so these are imported outside the
# extras gate.
from .guard import (
    AuditLog,
    ComplianceGuard,
    ComplianceViolationError,
    PIIDetector,
    guard,
)
from .regulatory_watch import ChangeEvent, RegulatoryWatcher
from .reporting import EvidenceReport, build_evidence_report

try:
    from .advanced import (
        AdaptivePrivacy,
        ComplianceLevel,
        ComplianceMetrics,
        ComplianceShard,
        ExplainableDTO,
        FederatedCompliance,
        ModelWatermarking,
        RegulatoryChangeDetector,
        SelfHealingCompliance,
    )
    from .advanced_config import (
        AdaptivePrivacyConfig,
        ComplianceShardConfig,
        ExplainableDTOConfig,
        FederatedComplianceConfig,
        ModelWatermarkingConfig,
        RegulatoryChangeConfig,
        SelfHealingConfig,
        load_advanced_config,
        save_advanced_config,
    )
    from .gdpr import (
        GDPRCompliance,
        GDPRPolicy,
        ProcessingDecision,
        PurposeRule,
        gdpr_enforce,
    )
    from .governance import GovernanceConfig, Regulation
    from .model_training import ComplianceTrainer
    from .privacy import (
        AuditAction,
        ComplianceStatus,
        DataCategory,
        NotificationType,
        PrivacyCompliance,
    )

    _EXTRAS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised on minimal installs
    # Degrade gracefully: the stdlib-only guard stays importable; gated names
    # raise a helpful error at access time via __getattr__ below.
    _EXTRAS_IMPORT_ERROR = exc


def __getattr__(name: str):
    if _EXTRAS_IMPORT_ERROR is not None and name in _EXTRAS_GATED_NAMES:
        raise ImportError(
            f"multimind.compliance.{name} requires additional dependencies. "
            "Install with: pip install 'multimind-sdk[compliance]'"
        ) from _EXTRAS_IMPORT_ERROR
    raise AttributeError(f"module 'multimind.compliance' has no attribute {name!r}")


def _log_legacy_warning(message: str) -> None:
    """Log legacy warning only if explicitly enabled."""
    show_warnings = os.getenv("MULTIMIND_SHOW_LEGACY_WARNINGS", "false").lower() == "true"
    if show_warnings:
        warnings.warn(message)


__all__ = [
    # Runtime Guard
    "ComplianceGuard",
    "PIIDetector",
    "guard",
    "ComplianceViolationError",
    "AuditLog",
    # Evidence Reporting
    "EvidenceReport",
    "build_evidence_report",
    # Regulatory Watch
    "ChangeEvent",
    "RegulatoryWatcher",
    # Advanced Features
    "ComplianceShard",
    "SelfHealingCompliance",
    "ExplainableDTO",
    "ModelWatermarking",
    "AdaptivePrivacy",
    "RegulatoryChangeDetector",
    "FederatedCompliance",
    "ComplianceLevel",
    "ComplianceMetrics",
    # Advanced Configurations
    "ComplianceShardConfig",
    "SelfHealingConfig",
    "ExplainableDTOConfig",
    "ModelWatermarkingConfig",
    "AdaptivePrivacyConfig",
    "RegulatoryChangeConfig",
    "FederatedComplianceConfig",
    "load_advanced_config",
    "save_advanced_config",
    # Governance
    "GovernanceConfig",
    "Regulation",
    # GDPR
    "GDPRCompliance",
    "GDPRPolicy",
    "ProcessingDecision",
    "PurposeRule",
    "gdpr_enforce",
    # Privacy
    "PrivacyCompliance",
    "DataCategory",
    "NotificationType",
    "AuditAction",
    "ComplianceStatus",
    # Training
    "ComplianceTrainer",
]

_GUARD_NAMES = {
    "ComplianceGuard",
    "PIIDetector",
    "guard",
    "ComplianceViolationError",
    "AuditLog",
    "EvidenceReport",
    "build_evidence_report",
    "ChangeEvent",
    "RegulatoryWatcher",
}
_EXTRAS_GATED_NAMES = set(__all__) - _GUARD_NAMES

# Backward compatibility: import legacy CLI and API functions if available
try:
    from .cli import configure_alerts, generate_report, run_example, show_alerts, show_dashboard

    __all__.extend(
        [
            "run_example",
            "generate_report",
            "show_dashboard",
            "show_alerts",
            "configure_alerts",
        ]
    )
except ImportError:
    _log_legacy_warning(
        "multimind.compliance.cli legacy interface not found. If you rely on these functions, please update your code."
    )

try:
    from .api import *
except ImportError:
    _log_legacy_warning(
        "multimind.compliance.api legacy interface not found. If you rely on these functions, please update your code."
    )

__version__ = "1.0.0"
