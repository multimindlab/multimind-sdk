"""
MultiMind Compliance Module

This module provides comprehensive compliance monitoring and evaluation capabilities,
including advanced features for privacy, security, and regulatory compliance.
"""

from .advanced import (
    ComplianceShard,
    SelfHealingCompliance,
    ExplainableDTO,
    ModelWatermarking,
    AdaptivePrivacy,
    RegulatoryChangeDetector,
    FederatedCompliance
)

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

from .api import (
    ComplianceConfig,
    ComplianceResult,
    DashboardMetrics,
    start_api_server
)

from .cli import (
    run_compliance,
    run_example,
    generate_report,
    show_dashboard,
    show_alerts,
    configure_alerts
)

__all__ = [
    # Advanced Features
    'ComplianceShard',
    'SelfHealingCompliance',
    'ExplainableDTO',
    'ModelWatermarking',
    'AdaptivePrivacy',
    'RegulatoryChangeDetector',
    'FederatedCompliance',
    
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
    
    # API Components
    'ComplianceConfig',
    'ComplianceResult',
    'DashboardMetrics',
    'start_api_server',
    
    # CLI Commands
    'run_compliance',
    'run_example',
    'generate_report',
    'show_dashboard',
    'show_alerts',
    'configure_alerts'
]

__version__ = '1.0.0' 