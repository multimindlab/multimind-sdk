"""
Governance configuration for compliance management.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timedelta

class Regulation(str, Enum):
    """Compliance regulations."""
    
    GDPR = "GDPR"  # General Data Protection Regulation
    AI_ACT = "AI_ACT"  # EU AI Act
    HIPAA = "HIPAA"  # Health Insurance Portability and Accountability Act
    HITECH = "HITECH"  # Health Information Technology for Economic and Clinical Health Act
    ISO27001 = "ISO27001"  # ISO/IEC 27001
    ISO27701 = "ISO27701"  # ISO/IEC 27701
    ISO31000 = "ISO31000"  # ISO 31000
    PIPEDA = "PIPEDA"  # Personal Information Protection and Electronic Documents Act
    LGPD = "LGPD"  # Lei Geral de Proteção de Dados
    NIST = "NIST"  # National Institute of Standards and Technology
    SOC2 = "SOC2"  # Service Organization Control 2
    PCI_DSS = "PCI_DSS"  # Payment Card Industry Data Security Standard
    FERPA = "FERPA"  # Family Educational Rights and Privacy Act
    COPPA = "COPPA"  # Children's Online Privacy Protection Act
    GLBA = "GLBA"  # Gramm-Leach-Bliley Act
    SOX = "SOX"  # Sarbanes-Oxley Act
    FISMA = "FISMA"  # Federal Information Security Management Act
    CMMC = "CMMC"  # Cybersecurity Maturity Model Certification
    APPI = "APPI"  # Act on the Protection of Personal Information (Japan)
    POPIA = "POPIA"  # Protection of Personal Information Act (South Africa)
    PDPA = "PDPA"  # Personal Data Protection Act (Singapore)
    PDPO = "PDPO"  # Personal Data (Privacy) Ordinance (Hong Kong)
    KVKK = "KVKK"  # Kişisel Verilerin Korunması Kanunu (Turkey)
    PDPL = "PDPL"  # Personal Data Protection Law (Saudi Arabia)
    PDPB = "PDPB"  # Personal Data Protection Bill (India)
    PIPL = "PIPL"  # Personal Information Protection Law (China)
    
    # New regulations and standards
    EPRIVACY = "EPRIVACY"  # ePrivacy Directive/Regulation
    DORA = "DORA"  # Digital Operational Resilience Act
    OECD_AI = "OECD_AI"  # OECD AI Principles
    UN_GUIDING = "UN_GUIDING"  # UN Guiding Principles on Business & Human Rights
    UK_AI = "UK_AI"  # UK AI Regulation
    US_AI_RIGHTS = "US_AI_RIGHTS"  # U.S. AI Bill of Rights
    SCHREMS_II = "SCHREMS_II"  # Schrems II / Standard Contractual Clauses
    BCR = "BCR"  # Binding Corporate Rules
    WCAG = "WCAG"  # Web Content Accessibility Guidelines
    ADA = "ADA"  # Americans with Disabilities Act
    SIG = "SIG"  # Standard Information Gathering
    CAIQ = "CAIQ"  # Consensus Assessments Initiative Questionnaire
    SCA = "SCA"  # Software Composition Analysis
    BCP = "BCP"  # Business Continuity Planning
    DR = "DR"  # Disaster Recovery

class RiskLevel(Enum):
    """AI system risk levels."""
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"

class DataCategory(Enum):
    """Data classification categories."""
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    PUBLIC = "public"
    RESTRICTED = "restricted"

class GovernanceConfig(BaseModel):
    """Configuration for compliance governance."""
    
    # Organization settings
    organization_id: str
    organization_name: str
    dpo_email: str
    dpo_phone: Optional[str] = None
    
    # Regulation settings
    enabled_regulations: List[Regulation] = Field(
        default=[Regulation.GDPR, Regulation.AI_ACT],
        description="List of regulations to enforce"
    )
    
    # Retention settings
    data_retention_days: int = Field(
        default=365,
        description="Default data retention period in days"
    )
    audit_log_retention_days: int = Field(
        default=730,
        description="Audit log retention period in days"
    )
    
    # Risk assessment settings
    risk_assessment_threshold: float = Field(
        default=0.7,
        description="Threshold for triggering risk assessment"
    )
    enable_continuous_monitoring: bool = Field(
        default=True,
        description="Enable continuous risk monitoring"
    )
    
    # Data protection settings
    enable_encryption: bool = Field(
        default=True,
        description="Enable data encryption"
    )
    enable_pseudonymization: bool = Field(
        default=True,
        description="Enable data pseudonymization"
    )
    
    # Audit settings
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    audit_log_level: str = Field(
        default="INFO",
        description="Audit log level"
    )
    
    # Policy settings
    policy_update_interval: int = Field(
        default=30,
        description="Policy update check interval in days"
    )
    
    # Documentation settings
    enable_auto_documentation: bool = Field(
        default=True,
        description="Enable automatic documentation generation"
    )
    documentation_update_interval: int = Field(
        default=90,
        description="Documentation update interval in days"
    )
    
    # Custom settings
    custom_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom compliance settings"
    )
    
    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True
    )

class ComplianceMetadata(BaseModel):
    """Metadata for compliance tracking."""

    data_category: DataCategory
    risk_level: RiskLevel
    regulation_tags: List[Regulation]
    lawful_basis: Optional[str] = None
    consent_granted: bool = False
    data_subject_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    version: int = 1
    metadata_hash: Optional[str] = None

    model_config = ConfigDict(
        use_enum_values=True
    )