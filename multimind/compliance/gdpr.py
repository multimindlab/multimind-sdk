"""
GDPR compliance implementation.

Besides the :class:`GDPRCompliance` registry, this module provides runtime
enforcement utilities: :class:`GDPRPolicy` evaluates concrete processing
requests against a configured lawful-basis/purpose/retention policy, and
:func:`gdpr_enforce` translates that policy into settings for the runtime
:class:`~multimind.compliance.guard.ComplianceGuard`.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .governance import ComplianceMetadata, DataCategory, GovernanceConfig, Regulation

# GDPR Article 6(1) lawful bases for processing.
GDPR_LAWFUL_BASES = (
    "consent",
    "contract",
    "legal_obligation",
    "vital_interests",
    "public_task",
    "legitimate_interests",
)

# PII types (as detected by the runtime guard) that map to GDPR special-category
# or high-risk identifier data and are blocked when no purpose permits
# DataCategory.SENSITIVE.
SPECIAL_CATEGORY_PII_TYPES = ("ssn", "credit_card", "iban", "passport", "dob")


def _category_value(category: Union[DataCategory, str]) -> str:
    """Normalize a category to its string value; raises on unknown categories."""
    if isinstance(category, DataCategory):
        return category.value
    return DataCategory(category).value


class ProcessingDecision(BaseModel):
    """Outcome of a GDPR processing check.

    ``status`` is one of ``"allowed"``, ``"denied"``, or ``"not_configured"``
    (the purpose is unknown to the policy — the policy refuses to guess).
    """

    allowed: bool
    status: str
    purpose: str
    reasons: List[str] = Field(default_factory=list)


class PurposeRule(BaseModel):
    """Configured processing purpose: what it may touch and on which basis."""

    purpose: str
    allowed_categories: List[DataCategory]
    lawful_basis: str
    retention_days: Optional[int] = None

    @field_validator("lawful_basis")
    @classmethod
    def _validate_basis(cls, v: str) -> str:
        if v not in GDPR_LAWFUL_BASES:
            raise ValueError(f"Unknown lawful basis {v!r}; must be one of {GDPR_LAWFUL_BASES}")
        return v

    @field_validator("retention_days")
    @classmethod
    def _validate_retention(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("retention_days must be positive")
        return v


class GDPRPolicy(BaseModel):
    """Evaluable GDPR policy: lawful-basis table, purpose limitation, retention.

    Purposes must be configured explicitly. A check against a purpose the
    policy does not know returns a ``not_configured`` denial rather than a
    fabricated pass.
    """

    config: GovernanceConfig
    purposes: Dict[str, PurposeRule] = Field(default_factory=dict)

    def register_purpose(self, rule: PurposeRule) -> None:
        """Register (or replace) a processing purpose rule."""
        self.purposes[rule.purpose] = rule

    def retention_cap_days(self, purpose: str) -> int:
        """Effective retention cap for a purpose (rule cap, else config default)."""
        rule = self.purposes.get(purpose)
        if rule is not None and rule.retention_days is not None:
            return rule.retention_days
        return self.config.data_retention_days

    def check_processing(
        self,
        purpose: str,
        data_categories: List[Union[DataCategory, str]],
        consent: bool = False,
        retention_days: Optional[int] = None,
    ) -> ProcessingDecision:
        """Evaluate a concrete processing request against the policy.

        Checks, in order: purpose is configured; every data category is
        permitted for the purpose (purpose limitation, Art. 5(1)(b)); the
        purpose's lawful basis is satisfiable (consent-based purposes and any
        special-category/sensitive processing require ``consent=True``); the
        requested retention does not exceed the effective cap (storage
        limitation, Art. 5(1)(e)).
        """
        rule = self.purposes.get(purpose)
        if rule is None:
            return ProcessingDecision(
                allowed=False,
                status="not_configured",
                purpose=purpose,
                reasons=[
                    f"Purpose {purpose!r} is not configured in the GDPR policy; "
                    "processing cannot be authorized for unknown purposes."
                ],
            )

        categories = [_category_value(c) for c in data_categories]
        allowed_categories = {_category_value(c) for c in rule.allowed_categories}
        reasons: List[str] = []

        for category in categories:
            if category not in allowed_categories:
                reasons.append(
                    f"Purpose limitation: category {category!r} is not permitted "
                    f"for purpose {purpose!r} (allowed: {sorted(allowed_categories)})."
                )

        if rule.lawful_basis == "consent" and not consent:
            reasons.append(
                f"Lawful basis for {purpose!r} is 'consent' but consent was not granted."
            )
        if DataCategory.SENSITIVE.value in categories and not consent:
            reasons.append(
                "Special-category (sensitive) data requires explicit consent "
                "(Art. 9) but consent was not granted."
            )

        cap = self.retention_cap_days(purpose)
        if retention_days is not None and retention_days > cap:
            reasons.append(
                f"Retention of {retention_days} days exceeds the configured cap "
                f"of {cap} days for purpose {purpose!r}."
            )

        if reasons:
            return ProcessingDecision(
                allowed=False, status="denied", purpose=purpose, reasons=reasons
            )
        return ProcessingDecision(
            allowed=True,
            status="allowed",
            purpose=purpose,
            reasons=[
                f"Purpose {purpose!r} permits categories {sorted(set(categories))} "
                f"on lawful basis {rule.lawful_basis!r}; retention within {cap} days."
            ],
        )


def gdpr_enforce(policy: GDPRPolicy, **overrides: Any) -> Dict[str, Any]:
    """Derive runtime ComplianceGuard settings from a GDPR policy.

    Returns keyword arguments to splat into
    ``ComplianceGuard(model, **gdpr_enforce(policy))``:

    - ``block_on``: when no configured purpose permits sensitive data, the
      special-category/high-risk PII types are blocked outright.
    - ``strategy``: ``"hash"`` (stable pseudonymous tags) when the governance
      config enables pseudonymization, else ``"mask"``.
    - input/output redaction always on — GDPR data minimisation applies in
      both directions.

    Explicit ``overrides`` win over derived settings (e.g. pass
    ``audit_log=...`` to satisfy the config's audit-logging requirement; this
    helper cannot invent a log destination).
    """
    allows_sensitive = any(
        DataCategory.SENSITIVE.value in {_category_value(c) for c in rule.allowed_categories}
        for rule in policy.purposes.values()
    )
    kwargs: Dict[str, Any] = {
        "redact_input": True,
        "redact_output": True,
        "strategy": "hash" if policy.config.enable_pseudonymization else "mask",
        "block_on": () if allows_sensitive else SPECIAL_CATEGORY_PII_TYPES,
    }
    kwargs.update(overrides)
    return kwargs


class GDPRCompliance(BaseModel):
    """GDPR compliance manager."""

    config: GovernanceConfig
    data_registry: Dict[str, ComplianceMetadata] = Field(default_factory=dict)

    async def process_data(
        self, data_id: str, content: Any, metadata: Dict[str, Any]
    ) -> ComplianceMetadata:
        """Process data according to GDPR requirements."""
        # Create compliance metadata
        compliance_metadata = ComplianceMetadata(
            data_category=metadata.get("data_category", DataCategory.PERSONAL),
            risk_level=metadata.get("risk_level"),
            regulation_tags=[Regulation.GDPR],
            lawful_basis=metadata.get("lawful_basis"),
            consent_granted=metadata.get("consent_granted", False),
            data_subject_id=metadata.get("data_subject_id"),
            expires_at=datetime.now() + timedelta(days=self.config.data_retention_days),
        )

        # Store metadata
        self.data_registry[data_id] = compliance_metadata

        return compliance_metadata

    async def handle_dsar(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle Data Subject Access Request."""
        # Find all data for subject
        subject_data = {
            data_id: metadata
            for data_id, metadata in self.data_registry.items()
            if metadata.data_subject_id == data_subject_id
        }

        return {
            "data_subject_id": data_subject_id,
            "requested_at": datetime.now(),
            "data_items": subject_data,
        }

    async def handle_erasure(self, data_subject_id: str) -> bool:
        """Handle data erasure request."""
        # Find and remove all data for subject
        data_to_remove = [
            data_id
            for data_id, metadata in self.data_registry.items()
            if metadata.data_subject_id == data_subject_id
        ]

        for data_id in data_to_remove:
            del self.data_registry[data_id]

        return len(data_to_remove) > 0

    async def check_retention(self) -> List[str]:
        """Check for data that needs to be deleted due to retention policy."""
        now = datetime.now()
        expired_data = [
            data_id
            for data_id, metadata in self.data_registry.items()
            if metadata.expires_at and metadata.expires_at < now
        ]

        return expired_data

    async def validate_lawful_basis(
        self, data_category: DataCategory, lawful_basis: str, consent_granted: bool
    ) -> bool:
        """Validate if the lawful basis is appropriate for the data category."""
        if data_category in [DataCategory.PERSONAL, DataCategory.SENSITIVE]:
            if not lawful_basis or not consent_granted:
                return False

        return True

    async def get_processing_activities(self) -> List[Dict[str, Any]]:
        """Get record of processing activities."""
        return [
            {
                "data_id": data_id,
                "metadata": metadata.dict(),
                "last_accessed": metadata.last_accessed,
                "access_count": metadata.access_count,
            }
            for data_id, metadata in self.data_registry.items()
        ]

    async def update_metadata(
        self, data_id: str, updates: Dict[str, Any]
    ) -> Optional[ComplianceMetadata]:
        """Update compliance metadata for data."""
        if data_id not in self.data_registry:
            return None

        metadata = self.data_registry[data_id]
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        metadata.version += 1
        return metadata
