"""
Healthcare compliance implementation for HIPAA and HITECH.

Validation checks are evaluated against the manager's actual state (config
flags, PHI records, access log, breach log). Checks that depend on external
systems report status "unverifiable" instead of a fabricated pass.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .governance import GovernanceConfig
from .guard import AuditLog

# HIPAA Breach Notification Rule: notify without unreasonable delay, at most 60 days.
BREACH_NOTIFICATION_DEADLINE_DAYS = 60


class PHIData(BaseModel):
    """Protected Health Information (PHI) data model."""

    data_id: str
    patient_id: str
    data_type: str
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    access_count: int = 0


class HealthcareCompliance(BaseModel):
    """Healthcare compliance manager for HIPAA and HITECH."""

    config: GovernanceConfig
    phi_data: Dict[str, PHIData] = Field(default_factory=dict)
    breach_log: List[Dict[str, Any]] = Field(default_factory=list)
    access_log: List[Dict[str, Any]] = Field(default_factory=list)
    audit_log: Optional[AuditLog] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def process_phi(
        self,
        data_id: str,
        patient_id: str,
        content: Any,
        data_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PHIData:
        """Process Protected Health Information (PHI)."""
        phi_data = PHIData(
            data_id=data_id,
            patient_id=patient_id,
            data_type=data_type,
            content=content,
            metadata=metadata or {},
        )

        self.phi_data[data_id] = phi_data
        self._audit({"event": "phi_stored", "data_id": data_id, "data_type": data_type})

        return phi_data

    async def access_phi(self, data_id: str, user_id: str, purpose: str) -> Optional[PHIData]:
        """Access PHI data with audit logging."""
        if data_id not in self.phi_data:
            return None

        phi_data = self.phi_data[data_id]
        phi_data.last_accessed = datetime.now()
        phi_data.access_count += 1

        await self._log_access(data_id, user_id, purpose)

        return phi_data

    async def report_breach(
        self, breach_type: str, affected_data: List[str], description: str, severity: str
    ) -> Dict[str, Any]:
        """Report a PHI data breach."""
        breach = {
            # Use UUID to avoid non-unique breach IDs if the log is pruned/reset.
            "breach_id": f"breach_{uuid.uuid4()}",
            "timestamp": datetime.now(),
            "breach_type": breach_type,
            "affected_data": affected_data,
            "description": description,
            "severity": severity,
            "status": "reported",
            "resolution": None,
            "resolved_at": None,
            "notification_logged_at": None,
        }

        self.breach_log.append(breach)
        self._audit(
            {
                "event": "breach_reported",
                "breach_id": breach["breach_id"],
                "breach_type": breach_type,
                "severity": severity,
                "affected_count": len(affected_data),
            }
        )

        if severity in ["high", "critical"]:
            await self._trigger_breach_notification(breach)

        return breach

    async def get_phi_access_log(
        self,
        data_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get PHI access log, optionally filtered."""
        entries = []
        for entry in self.access_log:
            if data_id is not None and entry["data_id"] != data_id:
                continue
            if patient_id is not None and entry.get("patient_id") != patient_id:
                continue
            if start_time is not None and entry["timestamp"] < start_time:
                continue
            if end_time is not None and entry["timestamp"] > end_time:
                continue
            entries.append(entry)
        return entries

    async def validate_hipaa_compliance(self, system_id: str) -> Dict[str, Any]:
        """Validate HIPAA compliance requirements against actual state."""
        requirements = [
            self._check_privacy_rule(),
            self._check_security_rule(),
            self._check_breach_notification(),
            self._check_data_retention(),
            {
                "requirement": "enforcement_rule",
                "status": "unverifiable",
                "details": (
                    "Enforcement Rule compliance depends on external OCR audits "
                    "and organizational processes; it cannot be verified from SDK state"
                ),
            },
        ]

        status = self._aggregate_status(requirements)
        validation = {
            "system_id": system_id,
            "validated_at": datetime.now(),
            "requirements": requirements,
            "status": status,
        }
        self._audit({"event": "hipaa_validation", "system_id": system_id, "status": status})
        return validation

    async def validate_hitech_compliance(self, system_id: str) -> Dict[str, Any]:
        """Validate HITECH compliance requirements against actual state."""
        requirements = [
            self._check_breach_notification(),
            self._check_security_rule(),
            {
                "requirement": "meaningful_use",
                "status": "unverifiable",
                "details": (
                    "Meaningful Use requires certified EHR usage metrics from "
                    "external systems; it cannot be verified from SDK state"
                ),
            },
            {
                "requirement": "health_information_exchange",
                "status": "unverifiable",
                "details": (
                    "HIE compliance requires inspection of external exchange "
                    "infrastructure; it cannot be verified from SDK state"
                ),
            },
        ]

        status = self._aggregate_status(requirements)
        validation = {
            "system_id": system_id,
            "validated_at": datetime.now(),
            "requirements": requirements,
            "status": status,
        }
        self._audit({"event": "hitech_validation", "system_id": system_id, "status": status})
        return validation

    def _check_privacy_rule(self) -> Dict[str, Any]:
        """Privacy Rule: audit logging enabled and every PHI access has a log entry."""
        if not self.config.enable_audit_logging:
            return {
                "requirement": "privacy_rule",
                "status": "non_compliant",
                "details": "Audit logging is disabled in governance config",
            }

        logged_counts: Dict[str, int] = {}
        for entry in self.access_log:
            logged_counts[entry["data_id"]] = logged_counts.get(entry["data_id"], 0) + 1
        unlogged = [
            data_id
            for data_id, phi in self.phi_data.items()
            if phi.access_count > logged_counts.get(data_id, 0)
        ]
        if unlogged:
            return {
                "requirement": "privacy_rule",
                "status": "non_compliant",
                "details": f"PHI records accessed without audit trail: {sorted(unlogged)}",
            }

        return {
            "requirement": "privacy_rule",
            "status": "compliant",
            "details": (
                f"Audit logging enabled; all {len(self.access_log)} recorded "
                f"accesses across {len(self.phi_data)} PHI records have audit entries"
            ),
        }

    def _check_security_rule(self) -> Dict[str, Any]:
        """Security Rule: technical safeguards configured (encryption, pseudonymization)."""
        missing = []
        if not self.config.enable_encryption:
            missing.append("encryption")
        if not self.config.enable_pseudonymization:
            missing.append("pseudonymization")
        if missing:
            return {
                "requirement": "security_rule",
                "status": "non_compliant",
                "details": f"Technical safeguards disabled in governance config: {missing}",
            }
        return {
            "requirement": "security_rule",
            "status": "compliant",
            "details": "Encryption and pseudonymization safeguards are enabled",
        }

    def _check_breach_notification(self) -> Dict[str, Any]:
        """Breach Notification Rule: reportable breaches notified and resolved in time."""
        if not self.breach_log:
            return {
                "requirement": "breach_notification",
                "status": "compliant",
                "details": "No breaches recorded",
            }

        problems = []
        deadline = timedelta(days=BREACH_NOTIFICATION_DEADLINE_DAYS)
        now = datetime.now()
        for breach in self.breach_log:
            if breach["severity"] in ["high", "critical"] and not breach.get(
                "notification_logged_at"
            ):
                problems.append(f"{breach['breach_id']}: no notification logged")
            if breach["status"] == "reported" and now - breach["timestamp"] > deadline:
                problems.append(
                    f"{breach['breach_id']}: unresolved past "
                    f"{BREACH_NOTIFICATION_DEADLINE_DAYS}-day notification window"
                )

        if problems:
            return {
                "requirement": "breach_notification",
                "status": "non_compliant",
                "details": "; ".join(problems),
            }
        return {
            "requirement": "breach_notification",
            "status": "compliant",
            "details": (
                f"All {len(self.breach_log)} recorded breaches have notifications "
                "logged where required and none exceed the notification window"
            ),
        }

    def _check_data_retention(self) -> Dict[str, Any]:
        """Retention: no PHI records held past the configured retention period."""
        limit = timedelta(days=self.config.data_retention_days)
        now = datetime.now()
        expired = sorted(
            data_id for data_id, phi in self.phi_data.items() if now - phi.created_at > limit
        )
        if expired:
            return {
                "requirement": "data_retention",
                "status": "non_compliant",
                "details": (
                    f"PHI records held past {self.config.data_retention_days}-day "
                    f"retention period: {expired}"
                ),
            }
        return {
            "requirement": "data_retention",
            "status": "compliant",
            "details": (
                f"All {len(self.phi_data)} PHI records within the "
                f"{self.config.data_retention_days}-day retention period"
            ),
        }

    @staticmethod
    def _aggregate_status(requirements: List[Dict[str, Any]]) -> str:
        statuses = {req["status"] for req in requirements}
        if "non_compliant" in statuses:
            return "non_compliant"
        if "unverifiable" in statuses:
            return "unverifiable"
        return "compliant"

    async def _log_access(self, data_id: str, user_id: str, purpose: str) -> None:
        """Log PHI access to the in-memory access log and the audit trail."""
        phi = self.phi_data.get(data_id)
        entry = {
            "event": "phi_access",
            "timestamp": datetime.now(),
            "data_id": data_id,
            "patient_id": phi.patient_id if phi else None,
            "user_id": user_id,
            "purpose": purpose,
        }
        self.access_log.append(entry)
        self._audit(
            {
                "event": "phi_access",
                "data_id": data_id,
                "user_id": user_id,
                "purpose": purpose,
            }
        )

    async def _trigger_breach_notification(self, breach: Dict[str, Any]) -> None:
        """Record the breach notification in the audit trail.

        Honest scope: this logs that notification is required and addressed to
        the DPO; delivering notices to individuals/HHS needs external systems.
        """
        breach["notification_logged_at"] = datetime.now()
        self._audit(
            {
                "event": "breach_notification",
                "breach_id": breach["breach_id"],
                "severity": breach["severity"],
                "dpo_email": self.config.dpo_email,
                "affected_count": len(breach["affected_data"]),
            }
        )

    def _audit(self, record: Dict[str, Any]) -> None:
        if self.audit_log is not None:
            self.audit_log.write(record)
