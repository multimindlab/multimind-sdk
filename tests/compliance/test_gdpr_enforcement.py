"""Known-answer tests for GDPR runtime enforcement (GDPRPolicy / gdpr_enforce)."""

import pytest
from pydantic import ValidationError

from multimind.compliance.gdpr import (
    GDPR_LAWFUL_BASES,
    SPECIAL_CATEGORY_PII_TYPES,
    GDPRPolicy,
    ProcessingDecision,
    PurposeRule,
    gdpr_enforce,
)
from multimind.compliance.governance import DataCategory, GovernanceConfig
from multimind.compliance.guard import ComplianceGuard, ComplianceViolationError


def _config(**kwargs) -> GovernanceConfig:
    defaults = dict(
        organization_id="org-1",
        organization_name="Test Org",
        dpo_email="dpo@example.org",
        data_retention_days=365,
    )
    defaults.update(kwargs)
    return GovernanceConfig(**defaults)


def _policy() -> GDPRPolicy:
    policy = GDPRPolicy(config=_config())
    policy.register_purpose(
        PurposeRule(
            purpose="support",
            allowed_categories=[DataCategory.PERSONAL, DataCategory.PUBLIC],
            lawful_basis="consent",
        )
    )
    policy.register_purpose(
        PurposeRule(
            purpose="fraud_detection",
            allowed_categories=[DataCategory.PERSONAL],
            lawful_basis="legal_obligation",
            retention_days=90,
        )
    )
    return policy


class TestCheckProcessing:
    def test_consent_purpose_allowed_with_consent(self):
        decision = _policy().check_processing(
            "support", [DataCategory.PERSONAL], consent=True
        )
        assert isinstance(decision, ProcessingDecision)
        assert decision.allowed is True
        assert decision.status == "allowed"
        assert decision.reasons  # rationale is stated, not silent

    def test_consent_purpose_denied_without_consent(self):
        decision = _policy().check_processing(
            "support", [DataCategory.PERSONAL], consent=False
        )
        assert decision.allowed is False
        assert decision.status == "denied"
        assert any("consent" in r for r in decision.reasons)

    def test_unknown_purpose_is_not_configured_not_a_guess(self):
        decision = _policy().check_processing(
            "marketing", [DataCategory.PERSONAL], consent=True
        )
        assert decision.allowed is False
        assert decision.status == "not_configured"
        assert any("not configured" in r for r in decision.reasons)

    def test_purpose_limitation_denies_unlisted_category(self):
        decision = _policy().check_processing(
            "support", [DataCategory.SENSITIVE], consent=True
        )
        assert decision.allowed is False
        assert any("Purpose limitation" in r for r in decision.reasons)

    def test_sensitive_requires_consent_even_on_non_consent_basis(self):
        policy = _policy()
        policy.register_purpose(
            PurposeRule(
                purpose="medical",
                allowed_categories=[DataCategory.SENSITIVE],
                lawful_basis="legal_obligation",
            )
        )
        denied = policy.check_processing("medical", [DataCategory.SENSITIVE], consent=False)
        assert denied.allowed is False
        assert any("Art. 9" in r for r in denied.reasons)
        allowed = policy.check_processing("medical", [DataCategory.SENSITIVE], consent=True)
        assert allowed.allowed is True

    def test_non_consent_basis_allows_without_consent(self):
        decision = _policy().check_processing(
            "fraud_detection", [DataCategory.PERSONAL], consent=False
        )
        assert decision.allowed is True

    def test_retention_over_purpose_cap_denied(self):
        decision = _policy().check_processing(
            "fraud_detection", [DataCategory.PERSONAL], retention_days=120
        )
        assert decision.allowed is False
        assert any("Retention" in r and "90" in r for r in decision.reasons)

    def test_retention_falls_back_to_config_default(self):
        policy = _policy()
        assert policy.retention_cap_days("support") == 365
        over = policy.check_processing(
            "support", [DataCategory.PERSONAL], consent=True, retention_days=400
        )
        assert over.allowed is False
        within = policy.check_processing(
            "support", [DataCategory.PERSONAL], consent=True, retention_days=365
        )
        assert within.allowed is True

    def test_multiple_violations_all_reported(self):
        decision = _policy().check_processing(
            "support",
            [DataCategory.RESTRICTED],
            consent=False,
            retention_days=1000,
        )
        assert decision.allowed is False
        assert len(decision.reasons) == 3

    def test_string_categories_accepted(self):
        decision = _policy().check_processing("support", ["personal"], consent=True)
        assert decision.allowed is True

    def test_unknown_category_raises_loudly(self):
        with pytest.raises(ValueError):
            _policy().check_processing("support", ["telepathic"], consent=True)

    def test_unknown_lawful_basis_rejected_at_construction(self):
        with pytest.raises(ValidationError):
            PurposeRule(
                purpose="bad",
                allowed_categories=[DataCategory.PERSONAL],
                lawful_basis="because_we_want_to",
            )

    def test_lawful_bases_are_the_article_6_set(self):
        assert set(GDPR_LAWFUL_BASES) == {
            "consent",
            "contract",
            "legal_obligation",
            "vital_interests",
            "public_task",
            "legitimate_interests",
        }


class TestGdprEnforce:
    def test_blocks_special_categories_when_no_sensitive_purpose(self):
        kwargs = gdpr_enforce(_policy())
        assert kwargs["block_on"] == SPECIAL_CATEGORY_PII_TYPES
        assert kwargs["redact_input"] is True
        assert kwargs["redact_output"] is True

    def test_no_block_when_a_purpose_permits_sensitive(self):
        policy = _policy()
        policy.register_purpose(
            PurposeRule(
                purpose="medical",
                allowed_categories=[DataCategory.SENSITIVE],
                lawful_basis="consent",
            )
        )
        assert gdpr_enforce(policy)["block_on"] == ()

    def test_strategy_follows_pseudonymization_setting(self):
        pseudo = GDPRPolicy(config=_config(enable_pseudonymization=True))
        plain = GDPRPolicy(config=_config(enable_pseudonymization=False))
        assert gdpr_enforce(pseudo)["strategy"] == "hash"
        assert gdpr_enforce(plain)["strategy"] == "mask"

    def test_overrides_win(self):
        kwargs = gdpr_enforce(_policy(), strategy="remove", block_on=("email",))
        assert kwargs["strategy"] == "remove"
        assert kwargs["block_on"] == ("email",)

    async def test_wires_into_real_compliance_guard(self):
        class FakeModel:
            async def generate(self, prompt, **kwargs):
                return f"echo: {prompt}"

        guard = ComplianceGuard(FakeModel(), **gdpr_enforce(_policy()))
        # Blocked: SSN is a special-category identifier and no purpose allows it.
        with pytest.raises(ComplianceViolationError):
            await guard.generate("My SSN is 123-45-6789")
        # Redacted, not blocked: plain personal identifier (email).
        out = await guard.generate("Contact me at jane.doe@example.com")
        assert "jane.doe@example.com" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
