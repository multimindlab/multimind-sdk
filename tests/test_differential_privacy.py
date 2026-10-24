"""Tests for DPMechanism, PrivacyBudget, BudgetExhaustedError and AdaptivePrivacy."""

import random

import pytest

from multimind.compliance.advanced import (
    AdaptivePrivacy,
    BudgetExhaustedError,
    DPMechanism,
    PrivacyBudget,
)

# --- PrivacyBudget -------------------------------------------------------------


def test_budget_requires_positive_total():
    with pytest.raises(ValueError):
        PrivacyBudget(0)
    with pytest.raises(ValueError):
        PrivacyBudget(-1.0)


def test_budget_spend_decrements_remaining():
    budget = PrivacyBudget(1.0)
    assert budget.remaining_epsilon == 1.0
    budget.spend(0.4)
    assert budget.spent_epsilon == pytest.approx(0.4)
    assert budget.remaining_epsilon == pytest.approx(0.6)


def test_budget_spend_rejects_non_positive_epsilon():
    budget = PrivacyBudget(1.0)
    with pytest.raises(ValueError):
        budget.spend(0)
    with pytest.raises(ValueError):
        budget.spend(-0.1)


def test_budget_exhaustion_raises_and_does_not_overspend():
    budget = PrivacyBudget(1.0)
    budget.spend(0.7)
    with pytest.raises(BudgetExhaustedError, match="exhausted"):
        budget.spend(0.5)
    assert budget.spent_epsilon == pytest.approx(0.7)


def test_budget_allows_spending_exactly_to_zero():
    budget = PrivacyBudget(1.0)
    budget.spend(0.5)
    budget.spend(0.5)
    assert budget.remaining_epsilon == pytest.approx(0.0)


# --- DPMechanism ---------------------------------------------------------------


def test_privatize_adds_noise_to_scalars():
    random.seed(1234)
    dp = DPMechanism(epsilon=0.5, sensitivity=1.0)
    outs = [dp.privatize(10) for _ in range(20)]

    assert all(isinstance(o, float) for o in outs)
    assert any(o != 10.0 for o in outs)
    assert len(set(outs)) > 1  # noise is random, not a constant offset


def test_privatize_noise_scale_tracks_epsilon():
    random.seed(0)
    tight = [abs(DPMechanism(epsilon=100.0).privatize(0.0)) for _ in range(500)]
    random.seed(0)
    loose = [abs(DPMechanism(epsilon=0.1).privatize(0.0)) for _ in range(500)]
    # Laplace scale = sensitivity/epsilon -> larger epsilon means smaller noise
    assert sum(tight) / len(tight) < sum(loose) / len(loose)


def test_privatize_recurses_into_dicts_and_lists():
    random.seed(7)
    dp = DPMechanism(epsilon=1.0)
    out = dp.privatize({"a": 1, "b": [2.0, 3]})

    assert set(out) == {"a", "b"}
    assert isinstance(out["a"], float)
    assert len(out["b"]) == 2 and all(isinstance(v, float) for v in out["b"])


@pytest.mark.parametrize("bad", [True, "text", None, object(), (1, 2)])
def test_privatize_rejects_non_numeric_types(bad):
    with pytest.raises(NotImplementedError):
        DPMechanism(epsilon=1.0).privatize(bad)


def test_privatize_rejects_bool_nested_in_container():
    with pytest.raises(NotImplementedError):
        DPMechanism(epsilon=1.0).privatize({"flag": True})


def test_privatize_spends_budget_once_per_call():
    budget = PrivacyBudget(total_epsilon=1.0)
    dp = DPMechanism(epsilon=0.3, budget=budget)

    dp.privatize([1, 2, 3])  # nested data still costs one epsilon
    assert budget.spent_epsilon == pytest.approx(0.3)
    dp.privatize(5)
    assert budget.spent_epsilon == pytest.approx(0.6)
    assert budget.remaining_epsilon == pytest.approx(0.4)


def test_privatize_raises_when_budget_exhausted():
    budget = PrivacyBudget(total_epsilon=0.5)
    dp = DPMechanism(epsilon=0.3, budget=budget)

    dp.privatize(1.0)
    with pytest.raises(BudgetExhaustedError):
        dp.privatize(1.0)
    assert budget.spent_epsilon == pytest.approx(0.3)


def test_privatize_without_budget_is_unlimited():
    dp = DPMechanism(epsilon=0.01)
    for _ in range(50):
        dp.privatize(1.0)
    assert dp.budget is None


# --- AdaptivePrivacy -----------------------------------------------------------


def test_adaptive_privacy_initial_epsilon_from_config():
    ap = AdaptivePrivacy({"initial_epsilon": 2.5, "dp_sensitivity": 0.5})
    assert ap.dp_mechanism.epsilon == 2.5
    assert ap.dp_mechanism.sensitivity == 0.5
    assert ap.adaptation_strategy.initial_epsilon == 2.5


def test_adaptive_privacy_uses_default_initial_epsilon():
    ap = AdaptivePrivacy({})
    assert ap.dp_mechanism.epsilon == 1.0
    assert ap.adaptation_strategy.min_epsilon == 0.1
    assert ap.adaptation_strategy.max_epsilon == 10.0


async def test_low_compliance_raises_epsilon_by_adaptation_rate():
    ap = AdaptivePrivacy({"min_epsilon": 0.1, "max_epsilon": 10.0, "adaptation_rate": 0.25})
    start = ap.homomorphic_encryption.epsilon

    await ap.adapt_privacy({"compliance_score": 0.2, "loss": 0.9})

    assert ap.homomorphic_encryption.epsilon == pytest.approx(start + 0.25)
    assert ap.dp_mechanism.epsilon == pytest.approx(start + 0.25)
    assert len(ap.feedback_history) == 1
    entry = ap.feedback_history[0]
    assert entry["compliance_score"] == 0.2
    assert entry["current_epsilon"] == start
    assert "timestamp" in entry


async def test_high_compliance_lowers_epsilon_but_not_below_min():
    # HomomorphicEncryption starts at epsilon=0.1; step down by 0.02 towards min 0.05
    ap = AdaptivePrivacy({"min_epsilon": 0.05, "adaptation_rate": 0.02})

    await ap.adapt_privacy({"compliance_score": 0.95})
    assert ap.dp_mechanism.epsilon == pytest.approx(0.08)
    await ap.adapt_privacy({"compliance_score": 0.95})
    assert ap.dp_mechanism.epsilon == pytest.approx(0.06)
    await ap.adapt_privacy({"compliance_score": 0.95})
    assert ap.dp_mechanism.epsilon == pytest.approx(0.05)  # clamped to min_epsilon
    await ap.adapt_privacy({"compliance_score": 0.99})
    assert ap.dp_mechanism.epsilon == pytest.approx(0.05)
    assert ap.homomorphic_encryption.epsilon == pytest.approx(0.05)


async def test_mid_compliance_keeps_epsilon_unchanged():
    ap = AdaptivePrivacy({"adaptation_rate": 0.3})
    await ap.adapt_privacy({"compliance_score": 0.2})
    before = ap.dp_mechanism.epsilon
    await ap.adapt_privacy({"compliance_score": 0.8})
    # history average (0.2 + 0.8)/2 = 0.5 < 0.7 -> still raises
    assert ap.dp_mechanism.epsilon == pytest.approx(before + 0.3)

    ap2 = AdaptivePrivacy({"adaptation_rate": 0.3})
    await ap2.adapt_privacy({"compliance_score": 0.8})
    assert ap2.dp_mechanism.epsilon == pytest.approx(0.1)  # unchanged from HE start


async def test_epsilon_is_capped_at_max_epsilon():
    ap = AdaptivePrivacy({"min_epsilon": 0.1, "max_epsilon": 0.3, "adaptation_rate": 1.0})
    await ap.adapt_privacy({"compliance_score": 0.0})
    assert ap.dp_mechanism.epsilon == pytest.approx(0.3)
    await ap.adapt_privacy({"compliance_score": 0.0})
    assert ap.dp_mechanism.epsilon == pytest.approx(0.3)


async def test_feedback_updates_privacy_metrics():
    ap = AdaptivePrivacy({})
    await ap.adapt_privacy({"compliance_score": 1.0, "loss": 2.0})

    # exponential moving averages with alpha=0.1
    assert ap.privacy_metrics["avg_loss"] == pytest.approx(0.2)
    assert ap.privacy_metrics["avg_compliance"] == pytest.approx(0.5 * 0.9 + 1.0 * 0.1)


def test_validate_epsilon_bounds():
    ap = AdaptivePrivacy({"min_epsilon": 0.2, "max_epsilon": 2.0})
    assert ap._validate_epsilon(0.2)
    assert ap._validate_epsilon(2.0)
    assert not ap._validate_epsilon(0.1)
    assert not ap._validate_epsilon(2.1)
    assert not ap._validate_epsilon(-1.0)
