"""Tests for PrivacyBudget epsilon accounting on top of the Laplace DPMechanism."""

from __future__ import annotations

import pytest

from multimind.compliance.advanced import (
    AdaptivePrivacy,
    BudgetExhaustedError,
    DPMechanism,
    PrivacyBudget,
)


def test_budget_starts_unspent():
    budget = PrivacyBudget(total_epsilon=2.0)

    assert budget.total_epsilon == 2.0
    assert budget.spent_epsilon == 0.0
    assert budget.remaining_epsilon == 2.0


def test_spend_deducts_from_remaining():
    budget = PrivacyBudget(total_epsilon=2.0)

    budget.spend(0.5)

    assert budget.spent_epsilon == 0.5
    assert budget.remaining_epsilon == pytest.approx(1.5)


def test_spend_exceeding_remaining_raises():
    budget = PrivacyBudget(total_epsilon=1.0)
    budget.spend(0.8)

    with pytest.raises(BudgetExhaustedError):
        budget.spend(0.5)

    # Failed spend must not partially deduct.
    assert budget.spent_epsilon == 0.8


def test_spend_exact_remaining_succeeds():
    budget = PrivacyBudget(total_epsilon=1.0)

    budget.spend(1.0)

    assert budget.remaining_epsilon == pytest.approx(0.0)


def test_non_positive_total_epsilon_rejected():
    with pytest.raises(ValueError):
        PrivacyBudget(total_epsilon=0.0)
    with pytest.raises(ValueError):
        PrivacyBudget(total_epsilon=-1.0)


def test_non_positive_spend_rejected():
    budget = PrivacyBudget(total_epsilon=1.0)
    with pytest.raises(ValueError):
        budget.spend(0.0)
    with pytest.raises(ValueError):
        budget.spend(-0.1)


def test_dp_mechanism_without_budget_unlimited():
    dp = DPMechanism(epsilon=0.5, sensitivity=1.0)

    # No budget configured: any number of calls succeed.
    for _ in range(10):
        result = dp.privatize(5.0)
        assert isinstance(result, float)


def test_dp_mechanism_spends_budget_per_call():
    budget = PrivacyBudget(total_epsilon=1.0)
    dp = DPMechanism(epsilon=0.4, sensitivity=1.0, budget=budget)

    dp.privatize(1.0)
    dp.privatize(2.0)

    assert budget.spent_epsilon == pytest.approx(0.8)


def test_dp_mechanism_raises_when_budget_exhausted():
    budget = PrivacyBudget(total_epsilon=1.0)
    dp = DPMechanism(epsilon=0.4, sensitivity=1.0, budget=budget)

    dp.privatize(1.0)
    dp.privatize(2.0)

    with pytest.raises(BudgetExhaustedError):
        dp.privatize(3.0)


def test_nested_structure_spends_epsilon_once_per_call():
    budget = PrivacyBudget(total_epsilon=1.0)
    dp = DPMechanism(epsilon=0.5, sensitivity=1.0, budget=budget)

    result = dp.privatize({"a": 1.0, "b": [2.0, 3.0]})

    assert budget.spent_epsilon == pytest.approx(0.5)
    assert set(result.keys()) == {"a", "b"}
    assert isinstance(result["a"], float)
    assert len(result["b"]) == 2


def test_privatize_boolean_raises_not_implemented():
    dp = DPMechanism(epsilon=0.5)
    with pytest.raises(NotImplementedError):
        dp.privatize(True)


def test_privatize_unsupported_type_raises_not_implemented():
    dp = DPMechanism(epsilon=0.5)
    with pytest.raises(NotImplementedError):
        dp.privatize("a string")


def test_adaptive_privacy_backward_compatible_without_budget():
    ap = AdaptivePrivacy({})

    assert ap.dp_mechanism.budget is None
    result = ap.dp_mechanism.privatize(5.0)
    assert isinstance(result, float)
