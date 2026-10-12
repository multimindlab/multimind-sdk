"""Edge-case tests for the MoE (Mixture of Experts) "Beta" feature area.

Covers:
  * ``MoEConfig`` validation/round-trip boundaries (pure Python, no torch needed).
  * The honest "PyTorch is required" fallback path used by ``multimind.models.moe.*``
    and ``MoEFactory`` when torch is absent (this venv has no torch installed).
  * Real routing/capacity edge cases in ``MoELayer``/``AdvancedMoELayer`` — gated
    with ``pytest.importorskip("torch")`` so they run in CI's torch-enabled job.
"""

import pytest

from multimind.config import MoEConfig


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


# --- MoEConfig: pure-Python validation/round-trip edge cases ----------------


def test_moe_config_k_greater_than_num_experts_rejected():
    config = MoEConfig(input_dim=8, hidden_dim=16, num_experts=2, num_layers=1, k=3)
    with pytest.raises(AssertionError, match="k must be"):
        config.validate()


def test_moe_config_k_equal_num_experts_is_valid_boundary():
    config = MoEConfig(input_dim=8, hidden_dim=16, num_experts=4, num_layers=1, k=4)
    config.validate()  # should not raise


def test_moe_config_capacity_factor_zero_rejected():
    config = MoEConfig(
        input_dim=8, hidden_dim=16, num_experts=2, num_layers=1, capacity_factor=0.0
    )
    with pytest.raises(AssertionError, match="capacity_factor"):
        config.validate()


def test_moe_config_capacity_factor_above_bound_rejected():
    config = MoEConfig(
        input_dim=8, hidden_dim=16, num_experts=2, num_layers=1, capacity_factor=2.5
    )
    with pytest.raises(AssertionError, match="capacity_factor"):
        config.validate()


def test_moe_config_single_expert_round_trip():
    """num_experts=1, k=1 is the degenerate single-expert MoE — should round-trip cleanly."""
    config = MoEConfig(input_dim=4, hidden_dim=8, num_experts=1, num_layers=1, k=1)
    config.validate()
    restored = MoEConfig.from_dict(config.to_dict())
    assert restored.num_experts == 1
    assert restored.k == 1


# --- Honest missing-dependency fallback (torch absent in this venv) --------


@pytest.mark.skipif(_has_torch(), reason="torch installed; no-deps fallback path untestable")
def test_moe_layer_raises_clean_import_error_without_torch():
    from multimind.models.moe.moe_layer import MoELayer

    with pytest.raises(ImportError, match="PyTorch is required"):
        MoELayer(input_dim=4, num_experts=2, expert_dim=8)


@pytest.mark.skipif(_has_torch(), reason="torch installed; no-deps fallback path untestable")
def test_moe_model_raises_clean_import_error_without_torch():
    from multimind.models.moe.moe_model import MoEModel

    with pytest.raises(ImportError, match="PyTorch is required"):
        MoEModel(input_dim=4, hidden_dim=8, num_experts=2, num_layers=1)


@pytest.mark.skipif(_has_torch(), reason="torch installed; no-deps fallback path untestable")
def test_advanced_moe_layer_raises_clean_import_error_without_torch():
    from multimind.models.moe.advanced_moe import AdvancedMoELayer

    with pytest.raises(ImportError, match="PyTorch is required"):
        AdvancedMoELayer(input_dim=4, num_experts=2, expert_dim=8)


@pytest.mark.skipif(_has_torch(), reason="torch installed; no-deps fallback path untestable")
def test_unified_moe_raises_clean_import_error_without_torch():
    from multimind.models.moe.unified_moe import UnifiedMoE

    with pytest.raises(ImportError, match="PyTorch is required"):
        UnifiedMoE()


@pytest.mark.skipif(_has_torch(), reason="torch installed; no-deps fallback path untestable")
def test_moe_factory_create_moe_model_raises_clean_import_error_without_torch():
    """MoEFactory.create_moe_model should surface the honest torch-missing error,
    not a confusing TypeError from mismatched constructor arguments."""
    from multimind.models.moe.moe_factory import MoEFactory

    factory = MoEFactory()
    with pytest.raises(ImportError, match="PyTorch is required"):
        factory.create_moe_model(
            {"input_dim": 4, "hidden_dim": 8, "num_experts": 2, "num_layers": 1}
        )


def test_moe_factory_get_and_remove_unknown_model_ids():
    """get_model/remove_model on unknown ids should be no-ops, not raise — independent of torch."""
    from multimind.models.moe.moe_factory import MoEFactory

    factory = MoEFactory()
    assert factory.get_model("does-not-exist") is None
    assert factory.remove_model("does-not-exist") is False
    assert factory.list_models() == {}


# --- Real torch-backed routing/capacity edge cases (CI GPU/full job) --------


def test_moe_layer_top_k_greater_than_num_experts_raises():
    torch = pytest.importorskip("torch")
    from multimind.models.moe.moe_layer import MoELayer

    layer = MoELayer(input_dim=4, num_experts=2, expert_dim=8, k=3)
    x = torch.randn(1, 2, 4)
    with pytest.raises(RuntimeError):
        layer(x)


def test_moe_layer_single_expert_routes_all_tokens():
    torch = pytest.importorskip("torch")
    from multimind.models.moe.moe_layer import MoELayer

    layer = MoELayer(input_dim=4, num_experts=1, expert_dim=8, k=1)
    x = torch.randn(2, 3, 4)
    output, aux_loss = layer(x, return_aux_loss=True)
    assert output.shape == x.shape
    assert aux_loss is not None


def test_moe_layer_use_aux_loss_false_returns_no_aux_loss():
    torch = pytest.importorskip("torch")
    from multimind.models.moe.moe_layer import MoELayer

    layer = MoELayer(input_dim=4, num_experts=2, expert_dim=8, k=2, use_aux_loss=False)
    x = torch.randn(1, 2, 4)
    output, aux_loss = layer(x, return_aux_loss=True)
    assert output.shape == x.shape
    assert aux_loss is None


def test_advanced_moe_layer_zero_capacity_falls_back_to_basic_routing():
    """When dynamic capacity collapses to zero for every expert, AdvancedMoELayer
    must fall back to MoELayer's basic routing instead of producing an empty/broken output."""
    torch = pytest.importorskip("torch")
    from multimind.models.moe.advanced_moe import AdvancedMoELayer

    layer = AdvancedMoELayer(
        input_dim=4, num_experts=2, expert_dim=8, k=2, min_expert_capacity=0
    )
    x = torch.randn(1, 3, 4)
    output, _ = layer(x, return_aux_loss=False)
    assert output.shape == x.shape
