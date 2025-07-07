import pytest
from multimind.config import MoEConfig, MultiModalConfig
from multimind.config.multi_modal_config import ModelConfig

def test_moe_config_init():
    config = MoEConfig(input_dim=16, hidden_dim=32, num_experts=2, num_layers=1)
    assert config.input_dim == 16
    assert config.hidden_dim == 32
    assert config.num_experts == 2
    assert config.num_layers == 1

def test_multi_modal_config_init():
    model = ModelConfig(name="test", type="llm", modality="text")
    config = MultiModalConfig(models={"test": model})
    assert "test" in config.models 