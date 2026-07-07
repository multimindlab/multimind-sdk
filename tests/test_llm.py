import sys
from unittest.mock import MagicMock

import pytest

from multimind.llm.non_transformer_llm import CompacterLLM, QLoRALLM


class DummyModel:
    model_name = "dummy"

    async def generate(self, prompt, **kwargs):
        return "dummy output"

    model = "dummy_model"


def _has_qlora_deps():
    try:
        import bitsandbytes  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(_has_qlora_deps(), reason="QLoRA deps installed; no-deps path untestable")
@pytest.mark.asyncio
async def test_qlora_llm_raises_without_deps():
    base = DummyModel()
    llm = QLoRALLM(base, base.model_name)
    with pytest.raises(NotImplementedError, match="finetune-gpu"):
        await llm.generate("prompt")


@pytest.mark.asyncio
async def test_compacter_llm_is_scaffold():
    base = DummyModel()
    llm = CompacterLLM(base, base.model_name)
    with pytest.raises(NotImplementedError):
        await llm.generate("prompt")


def _install_fake_deps(monkeypatch):
    fake_torch = MagicMock()
    fake_torch.bfloat16 = "bfloat16"
    fake_transformers = MagicMock()
    fake_peft = MagicMock()
    fake_bnb = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)
    return fake_torch, fake_transformers, fake_peft


def test_qlora_apply_qlora_plumbing(monkeypatch):
    fake_torch, fake_transformers, fake_peft = _install_fake_deps(monkeypatch)
    base = DummyModel()
    llm = QLoRALLM(
        base, base.model_name, lora_r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj"]
    )

    model = llm.apply_qlora()

    bnb_kwargs = fake_transformers.BitsAndBytesConfig.call_args.kwargs
    assert bnb_kwargs["load_in_4bit"] is True
    assert bnb_kwargs["bnb_4bit_quant_type"] == "nf4"
    assert bnb_kwargs["bnb_4bit_use_double_quant"] is True
    assert bnb_kwargs["bnb_4bit_compute_dtype"] == fake_torch.bfloat16

    load_call = fake_transformers.AutoModelForCausalLM.from_pretrained.call_args
    assert load_call.args == ("dummy",)
    assert (
        load_call.kwargs["quantization_config"] is fake_transformers.BitsAndBytesConfig.return_value
    )

    fake_peft.prepare_model_for_kbit_training.assert_called_once_with(
        fake_transformers.AutoModelForCausalLM.from_pretrained.return_value
    )

    lora_kwargs = fake_peft.LoraConfig.call_args.kwargs
    assert lora_kwargs["r"] == 8
    assert lora_kwargs["lora_alpha"] == 16
    assert lora_kwargs["lora_dropout"] == 0.1
    assert lora_kwargs["target_modules"] == ["q_proj"]
    assert lora_kwargs["task_type"] == "CAUSAL_LM"

    fake_peft.get_peft_model.assert_called_once_with(
        fake_peft.prepare_model_for_kbit_training.return_value,
        fake_peft.LoraConfig.return_value,
    )
    assert model is fake_peft.get_peft_model.return_value
    assert llm.peft_model is model


@pytest.mark.asyncio
async def test_qlora_generate_requires_tokenizer(monkeypatch):
    _install_fake_deps(monkeypatch)
    base = DummyModel()  # no tokenizer attribute
    llm = QLoRALLM(base, base.model_name)
    with pytest.raises(NotImplementedError, match="tokenizer"):
        await llm.generate("prompt")


@pytest.mark.asyncio
async def test_qlora_generate_through_peft_model(monkeypatch):
    fake_torch, _, fake_peft = _install_fake_deps(monkeypatch)

    tokenizer = MagicMock()
    inputs = MagicMock()
    tokenizer.return_value.to.return_value = inputs
    inputs.keys.return_value = []
    tokenizer.decode.return_value = "decoded output"

    base = DummyModel()
    base.tokenizer = tokenizer
    llm = QLoRALLM(base, base.model_name)

    result = await llm.generate("prompt", max_tokens=5)

    assert llm.peft_model is fake_peft.get_peft_model.return_value
    gen_kwargs = llm.peft_model.generate.call_args.kwargs
    assert gen_kwargs["max_new_tokens"] == 5
    assert result == "decoded output"
