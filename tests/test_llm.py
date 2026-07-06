import pytest
from multimind.llm.non_transformer_llm import QLoRALLM, CompacterLLM


class DummyModel:
    model_name = "dummy"

    async def generate(self, prompt, **kwargs):
        return "dummy output"

    model = "dummy_model"


@pytest.mark.asyncio
async def test_qlora_llm_is_scaffold():
    base = DummyModel()
    llm = QLoRALLM(base, base.model_name)
    with pytest.raises(NotImplementedError):
        await llm.generate("prompt")


@pytest.mark.asyncio
async def test_compacter_llm_is_scaffold():
    base = DummyModel()
    llm = CompacterLLM(base, base.model_name)
    with pytest.raises(NotImplementedError):
        await llm.generate("prompt")
