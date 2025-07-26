import pytest
from multimind.llm.non_transformer_llm import QLoRALLM, CompacterLLM

class DummyModel:
    model_name = "dummy"
    async def generate(self, prompt, **kwargs):
        return "dummy output"
    model = "dummy_model"

@pytest.mark.asyncio
async def test_qlora_llm_warns_and_returns():
    base = DummyModel()
    llm = QLoRALLM(base, base.model_name)
    result = await llm.generate("prompt")
    assert result == "dummy output"

@pytest.mark.asyncio
async def test_compacter_llm_warns_and_returns():
    base = DummyModel()
    llm = CompacterLLM(base, base.model_name)
    result = await llm.generate("prompt")
    assert result == "dummy output" 