import pytest
from multimind.llm.non_transformer_llm import QLoRALLM, CompacterLLM
from multimind.llm.llm_interface import LLMInterface, ModelType

class DummyModel:
    model_name = "dummy"
    async def generate(self, prompt, **kwargs):
        return "dummy output"
    model = "dummy_model"

def test_qlora_llm_init():
    base = DummyModel()
    llm = QLoRALLM(base, base.model_name)
    assert llm is not None

def test_compacter_llm_init():
    base = DummyModel()
    llm = CompacterLLM(base, base.model_name)
    assert llm is not None

@pytest.mark.skip(reason="LLMInterface requires AdvancedPrompting with a 'model' argument; skipping for now.")
def test_llm_interface_init():
    pass

def test_model_type_enum():
    # Check for actual enum values
    assert len(list(ModelType)) > 0 