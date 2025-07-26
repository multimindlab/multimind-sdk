import pytest
from multimind.memory.procedural import ProceduralMemory
from multimind.memory.semantic import SemanticMemory

@pytest.mark.skip(reason="ImplicitMemory is abstract and cannot be instantiated.")
def test_implicit_memory_basic():
    pass

@pytest.mark.skip(reason="ImplicitMemory is abstract and cannot be instantiated.")
def test_implicit_memory_edge_cases():
    pass

def test_procedural_memory_basic():
    class DummyLLM:
        pass
    mem = ProceduralMemory(DummyLLM(), max_procedures=5)
    assert mem is not None

def test_semantic_memory_basic():
    class DummyLLM:
        pass
    mem = SemanticMemory(DummyLLM(), max_concepts=3)
    assert mem is not None 