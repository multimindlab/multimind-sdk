import pytest
from multimind.orchestration.prompt_chain import PromptChain
from multimind.orchestration.task_runner import TaskRunner
from multimind.router import ModelRouter
from multimind.pipeline.pipeline import Pipeline, PipelineBuilder

import asyncio

@pytest.mark.asyncio
async def test_prompt_chain():
    chain = PromptChain([lambda x: x + "1", lambda x: x + "2"])
    result = await chain.run("input")
    assert isinstance(result, list)

@pytest.mark.asyncio
async def test_task_runner():
    class DummyModel:
        async def generate(self, prompt, **kwargs):
            return "ok"
    runner = TaskRunner(model=DummyModel())
    assert runner is not None

@pytest.mark.asyncio
async def test_model_router():
    router = ModelRouter()
    class DummyModel:
        async def generate(self, prompt, **kwargs):
            return "ok"
    router.register_model("dummy", DummyModel())
    model = await router.get_model("dummy")
    assert model is not None

@pytest.mark.asyncio
async def test_pipeline():
    class DummyRouter:
        pass
    pipeline = Pipeline(DummyRouter())
    assert pipeline is not None 