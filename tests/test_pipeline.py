import pytest
from multimind.pipeline.pipeline import Pipeline, PipelineBuilder

class DummyRouter:
    pass

@pytest.mark.asyncio
async def test_pipeline():
    pipeline = Pipeline(DummyRouter())
    assert pipeline is not None

@pytest.mark.asyncio
async def test_pipeline_builder():
    builder = PipelineBuilder(DummyRouter())
    assert builder is not None

@pytest.mark.asyncio
async def test_pipeline_init():
    pipeline = Pipeline(DummyRouter())
    assert pipeline is not None

@pytest.mark.asyncio
async def test_pipeline_builder_init():
    builder = PipelineBuilder(DummyRouter())
    assert builder is not None

@pytest.mark.asyncio
async def test_pipeline_builder_build():
    builder = PipelineBuilder(DummyRouter())
    assert builder is not None 