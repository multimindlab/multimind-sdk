"""
Tests for cost optimization example.
"""

import pytest
pytest.skip("Skipping example test not structured as importable module.", allow_module_level=True)
import asyncio
from pathlib import Path
import sys
import os

# Add examples directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from examples.model_management.advanced.cost_optimization import (
    CostOptimizedWrapper,
    main as cost_optimization_main
)

@pytest.mark.asyncio
async def test_cost_optimization():
    """Test cost optimization example."""
    try:
        await cost_optimization_main()
        assert True
    except Exception as e:
        pytest.fail(f"Cost optimization example failed: {str(e)}")

@pytest.mark.asyncio
async def test_cost_optimized_wrapper():
    """Test CostOptimizedWrapper class."""
    from multimind.models.factory import ModelFactory
    
    # Initialize wrapper
    factory = ModelFactory()
    wrapper = CostOptimizedWrapper(
        model_factory=factory,
        primary_model="gpt-3.5-turbo",
        fallback_models=["gpt-4", "claude"],
        budget=0.1
    )
    
    # Test simple prompt
    response = await wrapper.generate("What is the weather?")
    assert response is not None
    assert len(response) > 0
    
    # Test cost tracking
    assert wrapper.cost_tracker.get_total_cost() > 0
    assert wrapper.cost_tracker.get_model_usage() is not None

def test_environment_variables():
    """Test required environment variables."""
    required_vars = ["OPENAI_API_KEY", "CLAUDE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    assert not missing_vars, f"Missing environment variables: {', '.join(missing_vars)}" 