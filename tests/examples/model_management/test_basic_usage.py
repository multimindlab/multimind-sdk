"""
Tests for basic model management examples.
"""

import pytest
pytest.skip("Skipping example test not structured as importable module.", allow_module_level=True)
import asyncio
from pathlib import Path
import sys
import os

# Add examples directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from examples.model_management.basic.basic_usage import main as basic_usage_main
from examples.model_management.basic.cli_usage import main as cli_usage_main
from examples.model_management.basic.api_usage import main as api_usage_main

@pytest.mark.asyncio
async def test_basic_usage():
    """Test basic usage example."""
    try:
        await basic_usage_main()
        assert True
    except Exception as e:
        pytest.fail(f"Basic usage example failed: {str(e)}")

@pytest.mark.asyncio
async def test_cli_usage():
    """Test CLI usage example."""
    try:
        await cli_usage_main()
        assert True
    except Exception as e:
        pytest.fail(f"CLI usage example failed: {str(e)}")

@pytest.mark.asyncio
async def test_api_usage():
    """Test API usage example."""
    try:
        await api_usage_main()
        assert True
    except Exception as e:
        pytest.fail(f"API usage example failed: {str(e)}")

def test_environment_variables():
    """Test required environment variables."""
    required_vars = ["OPENAI_API_KEY", "CLAUDE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    assert not missing_vars, f"Missing environment variables: {', '.join(missing_vars)}" 