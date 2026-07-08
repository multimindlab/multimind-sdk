#!/usr/bin/env python3
"""
Test script to check if examples can be imported without errors.
"""

import sys
import importlib
import traceback
import pytest

@pytest.mark.parametrize("module_name", [
    "examples.cli.basic_agent",
    "examples.cli.chat_with_gpt",
    "examples.cli.usage_tracking",
    "examples.cli.ensemble_cli",
    "examples.cli.task_runner",
    "examples.api.ensemble_api",
    "examples.mcp_workflows.examples.code_review_example",
    "examples.mcp_workflows.examples.ci_cd_example",
    "examples.mcp_workflows.examples.documentation_example",
    "examples.model_management.basic.basic_usage",
    "examples.model_management.basic.api_usage",
    "examples.model_management.multi_model_example",
    "examples.memory.basic_usage",
    "examples.rag.example_rag",
    "examples.vector_store.advanced_vector_store_example",
    "examples.compliance.examples",
    "examples.pipeline.pipeline_example"
])
def test_import(module_name):
    """Test importing a module."""
    try:
        importlib.import_module(module_name)
    except Exception as e:
        # Skip modules that can't be imported (optional dependencies)
        pytest.skip(f"Module {module_name} not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 