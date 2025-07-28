#!/usr/bin/env python3
"""
Test script to check if examples can be imported without errors.
"""

import sys
import importlib
import traceback

def test_import(module_name):
    """Test importing a module."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        traceback.print_exc()
        return False

def main():
    """Test all example modules."""
    examples = [
        "examples.cli.basic_agent",
        "examples.cli.chat_with_gpt",
        "examples.cli.usage_tracking",
        "examples.cli.ensemble_cli",
        "examples.cli.task_runner",
        "examples.api.ensemble_api",
        "examples.mcp.examples.code_review_example",
        "examples.mcp.examples.ci_cd_example",
        "examples.mcp.examples.documentation_example",
        "examples.model_management.basic.basic_usage",
        "examples.model_management.basic.api_usage",
        "examples.model_management.multi_model_example",
        "examples.memory.basic_usage",
        "examples.rag.example_rag",
        "examples.vector_store.advanced_vector_store_example",
        "examples.compliance.examples",
        "examples.pipeline.pipeline_example"
    ]
    
    print("Testing example imports...")
    print("=" * 50)
    
    success_count = 0
    total_count = len(examples)
    
    for example in examples:
        if test_import(example):
            success_count += 1
    
    print("=" * 50)
    print(f"Results: {success_count}/{total_count} examples imported successfully")
    
    if success_count == total_count:
        print("All examples can be imported successfully!")
        return 0
    else:
        print("Some examples failed to import.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 