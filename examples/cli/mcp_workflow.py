"""
MCP workflow example demonstrating how to use Model Composition Protocol for complex workflows.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from multimind import (
    OpenAIModel, ClaudeModel,
    MCPParser, MCPExecutor
)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Create models
    openai_model = OpenAIModel(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    claude_model = ClaudeModel(
        model="claude-3-sonnet-20240229",
        temperature=0.7
    )
    
    # Create MCP executor
    executor = MCPExecutor()
    
    # Register models
    executor.register_model("gpt-3.5", openai_model)
    executor.register_model("claude-3", claude_model)
    
    # Define MCP workflow
    workflow = {
        "version": "1.0.0",
        "models": [
            {
                "name": "gpt-3.5",
                "type": "openai",
                "config": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                }
            },
            {
                "name": "claude-3",
                "type": "claude",
                "config": {
                    "model": "claude-3-sonnet-20240229",
                    "temperature": 0.7
                }
            }
        ],
        "workflow": {
            "steps": [
                {
                    "id": "initial_analysis",
                    "type": "model",
                    "config": {
                        "model": "gpt-3.5",
                        "prompt_template": "Analyze the following topic: {topic}\nProvide a detailed analysis."
                    }
                },
                {
                    "id": "expert_review",
                    "type": "model",
                    "config": {
                        "model": "claude-3",
                        "prompt_template": "Review and enhance the following analysis:\n{initial_analysis}\nProvide expert insights and additional perspectives."
                    }
                },
                {
                    "id": "synthesis",
                    "type": "transform",
                    "config": {
                        "type": "join",
                        "separator": "\n\n"
                    }
                },
                {
                    "id": "quality_check",
                    "type": "condition",
                    "config": {
                        "type": "contains",
                        "value": "key insights"
                    }
                }
            ],
            "connections": [
                {
                    "from": "initial_analysis",
                    "to": "expert_review"
                },
                {
                    "from": "initial_analysis",
                    "to": "synthesis"
                },
                {
                    "from": "expert_review",
                    "to": "synthesis"
                },
                {
                    "from": "synthesis",
                    "to": "quality_check"
                }
            ]
        }
    }
    
    # Save workflow to file (optional)
    with open("workflow.json", "w") as f:
        json.dump(workflow, f, indent=2)
    
    # Execute workflow
    topic = "The Future of Artificial Intelligence"
    results = await executor.execute(workflow, {"topic": topic})
    
    # Print results
    print("MCP Workflow Results:")
    print("====================")
    
    for step_id, result in results.items():
        print(f"\n{step_id.upper()}:")
        print("-" * len(step_id))
        print(result)
        print()

if __name__ == "__main__":
    asyncio.run(main()) 