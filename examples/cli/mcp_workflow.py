"""
MCP workflow example demonstrating how to use Model Composition Protocol for complex workflows.
"""

import asyncio
import os

from dotenv import load_dotenv

from multimind import ClaudeModel, MCPExecutor, OpenAIModel


async def main():
    # Load environment variables
    load_dotenv()

    # Check which API keys are available
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")

    has_openai = openai_key is not None and openai_key.strip() != ""
    has_claude = anthropic_key is not None and anthropic_key.strip() != ""

    if not has_openai and not has_claude:
        print("Error: No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY/CLAUDE_API_KEY")
        return

    # Create MCP executor
    executor = MCPExecutor()

    # Create and register available models
    available_models = []
    model_registry = {}

    if has_openai:
        try:
            openai_model = OpenAIModel(
                model_name="gpt-3.5-turbo",
                temperature=0.7
            )
            executor.register_model("gpt-3.5", openai_model)
            model_registry["gpt-3.5"] = {
                "name": "gpt-3.5",
                "type": "openai",
                "config": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                }
            }
            available_models.append("gpt-3.5")
            print("✓ OpenAI model registered")
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI model: {e}")
            has_openai = False

    if has_claude:
        try:
            claude_model = ClaudeModel(
                model_name="claude-3-sonnet-20240229",
                temperature=0.7
            )
            executor.register_model("claude-3", claude_model)
            model_registry["claude-3"] = {
                "name": "claude-3",
                "type": "claude",
                "config": {
                    "model": "claude-3-sonnet-20240229",
                    "temperature": 0.7
                }
            }
            available_models.append("claude-3")
            print("✓ Claude model registered")
        except Exception as e:
            print(f"Warning: Failed to initialize Claude model: {e}")
            has_claude = False

    # Determine which models to use for each step
    if has_openai and has_claude:
        # Both models available - use both
        initial_model = "gpt-3.5"
        review_model = "claude-3"
        print("\nUsing both models: OpenAI for initial analysis, Claude for expert review")
    elif has_openai:
        # Only OpenAI available - use it for both steps
        initial_model = "gpt-3.5"
        review_model = "gpt-3.5"
        print("\nUsing OpenAI for both steps (Claude API key not found)")
    else:
        # Only Claude available - use it for both steps
        initial_model = "claude-3"
        review_model = "claude-3"
        print("\nUsing Claude for both steps (OpenAI API key not found)")

    # Ask user for topic (or use default if empty)
    print("\n" + "="*50)
    print("MCP Workflow - Topic Analysis")
    print("="*50)
    user_topic = input("\nEnter a topic to analyze (or press Enter for default): ").strip()

    if not user_topic:
        topic = "The Future of Artificial Intelligence"
        print(f"\nUsing default topic: {topic}")
    else:
        topic = user_topic
        print(f"\nAnalyzing topic: {topic}")

    # Calculate quality check word from topic
    # Extract meaningful words from topic for quality check (skip common words)
    topic_words = topic.lower().split()
    articles = ["the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but"]
    significant_words = [w for w in topic_words if w not in articles]

    # Use the last significant word (usually the main subject) or first if only one
    # This works better when there are typos in the topic
    if len(significant_words) > 1:
        quality_check_word = significant_words[-1]  # Usually the main noun
    elif significant_words:
        quality_check_word = significant_words[0]
    else:
        quality_check_word = "analysis"

    # Build workflow dynamically based on available models
    workflow_models = [model_registry[model] for model in available_models]

    workflow = {
        "version": "1.0.0",
        "models": workflow_models,
        "workflow": {
            "steps": [
                {
                    "id": "initial_analysis",
                    "type": "model",
                    "config": {
                        "model": initial_model,
                        "prompt_template": "Analyze the following topic: {topic}\nProvide a detailed analysis."
                    }
                },
                {
                    "id": "expert_review",
                    "type": "model",
                    "config": {
                        "model": review_model,
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
                        "value": quality_check_word  # Dynamically set based on topic
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
