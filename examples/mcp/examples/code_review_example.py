"""
Example usage of the Code Review workflow.

This script demonstrates how to use the Code Review workflow API to automate code reviews.
"""

import asyncio
from multimind import CodeReviewWorkflow, OpenAIModel, ClaudeModel, GitHubIntegrationHandler, SlackIntegrationHandler, DiscordIntegrationHandler

async def main():
    # Initialize models
    models = {
        "gpt4": OpenAIModel(model="gpt-4"),
        "claude": ClaudeModel(model="claude-3-opus")
    }

    # Initialize integrations
    integrations = {
        "github": GitHubIntegrationHandler(token="your-github-token"),
        "slack": SlackIntegrationHandler(token="your-slack-token"),
        "discord": DiscordIntegrationHandler(token="your-discord-token")
    }

    # Create workflow instance
    workflow = CodeReviewWorkflow(
        models=models,
        integrations=integrations
    )

    # Example context
    context = {
        "code_changes": """
        def calculate_total(items):
            total = 0
            for item in items:
                total += item.price
            return total
        """,
        "pr_description": "Add total calculation function",
        "pr_number": "123",
        "slack_channel": "#code-reviews",
        "discord_channel": "code-reviews"
    }

    # Define callbacks
    callbacks = {
        "on_success": lambda result: print("Workflow completed successfully:", result),
        "on_error": lambda error, state: print("Workflow failed:", error)
    }

    # Execute workflow
    result = await workflow.execute(context, callbacks)
    print("Workflow result:", result)

if __name__ == "__main__":
    asyncio.run(main()) 