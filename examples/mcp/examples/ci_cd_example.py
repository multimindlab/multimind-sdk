"""
Example usage of the CI/CD workflow.

This script demonstrates how to use the CI/CD workflow API to automate continuous integration and deployment.
"""

import asyncio
from multimind import CICDWorkflow, OpenAIModel, ClaudeModel, GitHubIntegrationHandler, SlackIntegrationHandler

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
    workflow = CICDWorkflow(
        models=models,
        integrations=integrations
    )

    # Example context
    context = {
        "code_changes": """
        diff --git a/src/main.py b/src/main.py
        index abc123..def456 100644
        --- a/src/main.py
        +++ b/src/main.py
        @@ -1,5 +1,7 @@
         def process_data(data):
         -    return data
         +    # Add input validation
         +    if not isinstance(data, dict):
         +        raise ValueError("Data must be a dictionary")
         +    return {k: v for k, v in data.items() if v is not None}
        """,
        "pr_description": "Add input validation and filtering to process_data function",
        "pr_number": "123",
        "slack_channel": "#ci-cd",
        "discord_channel": "ci-cd"
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