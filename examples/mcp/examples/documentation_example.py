"""
Example usage of the Documentation workflow.
"""

import asyncio
from multimind.mcp.workflows import DocumentationWorkflow
from multimind.models import OpenAIModel, ClaudeModel
from multimind.integrations import GitHubIntegrationHandler, SlackIntegrationHandler, DiscordIntegrationHandler

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
    workflow = DocumentationWorkflow(
        models=models,
        integrations=integrations
    )

    # Example context
    context = {
        "code": """
        class DataProcessor:
            def __init__(self, config):
                self.config = config
                self.cache = {}
            
            def process(self, data):
                if not isinstance(data, dict):
                    raise ValueError("Data must be a dictionary")
                return {k: v for k, v in data.items() if v is not None}
            
            def get_cached(self, key):
                return self.cache.get(key)
            
            def set_cached(self, key, value):
                self.cache[key] = value
        """,
        "requirements": """
        - Python 3.8+
        - Support for dictionary processing
        - Caching mechanism
        - Error handling
        """,
        "github_repo": "myorg/myrepo",
        "github_branch": "main",
        "slack_channel": "#documentation",
        "discord_channel": "documentation"
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