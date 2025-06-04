# MultiMind Control Plane (MCP) Examples

This directory contains example usage of the MultiMind Control Plane (MCP) workflows and APIs.

## Directory Structure

```
examples/mcp/
├── examples/                    # Example usage scripts
│   ├── code_review_example.py   # Code review workflow example
│   ├── ci_cd_example.py         # CI/CD workflow example
│   └── documentation_example.py # Documentation workflow example
├── USER_JOURNEY.md             # User journey documentation
└── README.md                   # This file
```

## Available Examples

1. **Code Review Workflow** (`examples/code_review_example.py`)
   - Demonstrates automated code review using AI models
   - Integrates with GitHub, Slack, and Discord
   - Shows how to handle PR reviews and notifications

2. **CI/CD Workflow** (`examples/ci_cd_example.py`)
   - Shows CI/CD pipeline automation
   - Integrates with GitHub for PR management
   - Demonstrates automated testing and deployment

3. **Documentation Workflow** (`examples/documentation_example.py`)
   - Illustrates automated documentation generation
   - Shows integration with GitHub for documentation updates
   - Demonstrates multi-platform notifications

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GITHUB_TOKEN="your-github-token"
   export SLACK_TOKEN="your-slack-token"
   export DISCORD_TOKEN="your-discord-token"
   ```

3. Run an example:
   ```bash
   python examples/code_review_example.py
   ```

## User Journey

For a comprehensive guide on using the MCP system, including:
- Workflow creation and customization
- Integration setup and configuration
- Best practices and patterns
- Troubleshooting and debugging

Please refer to [USER_JOURNEY.md](USER_JOURNEY.md).

## Contributing

To add new examples:
1. Create a new Python file in the `examples` directory
2. Follow the existing example patterns
3. Include clear documentation and comments
4. Update this README with the new example details

## Support

For issues, questions, or suggestions:
1. Check the [USER_JOURNEY.md](USER_JOURNEY.md) documentation
2. Open an issue in the repository
3. Contact the development team 