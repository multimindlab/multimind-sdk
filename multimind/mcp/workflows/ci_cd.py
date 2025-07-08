"""
CI/CD Workflow

This workflow automates the CI/CD pipeline process using AI-powered analysis and multi-platform notifications.
"""

from typing import Any, Dict, List
from ...api.mcp.base import MCPWorkflowAPI
from ...api.mcp.registry import WorkflowRegistry

@WorkflowRegistry.register
class CICDWorkflow(MCPWorkflowAPI):
    """CI/CD workflow implementation."""
    
    def __init__(
        self,
        models: Dict[str, Any],
        integrations: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize the CI/CD workflow."""
        super().__init__(
            name="CI/CD Pipeline",
            description="Automated CI/CD pipeline with AI analysis and multi-platform notifications",
            models=models,
            integrations=integrations,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def _build_workflow_spec(self) -> Dict[str, Any]:
        """Build the workflow specification."""
        return {
            "workflow": {
                "name": "CI/CD Pipeline",
                "parallel": True,
                "steps": [
                    {
                        "name": "analyze_changes",
                        "model": "gpt4",
                        "inputs": {
                            "code_changes": "{{context.code_changes}}",
                            "pr_description": "{{context.pr_description}}"
                        },
                        "prompt": """
                        Analyze the following code changes and PR description:
                        
                        Code Changes:
                        {{inputs.code_changes}}
                        
                        PR Description:
                        {{inputs.pr_description}}
                        
                        Provide a detailed analysis including:
                        1. Impact assessment
                        2. Test coverage analysis
                        3. Deployment considerations
                        4. Potential risks
                        5. Security implications
                        """
                    },
                    {
                        "name": "run_tests",
                        "integration": "github",
                        "operation": "run_tests",
                        "inputs": {
                            "pr_number": "{{context.pr_number}}"
                        }
                    },
                    {
                        "name": "generate_deployment_plan",
                        "model": "claude",
                        "inputs": {
                            "analysis": "{{steps.analyze_changes.output}}",
                            "test_results": "{{steps.run_tests.output}}"
                        },
                        "prompt": """
                        Based on the following analysis and test results, generate a deployment plan:
                        
                        Analysis:
                        {{inputs.analysis}}
                        
                        Test Results:
                        {{inputs.test_results}}
                        
                        The plan should include:
                        1. Deployment steps
                        2. Rollback procedures
                        3. Monitoring requirements
                        4. Success criteria
                        5. Risk mitigation strategies
                        """
                    },
                    {
                        "name": "deploy",
                        "integration": "github",
                        "operation": "deploy",
                        "inputs": {
                            "deployment_plan": "{{steps.generate_deployment_plan.output}}",
                            "pr_number": "{{context.pr_number}}"
                        }
                    },
                    {
                        "name": "send_slack_notification",
                        "integration": "slack",
                        "operation": "send_message",
                        "inputs": {
                            "channel": "{{context.slack_channel}}",
                            "message": """
                            *CI/CD Pipeline Update*
                            
                            PR: #{{context.pr_number}}
                            Analysis: {{steps.analyze_changes.output}}
                            Test Results: {{steps.run_tests.output}}
                            Deployment Plan: {{steps.generate_deployment_plan.output}}
                            Deployment Status: {{steps.deploy.output}}
                            """
                        }
                    },
                    {
                        "name": "send_discord_notification",
                        "integration": "discord",
                        "operation": "send_message",
                        "inputs": {
                            "channel_id": "{{context.discord_channel}}",
                            "message": """
                            **CI/CD Pipeline Update**
                            
                            PR: #{{context.pr_number}}
                            Analysis: {{steps.analyze_changes.output}}
                            Test Results: {{steps.run_tests.output}}
                            Deployment Plan: {{steps.generate_deployment_plan.output}}
                            Deployment Status: {{steps.deploy.output}}
                            """
                        }
                    }
                ],
                "connections": [
                    {
                        "from": "analyze_changes",
                        "to": ["generate_deployment_plan", "send_slack_notification", "send_discord_notification"]
                    },
                    {
                        "from": "run_tests",
                        "to": ["generate_deployment_plan", "send_slack_notification", "send_discord_notification"]
                    },
                    {
                        "from": "generate_deployment_plan",
                        "to": ["deploy", "send_slack_notification", "send_discord_notification"]
                    },
                    {
                        "from": "deploy",
                        "to": ["send_slack_notification", "send_discord_notification"]
                    }
                ]
            }
        }
    
    def _validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate the workflow context."""
        required_fields = [
            "code_changes",
            "pr_description",
            "pr_number",
            "slack_channel",
            "discord_channel"
        ]
        return all(field in context for field in required_fields)
    
    @classmethod
    def _get_required_integrations(cls) -> List[str]:
        """Get required integrations."""
        return ["github", "slack", "discord"]
    
    @classmethod
    def _get_required_models(cls) -> List[str]:
        """Get required models."""
        return ["gpt4", "claude"] 