"""
Jira integration handler for MCP workflows.
"""

from typing import Dict, Any, Optional, List
import aiohttp
import logging
from datetime import datetime
import base64
from .base import IntegrationHandler, AsyncContextManager

logger = logging.getLogger(__name__)

class JiraIntegrationHandler(IntegrationHandler, AsyncContextManager):
    """Handler for Jira integration operations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Jira integration handler."""
        super().__init__(config)
        self.validate_config(["domain", "email", "api_token"])
        
        self.domain = config["domain"]
        self.email = config["email"]
        self.api_token = config["api_token"]
        self.api_base = f"https://{self.domain}/rest/api/3"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Create basic auth header
        auth_str = f"{self.email}:{self.api_token}"
        self.auth_header = f"Basic {base64.b64encode(auth_str.encode()).decode()}"

    async def __aenter__(self):
        """Set up aiohttp session."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": self.auth_header,
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up aiohttp session."""
        if self.session:
            await self.session.close()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Jira integration operation."""
        try:
            operation = inputs.get("operation", "create_issue")
            
            if operation == "create_issue":
                result = await self.create_issue(inputs)
            elif operation == "update_issue":
                result = await self.update_issue(inputs)
            elif operation == "get_issue":
                result = await self.get_issue(inputs)
            elif operation == "search_issues":
                result = await self.search_issues(inputs)
            else:
                raise ValueError(f"Unsupported Jira operation: {operation}")
            
            self._update_metadata(success=True)
            return result
            
        except Exception as e:
            self._update_metadata(success=False)
            raise

    async def create_issue(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Jira issue."""
        payload = {
            "fields": {
                "project": {"key": inputs["project_key"]},
                "summary": inputs["summary"],
                "description": inputs.get("description", ""),
                "issuetype": {"name": inputs.get("issue_type", "Task")}
            }
        }
        
        # Add optional fields
        if "priority" in inputs:
            payload["fields"]["priority"] = {"name": inputs["priority"]}
        if "assignee" in inputs:
            payload["fields"]["assignee"] = {"id": inputs["assignee"]}
        if "labels" in inputs:
            payload["fields"]["labels"] = inputs["labels"]

        async with self.session.post(
            f"{self.api_base}/issue",
            json=payload
        ) as response:
            result = await response.json()
            
            if response.status != 201:
                raise Exception(f"Failed to create Jira issue: {result.get('message', 'Unknown error')}")
            
            return {
                "issue_key": result["key"],
                "issue_id": result["id"],
                "self": result["self"]
            }

    async def update_issue(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing Jira issue."""
        issue_key = inputs["issue_key"]
        payload = {
            "fields": {}
        }
        
        # Update only provided fields
        if "summary" in inputs:
            payload["fields"]["summary"] = inputs["summary"]
        if "description" in inputs:
            payload["fields"]["description"] = inputs["description"]
        if "priority" in inputs:
            payload["fields"]["priority"] = {"name": inputs["priority"]}
        if "assignee" in inputs:
            payload["fields"]["assignee"] = {"id": inputs["assignee"]}
        if "labels" in inputs:
            payload["fields"]["labels"] = inputs["labels"]

        async with self.session.put(
            f"{self.api_base}/issue/{issue_key}",
            json=payload
        ) as response:
            if response.status != 204:
                result = await response.json()
                raise Exception(f"Failed to update Jira issue: {result.get('message', 'Unknown error')}")
            
            return {
                "issue_key": issue_key,
                "status": "updated",
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_issue(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get details of a Jira issue."""
        issue_key = inputs["issue_key"]
        
        async with self.session.get(
            f"{self.api_base}/issue/{issue_key}"
        ) as response:
            result = await response.json()
            
            if response.status != 200:
                raise Exception(f"Failed to get Jira issue: {result.get('message', 'Unknown error')}")
            
            return {
                "key": result["key"],
                "fields": result["fields"],
                "self": result["self"]
            }

    async def search_issues(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Search for Jira issues using JQL."""
        jql = inputs["jql"]
        max_results = inputs.get("max_results", 50)
        
        params = {
            "jql": jql,
            "maxResults": max_results,
            "fields": "summary,description,status,priority,assignee,labels"
        }
        
        async with self.session.get(
            f"{self.api_base}/search",
            params=params
        ) as response:
            result = await response.json()
            
            if response.status != 200:
                raise Exception(f"Failed to search Jira issues: {result.get('message', 'Unknown error')}")
            
            return {
                "total": result["total"],
                "issues": [
                    {
                        "key": issue["key"],
                        "fields": issue["fields"],
                        "self": issue["self"]
                    }
                    for issue in result["issues"]
                ]
            } 