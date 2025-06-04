"""
Slack integration handler for MCP workflows.
"""

from typing import Dict, Any, Optional
import aiohttp
import logging
from datetime import datetime
from .base import IntegrationHandler, AsyncContextManager

logger = logging.getLogger(__name__)

class SlackIntegrationHandler(IntegrationHandler, AsyncContextManager):
    """Handler for Slack integration operations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Slack integration handler."""
        super().__init__(config)
        self.validate_config(["token"])
        
        self.token = config["token"]
        self.default_channel = config.get("default_channel")
        self.api_base = "https://slack.com/api"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Set up aiohttp session."""
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.token}"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up aiohttp session."""
        if self.session:
            await self.session.close()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Slack integration operation."""
        try:
            operation = inputs.get("operation", "send_message")
            
            if operation == "send_message":
                result = await self.send_message(inputs)
            elif operation == "create_channel":
                result = await self.create_channel(inputs)
            elif operation == "list_channels":
                result = await self.list_channels()
            else:
                raise ValueError(f"Unsupported Slack operation: {operation}")
            
            self._update_metadata(success=True)
            return result
            
        except Exception as e:
            self._update_metadata(success=False)
            raise

    async def send_message(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to a Slack channel."""
        channel = inputs.get("channel", self.default_channel)
        if not channel:
            raise ValueError("No channel specified for Slack message")

        text = inputs["text"]
        blocks = inputs.get("blocks")
        
        payload = {
            "channel": channel,
            "text": text,
            "as_user": True
        }
        
        if blocks:
            payload["blocks"] = blocks

        async with self.session.post(
            f"{self.api_base}/chat.postMessage",
            json=payload
        ) as response:
            result = await response.json()
            
            if not result["ok"]:
                raise Exception(f"Failed to send Slack message: {result['error']}")
            
            return {
                "message_id": result["ts"],
                "channel": channel,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def create_channel(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Slack channel."""
        name = inputs["name"]
        is_private = inputs.get("is_private", False)
        
        payload = {
            "name": name,
            "is_private": is_private
        }
        
        async with self.session.post(
            f"{self.api_base}/conversations.create",
            json=payload
        ) as response:
            result = await response.json()
            
            if not result["ok"]:
                raise Exception(f"Failed to create Slack channel: {result['error']}")
            
            return {
                "channel_id": result["channel"]["id"],
                "name": result["channel"]["name"],
                "is_private": result["channel"]["is_private"]
            }

    async def list_channels(self) -> Dict[str, Any]:
        """List all accessible Slack channels."""
        async with self.session.get(
            f"{self.api_base}/conversations.list",
            params={"types": "public_channel,private_channel"}
        ) as response:
            result = await response.json()
            
            if not result["ok"]:
                raise Exception(f"Failed to list Slack channels: {result['error']}")
            
            return {
                "channels": [
                    {
                        "id": channel["id"],
                        "name": channel["name"],
                        "is_private": channel["is_private"],
                        "member_count": channel.get("num_members", 0)
                    }
                    for channel in result["channels"]
                ]
            } 