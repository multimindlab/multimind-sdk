"""
Parser for Model Composition Protocol (MCP) specifications.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path

class MCPParser:
    """Parses and validates MCP specifications."""

    def __init__(self, schema_path: Optional[str] = None):
        self.schema_path = schema_path or str(Path(__file__).parent / "schema.json")
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        """Load MCP schema from file."""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load MCP schema: {str(e)}")

    def parse(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate an MCP specification."""
        # Basic structure validation
        required_keys = {"version", "models", "workflow"}
        if not all(key in spec for key in required_keys):
            raise ValueError(f"MCP spec must contain: {required_keys}")

        # Version check
        if spec["version"] != self.schema["version"]:
            raise ValueError(
                f"Unsupported MCP version: {spec['version']}. "
                f"Expected: {self.schema['version']}"
            )

        # Validate models
        self._validate_models(spec["models"])

        # Validate workflow
        self._validate_workflow(spec["workflow"])

        return spec

    def _validate_models(self, models: List[Dict[str, Any]]) -> None:
        """Validate model specifications."""
        if not models:
            raise ValueError("At least one model must be specified")

        for model in models:
            # Check required fields
            required = {"name", "type", "config"}
            if not all(field in model for field in required):
                raise ValueError(f"Model must contain: {required}")

            # Validate model type
            if model["type"] not in self.schema["model_types"]:
                raise ValueError(
                    f"Unsupported model type: {model['type']}. "
                    f"Supported types: {self.schema['model_types']}"
                )

    def _validate_workflow(self, workflow: Dict[str, Any]) -> None:
        """Validate workflow specification."""
        # Check required fields
        required = {"steps", "connections"}
        if not all(field in workflow for field in required):
            raise ValueError(f"Workflow must contain: {required}")

        # Validate steps
        for step in workflow["steps"]:
            required = {"id", "type", "config"}
            if not all(field in step for field in required):
                raise ValueError(f"Workflow step must contain: {required}")

            # Validate step type
            if step["type"] not in self.schema["step_types"]:
                raise ValueError(
                    f"Unsupported step type: {step['type']}. "
                    f"Supported types: {self.schema['step_types']}"
                )

        # Validate connections
        step_ids = {step["id"] for step in workflow["steps"]}
        for conn in workflow["connections"]:
            required = {"from", "to"}
            if not all(field in conn for field in required):
                raise ValueError(f"Connection must contain: {required}")

            # Check if connected steps exis
            if conn["from"] not in step_ids or conn["to"] not in step_ids:
                raise ValueError(
                    f"Invalid connection: step {conn['from']} or {conn['to']} "
                    "does not exist"
                )

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse MCP specification from file."""
        try:
            with open(file_path, 'r') as f:
                spec = json.load(f)
            return self.parse(spec)
        except Exception as e:
            raise ValueError(f"Failed to parse MCP file {file_path}: {str(e)}")