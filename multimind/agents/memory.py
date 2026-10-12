"""
Memory management for agents.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


class AgentMemory:
    """Manages agent memory and state."""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.tasks: List[str] = []
        self.task_timestamps: List[datetime] = []
        self.responses: List[Dict[str, Any]] = []
        self.response_timestamps: List[datetime] = []
        self.state: Dict[str, Any] = {}
        self.created_at = datetime.now()

    def add_task(self, task: str) -> None:
        """Add a task to memory."""
        self.tasks.append(task)
        self.task_timestamps.append(datetime.now())
        if len(self.tasks) > self.max_history:
            self.tasks.pop(0)
            if self.task_timestamps:
                self.task_timestamps.pop(0)

    def add_response(self, response: Dict[str, Any]) -> None:
        """Add a response to memory."""
        self.responses.append(response)
        self.response_timestamps.append(datetime.now())
        if len(self.responses) > self.max_history:
            self.responses.pop(0)
            if self.response_timestamps:
                self.response_timestamps.pop(0)

    def update_state(self, key: str, value: Any) -> None:
        """Update agent state."""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from agent state."""
        return self.state.get(key, default)

    def get_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent history of tasks and responses."""
        if n is None:
            n = self.max_history

        history = []
        # Use timestamps recorded at insertion time (task/response) instead of `datetime.now()`
        recent_tasks = self.tasks[-n:]
        recent_task_timestamps = self.task_timestamps[-n:]
        recent_responses = self.responses[-n:]
        recent_response_timestamps = self.response_timestamps[-n:]

        for task, response, task_ts, resp_ts in zip(
            recent_tasks,
            recent_responses,
            recent_task_timestamps,
            recent_response_timestamps,
        ):
            history.append(
                {
                    "task": task,
                    "response": response,
                    # Prefer response timestamp because it reflects when the completion arrived.
                    "timestamp": resp_ts.isoformat() if isinstance(resp_ts, datetime) else None,
                    "task_timestamp": (
                        task_ts.isoformat() if isinstance(task_ts, datetime) else None
                    ),
                }
            )
        return history

    def clear(self) -> None:
        """Clear all memory."""
        self.tasks.clear()
        self.task_timestamps.clear()
        self.responses.clear()
        self.response_timestamps.clear()
        self.state.clear()
