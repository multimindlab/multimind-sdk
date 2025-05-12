"""
Memory management for agents.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

class AgentMemory:
    """Manages agent memory and state."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.tasks: List[str] = []
        self.responses: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {}
        self.created_at = datetime.now()
        
    def add_task(self, task: str) -> None:
        """Add a task to memory."""
        self.tasks.append(task)
        if len(self.tasks) > self.max_history:
            self.tasks.pop(0)
            
    def add_response(self, response: Dict[str, Any]) -> None:
        """Add a response to memory."""
        self.responses.append(response)
        if len(self.responses) > self.max_history:
            self.responses.pop(0)
            
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
        for task, response in zip(self.tasks[-n:], self.responses[-n:]):
            history.append({
                "task": task,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
        return history
        
    def clear(self) -> None:
        """Clear all memory."""
        self.tasks.clear()
        self.responses.clear()
        self.state.clear() 