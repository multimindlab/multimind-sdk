from typing import Dict, Callable, Any, Optional
import logging

class AgentRegistry:
    """
    Central registry for agents, with retry/fallback and conversational state memory.
    """
    def __init__(self):
        self.agents: Dict[str, Callable] = {}
        self.fallbacks: Dict[str, str] = {}  # agent_name -> fallback_agent_name
        self.state_memory: Dict[str, Any] = {}  # session_id -> state
        self.logger = logging.getLogger("AgentRegistry")

    def register_agent(self, name: str, agent: Callable):
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")

    def set_fallback(self, name: str, fallback_name: str):
        self.fallbacks[name] = fallback_name
        self.logger.info(f"Set fallback for {name} -> {fallback_name}")

    def get_agent(self, name: str) -> Optional[Callable]:
        return self.agents.get(name)

    def run_agent(self, name: str, *args, session_id: Optional[str] = None, **kwargs):
        agent = self.get_agent(name)
        if not agent:
            self.logger.warning(f"Agent {name} not found.")
            return None
        try:
            # Pass state if available
            state = self.state_memory.get(session_id) if session_id else None
            result = agent(*args, state=state, **kwargs)
            # Optionally update state
            if session_id is not None:
                self.state_memory[session_id] = result.get("state", state) if isinstance(result, dict) else state
            return result
        except Exception as e:
            self.logger.error(f"Agent {name} failed: {e}")
            fallback = self.fallbacks.get(name)
            if fallback:
                self.logger.info(f"Retrying with fallback agent: {fallback}")
                return self.run_agent(fallback, *args, session_id=session_id, **kwargs)
            return None

    def get_state(self, session_id: str):
        return self.state_memory.get(session_id)

    def set_state(self, session_id: str, state: Any):
        self.state_memory[session_id] = state 