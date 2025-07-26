# Example usage for demonstration purposes only
from multimind.agents.agent_registry import AgentRegistry

registry = AgentRegistry()
def agent_a(query, state=None):
    if "fail" in query:
        raise ValueError("Simulated failure")
    return {"response": f"A: {query}", "state": {"last": query}}
def agent_b(query, state=None):
    return {"response": f"B: {query}", "state": {"last": query}}
registry.register_agent("a", agent_a)
registry.register_agent("b", agent_b)
registry.set_fallback("a", "b")
print(registry.run_agent("a", "hello"))
print(registry.run_agent("a", "fail this"))
print("State after session:", registry.get_state(None)) 