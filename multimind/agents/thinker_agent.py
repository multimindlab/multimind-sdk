from typing import List, Optional
from multimind.memory.graph_memory import GraphMemoryAgent

class ThinkerAgent:
    """
    General-purpose reasoning agent (multi-domain, chain-of-thought capable).
    Performs strategic planning and reflection using GraphMemoryAgent.
    """
    def __init__(self, memory_agent: GraphMemoryAgent):
        self.memory_agent = memory_agent

    def plan(self, goal: str, context: Optional[str] = None) -> List[str]:
        """
        Generate a simple plan (list of steps) to achieve a goal, using memory.
        For demo: queries memory for relevant facts and chains them as steps.
        """
        # Query memory for facts related to the goal
        facts = self.memory_agent.query(predicate='related_to', obj=goal)
        steps = []
        for subj, pred, obj, meta in facts:
            steps.append(f"Step: Use {subj} because it is {pred} {obj}")
        if not steps:
            steps.append(f"No direct memory found for goal '{goal}'. Try researching or asking for more info.")
        return steps

    def reflect(self, last_plan: List[str], feedback: Optional[str] = None) -> List[str]:
        """
        Reflect on the last plan and optionally update it based on feedback.
        For demo: just appends feedback if provided.
        """
        if feedback:
            return last_plan + [f"Reflection: {feedback}"]
        return last_plan 