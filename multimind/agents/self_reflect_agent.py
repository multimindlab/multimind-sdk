from typing import List, Optional
from multimind.agents.thinker_agent import ThinkerAgent
from multimind.memory.graph_memory import GraphMemoryAgent

class SelfReflectAgent:
    """
    Loops through past plans/memories and updates them using feedback.
    Integrates with ThinkerAgent and GraphMemoryAgent for reflexive improvement.
    """
    def __init__(self, thinker_agent: ThinkerAgent, memory_agent: GraphMemoryAgent):
        self.thinker_agent = thinker_agent
        self.memory_agent = memory_agent
        self.past_plans: List[List[str]] = []
        self.feedback_history: List[str] = []

    def run_reflexion(self, goal: str, feedback: Optional[str] = None) -> List[str]:
        """
        Runs a self-improvement loop: plans, reflects, and updates memory with feedback.
        Returns the improved plan.
        """
        plan = self.thinker_agent.plan(goal)
        if feedback:
            plan = self.thinker_agent.reflect(plan, feedback)
            self.feedback_history.append(feedback)
        self.past_plans.append(plan)
        # Optionally, store the improved plan in memory as a new fact
        self.memory_agent.add_fact(goal, 'improved_plan', str(plan))
        return plan

    def get_past_plans(self) -> List[List[str]]:
        """
        Returns all past plans for inspection or further reflection.
        """
        return self.past_plans 