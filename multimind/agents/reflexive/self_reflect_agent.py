from typing import List, Optional, Any
from multimind.agents.reflexive.thinker_agent import ThinkerAgent

class SelfReflectAgent:
    """
    Loops through past plans/memories and updates them using feedback.
    Integrates with ThinkerAgent and memory for reflexive improvement.
    Modular and developer-friendly for use in pipelines or reflexive loops.
    """
    def __init__(self, thinker_agent: ThinkerAgent, memory=None):
        self.thinker_agent = thinker_agent
        self.memory = memory
        self.past_plans: List[List[str]] = []
        self.feedback_history: List[Any] = []

    def run_reflexion(self, goal: str, feedback: Optional[Any] = None) -> List[str]:
        """
        Runs a self-improvement loop: plans, reflects, and updates memory with feedback.
        Returns the improved plan.
        """
        plan = self.thinker_agent.think(goal)
        if feedback:
            plan = plan + [f"Reflection: {feedback}"]
            self.feedback_history.append(feedback)
        self.past_plans.append(plan)
        # Optionally, store the improved plan in memory as a new fact
        if self.memory and hasattr(self.memory, 'add_fact'):
            self.memory.add_fact(goal, 'improved_plan', str(plan))
        return plan

    def get_past_plans(self) -> List[List[str]]:
        """
        Returns all past plans for inspection or further reflection.
        """
        return self.past_plans 