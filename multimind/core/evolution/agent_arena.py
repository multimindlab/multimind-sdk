from typing import Any, List, Dict, Callable
import asyncio

class AgentArena:
    """
    Runs multiple agents in parallel for the same task and selects the best output using a judge agent.
    """
    def __init__(self, agents: List[Any], judge: Callable[[List[Any]], Any]):
        self.agents = agents
        self.judge = judge  # Function or agent that selects the best output

    async def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Runs all agents in parallel, collects outputs, and selects the best.
        Returns a dict with all outputs and the selected best.
        """
        tasks = [agent.run(input_data) for agent in self.agents]
        outputs = await asyncio.gather(*tasks)
        best = self.judge(outputs)
        return {
            'outputs': outputs,
            'best': best
        }

# Example judge function for demo: pick the longest output
async def longest_output_judge(outputs: List[Any]) -> Any:
    return max(outputs, key=lambda x: len(str(x))) 