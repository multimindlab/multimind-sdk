from multimind.agents.reflexive.planner_agent import PlannerAgent
from multimind.agents.reflexive.thinker_agent import ThinkerAgent
from multimind.agents.reflexive.judge_agent import JudgeAgent
from multimind.agents.reflexive.rewriter_agent import RewriterAgent
from multimind.agents.reflexive.self_reflect_agent import SelfReflectAgent
from multimind.core.evolution.agent_arena import AgentArena
from multimind.core.evolution.multi_objective_judge_agent import MultiObjectiveJudgeAgent
from multimind.core.evolution.meta_controller_agent import MetaControllerAgent
from multimind.core.evolution.agent_mutator import AgentMutator

# Dummy memory agent for ThinkerAgent (can be replaced with GraphMemoryAgent)
class DummyMemory:
    def query(self, predicate=None, obj=None):
        if obj == 'write a report':
            return [('Research', 'related_to', 'write a report', {})]
        return []

# Reflexive pipeline agent: planner -> thinker -> judge -> rewriter -> self-reflect
class ReflexivePipelineAgent:
    def __init__(self, name, memory=None):
        self.name = name
        self.planner = PlannerAgent()
        self.thinker = ThinkerAgent(memory=memory)
        self.judge = JudgeAgent()
        self.rewriter = RewriterAgent()
        self.self_reflect = SelfReflectAgent(self.thinker, memory=memory)

    async def run(self, prompt):
        sub_tasks = self.planner.plan(prompt)
        all_reasonings = []
        for task in sub_tasks:
            reasoning = self.thinker.think(task)
            all_reasonings.extend(reasoning)
        judged = self.judge.evaluate(all_reasonings)
        rewritten = [self.rewriter.rewrite(j['output'], feedback=f"{self.name} feedback") for j in judged]
        improved_plan = self.self_reflect.run_reflexion(prompt, feedback=f"{self.name} self-improvement")
        return {
            'agent': self.name,
            'sub_tasks': sub_tasks,
            'reasonings': all_reasonings,
            'judged': judged,
            'rewritten': rewritten,
            'improved_plan': improved_plan
        }

import asyncio

async def main():
    memory = DummyMemory()
    agents = [ReflexivePipelineAgent(f"ReflexiveAgent{i+1}", memory=memory) for i in range(3)]
    judge = MultiObjectiveJudgeAgent(weights={'accuracy': 1.0, 'cost': 1.0, 'speed': 1.0, 'creativity': 1.0})
    arena = AgentArena(agents, lambda outputs: judge.score([{'accuracy': len(a['rewritten']), 'cost': 1.0, 'speed': 1.0, 'creativity': 1.0} for a in outputs]))
    meta_controller = MetaControllerAgent()
    agent_mutator = AgentMutator()

    prompt = "Research the topic and write a report, then review the report for clarity"

    for round_num in range(2):
        print(f"\n=== Hybrid Evolutionary + Reflexive Round {round_num+1} ===")
        result = await arena.run(prompt)
        for i, output in enumerate(result['outputs']):
            print(f"Agent {i+1} output:", output)
        print("Best output (by judge):", result['best'])
        # Mutate agents for next round
        meta_controller.mutate_dag(arena, {a.name: a for a in agents})
        agent_mutator.mutate(arena, {a.name: a for a in agents})

if __name__ == '__main__':
    asyncio.run(main()) 