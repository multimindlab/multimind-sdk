from multimind.agents.reflexive.planner_agent import PlannerAgent
from multimind.agents.reflexive.thinker_agent import ThinkerAgent
from multimind.agents.reflexive.judge_agent import JudgeAgent
from multimind.agents.reflexive.rewriter_agent import RewriterAgent
from multimind.agents.reflexive.self_reflect_agent import SelfReflectAgent

# Dummy memory agent for ThinkerAgent (can be replaced with GraphMemoryAgent)
class DummyMemory:
    def query(self, predicate=None, obj=None):
        # For demo, return a fact if the goal is 'write a report'
        if obj == 'write a report':
            return [('Research', 'related_to', 'write a report', {})]
        return []

def main():
    prompt = "Research the topic and write a report, then review the report for clarity"
    planner = PlannerAgent()
    memory = DummyMemory()
    thinker = ThinkerAgent(memory=memory)
    judge = JudgeAgent()
    rewriter = RewriterAgent()
    self_reflect = SelfReflectAgent(thinker, memory=memory)

    # Step 1: Planning
    sub_tasks = planner.plan(prompt)
    print("Sub-tasks:", sub_tasks)

    # Step 2: Thinking/Reasoning
    all_reasonings = []
    for task in sub_tasks:
        reasoning = thinker.think(task)
        print(f"Reasoning for '{task}':", reasoning)
        all_reasonings.append(reasoning)

    # Step 3: Judging
    judged = judge.evaluate([step for reasoning in all_reasonings for step in reasoning])
    print("Judged steps:", judged)

    # Step 4: Rewriting (refinement)
    rewritten = [rewriter.rewrite(j['output'], feedback="Improve clarity") for j in judged]
    print("Rewritten steps:", rewritten)

    # Step 5: Reflexion (self-improvement loop)
    improved_plan = self_reflect.run_reflexion(prompt, feedback="Make the report more concise")
    print("Improved plan after reflexion:", improved_plan)

if __name__ == '__main__':
    main() 