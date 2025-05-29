"""
Task runner example demonstrating how to use the task runner for complex workflows.
"""

import asyncio
import os
from dotenv import load_dotenv
from multimind import OpenAIModel, TaskRunner, PromptChain

async def main():
    # Load environment variables
    load_dotenv()
    
    # Create model
    model = OpenAIModel(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create task runner
    runner = TaskRunner(model)
    
    # Create prompt chain for research
    research_chain = PromptChain(model)
    research_chain.add_prompt(
        prompt="Research the following topic: {topic}\nProvide key points and findings.",
        name="research"
    )
    research_chain.add_prompt(
        prompt="Based on the research, identify potential applications and implications:\n{last_response}",
        name="applications"
    )
    
    # Add tasks with dependencies
    runner.add_task(
        name="topic_analysis",
        prompt="Analyze the following topic and break it down into key aspects: {topic}",
        retry_prompt="Please analyze this topic again, focusing on the main components: {topic}"
    )
    
    runner.add_task(
        name="research",
        prompt=research_chain,
        dependencies=["topic_analysis"]
    )
    
    runner.add_task(
        name="summary",
        prompt="""Create a comprehensive summary of the research:
        Topic Analysis: {topic_analysis}
        Research Findings: {research}
        
        Include:
        1. Key points
        2. Applications
        3. Future implications""",
        dependencies=["research"]
    )
    
    runner.add_task(
        name="recommendations",
        prompt="""Based on the research summary, provide actionable recommendations:
        {summary}
        
        Focus on:
        1. Immediate actions
        2. Long-term strategies
        3. Potential challenges""",
        dependencies=["summary"]
    )
    
    # Example topic
    topic = "Artificial Intelligence in Healthcare"
    
    # Run tasks
    results = await runner.run({"topic": topic})
    
    # Print results
    print("Research Workflow Results:")
    print("========================")
    
    for task_name, result in results.items():
        print(f"\n{task_name.upper()}:")
        print("-" * len(task_name))
        print(result)
        print()

if __name__ == "__main__":
    asyncio.run(main()) 