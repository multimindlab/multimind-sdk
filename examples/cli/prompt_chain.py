"""
Prompt chain example demonstrating how to use prompt chains for complex reasoning tasks.
"""

import asyncio
import os
from dotenv import load_dotenv
from multimind import OpenAIModel, PromptChain

async def main():
    # Load environment variables
    load_dotenv()
    
    # Create model
    model = OpenAIModel(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create prompt chain for code review
    code_review_chain = PromptChain(model)
    
    # Add prompts for code review
    code_review_chain.add_prompt(
        prompt="""Analyze the following code for potential issues:
        {code}
        
        Focus on:
        1. Security vulnerabilities
        2. Performance issues
        3. Code style and best practices
        4. Potential bugs
        
        Provide a detailed analysis.""",
        name="code_analysis"
    )
    
    code_review_chain.add_prompt(
        prompt="""Based on the previous analysis, suggest specific improvements:
        {last_response}
        
        For each issue identified:
        1. Provide a concrete solution
        2. Include example code if applicable
        3. Explain why the solution is better
        
        Format the response as a list of improvements.""",
        name="improvements"
    )
    
    code_review_chain.add_prompt(
        prompt="""Summarize the code review in a concise format:
        {last_response}
        
        Include:
        1. Number of issues found
        2. Severity levels
        3. Key recommendations
        
        Keep it brief but informative.""",
        name="summary"
    )
    
    # Example code to review
    code = """
    def process_user_data(user_input):
        # Process user data without validation
        result = eval(user_input)  # Security risk!
        
        # Inefficient loop
        for i in range(1000000):
            result += i
            
        return result
    """
    
    # Set variables
    code_review_chain.set_variable("code", code)
    
    # Run chain
    results = await code_review_chain.run()
    
    # Print results
    print("Code Review Results:")
    print("===================")
    
    for result in results:
        print(f"\n{result['name'].upper()}:")
        print("-" * len(result['name']))
        print(result['response'])
        print()

if __name__ == "__main__":
    asyncio.run(main()) 