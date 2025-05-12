"""
Usage tracking example demonstrating how to monitor model usage and costs.
"""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from multimind import (
    OpenAIModel, ClaudeModel,
    UsageTracker, TraceLogger
)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize trackers
    usage_tracker = UsageTracker("usage.db")
    trace_logger = TraceLogger("logs")
    
    # Set model costs (example costs)
    usage_tracker.set_model_costs(
        model="gpt-3.5-turbo",
        input_cost_per_token=0.0000015,
        output_cost_per_token=0.000002
    )
    
    usage_tracker.set_model_costs(
        model="claude-3-sonnet",
        input_cost_per_token=0.000015,
        output_cost_per_token=0.000075
    )
    
    # Create models
    openai_model = OpenAIModel(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    claude_model = ClaudeModel(
        model="claude-3-sonnet-20240229",
        temperature=0.7
    )
    
    # Start trace
    trace_id = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trace_logger.start_trace(
        trace_id=trace_id,
        operation="model_comparison",
        metadata={
            "models": ["gpt-3.5-turbo", "claude-3-sonnet"],
            "task": "text generation comparison"
        }
    )
    
    # Example prompts
    prompts = [
        "Explain quantum computing in simple terms",
        "Write a short story about a robot learning to paint",
        "Analyze the impact of social media on modern society"
    ]
    
    # Run prompts through both models
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # OpenAI
        trace_logger.add_event(
            trace_id=trace_id,
            event_type="model_call",
            data={
                "model": "gpt-3.5-turbo",
                "prompt": prompt
            }
        )
        
        response = await openai_model.generate(prompt)
        
        # Track OpenAI usage
        usage_tracker.track_usage(
            model="gpt-3.5-turbo",
            operation="text_generation",
            input_tokens=len(prompt.split()),  # Approximate
            output_tokens=len(response.split()),  # Approximate
            metadata={
                "prompt": prompt,
                "response_length": len(response)
            }
        )
        
        print("\nOpenAI Response:")
        print(response)
        
        # Claude
        trace_logger.add_event(
            trace_id=trace_id,
            event_type="model_call",
            data={
                "model": "claude-3-sonnet",
                "prompt": prompt
            }
        )
        
        response = await claude_model.generate(prompt)
        
        # Track Claude usage
        usage_tracker.track_usage(
            model="claude-3-sonnet",
            operation="text_generation",
            input_tokens=len(prompt.split()),  # Approximate
            output_tokens=len(response.split()),  # Approximate
            metadata={
                "prompt": prompt,
                "response_length": len(response)
            }
        )
        
        print("\nClaude Response:")
        print(response)
    
    # End trace
    trace_logger.end_trace(
        trace_id=trace_id,
        status="success",
        result={"prompts_processed": len(prompts)}
    )
    
    # Get usage summary
    print("\nUsage Summary:")
    print("=============")
    
    # Last 24 hours
    start_date = (datetime.now() - timedelta(days=1)).isoformat()
    summary = usage_tracker.get_usage_summary(start_date=start_date)
    
    print("\nTotal Cost:", f"${summary['total_cost']:.4f}")
    
    for model, data in summary["models"].items():
        print(f"\n{model}:")
        print(f"Total Cost: ${data['total_cost']:.4f}")
        
        for operation, stats in data["operations"].items():
            print(f"\n  {operation}:")
            print(f"  Count: {stats['count']}")
            print(f"  Input Tokens: {stats['input_tokens']}")
            print(f"  Output Tokens: {stats['output_tokens']}")
            print(f"  Cost: ${stats['cost']:.4f}")
    
    # Export usage data
    usage_tracker.export_usage(
        "usage_report.json",
        format="json",
        start_date=start_date
    )
    print("\nUsage report exported to usage_report.json")

if __name__ == "__main__":
    asyncio.run(main()) 