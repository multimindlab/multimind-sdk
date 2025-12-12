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
    
    # Check which API keys are available
    openai_key = os.getenv("OPENAI_API_KEY")
    claude_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    
    # Set model costs (example costs)
    available_models = []
    
    if openai_key:
        usage_tracker.set_model_costs(
            model="gpt-3.5-turbo",
            input_cost_per_token=0.0000015,
            output_cost_per_token=0.000002
        )
        available_models.append("gpt-3.5-turbo")
        print("✓ OpenAI API key found - OpenAI model will be used")
    
    if claude_key:
        usage_tracker.set_model_costs(
            model="claude-3-sonnet",
            input_cost_per_token=0.000015,
            output_cost_per_token=0.000075
        )
        available_models.append("claude-3-sonnet")
        print("✓ Claude API key found - Claude model will be used")
    
    if not available_models:
        print("Error: No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY/CLAUDE_API_KEY in your .env file")
        return
    
    print(f"\nUsing {len(available_models)} model(s): {', '.join(available_models)}\n")
    
    # Create models only if API keys are available
    models = {}
    
    if openai_key:
        try:
            models["openai"] = {
                "model": OpenAIModel(
                    model_name="gpt-3.5-turbo",
                    temperature=0.7
                ),
                "name": "gpt-3.5-turbo",
                "display_name": "OpenAI"
            }
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI model: {e}")
            available_models = [m for m in available_models if m != "gpt-3.5-turbo"]
    
    if claude_key:
        try:
            models["claude"] = {
                "model": ClaudeModel(
                    model_name="claude-3-sonnet-20240229",
                    temperature=0.7
                ),
                "name": "claude-3-sonnet",
                "display_name": "Claude"
            }
        except Exception as e:
            print(f"Warning: Failed to initialize Claude model: {e}")
            available_models = [m for m in available_models if m != "claude-3-sonnet"]
    
    if not models:
        print("Error: Could not initialize any models")
        return
    
    # Start trace
    trace_id = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trace_logger.start_trace(
        trace_id=trace_id,
        operation="model_comparison",
        metadata={
            "models": available_models,
            "task": "text generation comparison"
        }
    )
    
    # Example prompts
    prompts = [
        "Explain quantum computing in simple terms",
        "Write a short story about a robot learning to paint",
        "Analyze the impact of social media on modern society"
    ]
    
    # Run prompts through available models
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {i+1}: {prompt}")
        print('='*60)
        
        # Process with each available model
        for model_key, model_info in models.items():
            try:
                print(f"\n[{model_info['display_name']}] Processing...")
                
                trace_logger.add_event(
                    trace_id=trace_id,
                    event_type="model_call",
                    data={
                        "model": model_info["name"],
                        "prompt": prompt
                    }
                )
                
                response = await model_info["model"].generate(prompt)
                
                # Track usage
                usage_tracker.track_usage(
                    model=model_info["name"],
                    operation="text_generation",
                    input_tokens=len(prompt.split()),  # Approximate
                    output_tokens=len(response.split()),  # Approximate
                    metadata={
                        "prompt": prompt,
                        "response_length": len(response)
                    }
                )
                
                print(f"\n{model_info['display_name']} Response:")
                print("-" * 40)
                print(response)
                
            except Exception as e:
                print(f"\nError with {model_info['display_name']}: {e}")
                trace_logger.add_event(
                    trace_id=trace_id,
                    event_type="error",
                    data={
                        "model": model_info["name"],
                        "error": str(e)
                    }
                )
    
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