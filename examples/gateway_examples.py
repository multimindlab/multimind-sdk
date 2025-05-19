"""
Examples demonstrating the MultiMind Gateway features
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from multimind.gateway import (
    MultiMindCLI,
    chat_manager,
    monitor,
    get_model_handler,
    config
)

async def example_chat_session():
    """Example of creating and managing chat sessions"""
    print("\n=== Chat Session Example ===")
    
    # Create a new chat session
    session = chat_manager.create_session(
        model="openai",
        system_prompt="You are a helpful AI assistant.",
        metadata={"purpose": "example", "user": "demo"}
    )
    print(f"Created session: {session.session_id}")
    
    # Add some messages
    session.add_message(
        role="user",
        content="What is the capital of France?",
        model="openai",
        metadata={"topic": "geography"}
    )
    
    # Get a response
    handler = get_model_handler("openai")
    response = await handler.generate("What is the capital of France?")
    
    session.add_message(
        role="assistant",
        content=response.content,
        model="openai",
        metadata={"tokens": response.usage.get("total_tokens", 0) if response.usage else 0}
    )
    
    # Save the session
    file_path = chat_manager.save_session(session.session_id)
    print(f"Saved session to: {file_path}")
    
    # Export session
    export_data = session.export(format="json")
    print("\nSession Export:")
    print(json.dumps(json.loads(export_data), indent=2))
    
    return session.session_id

async def example_monitoring():
    """Example of monitoring model health and metrics"""
    print("\n=== Monitoring Example ===")
    
    # Check health of all models
    print("\nChecking model health...")
    for model in config.validate().keys():
        if config.validate()[model]:
            handler = get_model_handler(model)
            health = await monitor.check_health(model, handler)
            print(f"\n{model.upper()} Health:")
            print(f"Status: {'✅' if health.is_healthy else '❌'}")
            print(f"Latency: {health.latency_ms:.0f}ms" if health.latency_ms else "N/A")
            print(f"Last Check: {health.last_check}")
    
    # Make some requests to generate metrics
    print("\nGenerating some requests for metrics...")
    models = ["openai", "anthropic"]  # Example models
    
    for model in models:
        if config.validate()[model]:
            handler = get_model_handler(model)
            try:
                # Successful request
                start_time = datetime.now()
                response = await handler.generate("Hello, world!")
                response_time = (datetime.now() - start_time).total_seconds()
                
                await monitor.track_request(
                    model=model,
                    tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
                    cost=0.0,  # Example cost
                    response_time=response_time,
                    success=True
                )
                
                # Failed request (example)
                await monitor.track_request(
                    model=model,
                    tokens=0,
                    cost=0.0,
                    response_time=0.1,
                    success=False,
                    error="Example error"
                )
            except Exception as e:
                print(f"Error with {model}: {e}")
    
    # Get and display metrics
    print("\nModel Metrics:")
    metrics = await monitor.get_metrics()
    for model, data in metrics.items():
        m = data["metrics"]
        print(f"\n{model.upper()}:")
        print(f"Total Requests: {m.total_requests}")
        print(f"Success Rate: {(m.successful_requests / m.total_requests * 100):.1f}%" if m.total_requests > 0 else "N/A")
        print(f"Average Response Time: {m.avg_response_time:.2f}s")
        print(f"Total Tokens: {m.total_tokens}")
        print(f"Total Cost: ${m.total_cost:.4f}")

async def example_rate_limiting():
    """Example of rate limiting"""
    print("\n=== Rate Limiting Example ===")
    
    # Set custom rate limits
    monitor.set_rate_limits(
        model="openai",
        requests_per_minute=10,
        tokens_per_minute=1000
    )
    
    print("Set rate limits for OpenAI:")
    print(f"Requests per minute: {monitor.rate_limits['openai']['requests_per_minute']}")
    print(f"Tokens per minute: {monitor.rate_limits['openai']['tokens_per_minute']}")
    
    # Check rate limits
    can_proceed = await monitor.check_rate_limit("openai", tokens=100)
    print(f"\nCan proceed with request: {can_proceed}")

async def main():
    """Run all examples"""
    try:
        # Initialize CLI to validate config
        cli = MultiMindCLI()
        cli.validate_config()
        
        # Run examples
        session_id = await example_chat_session()
        await example_monitoring()
        await example_rate_limiting()
        
        # Clean up
        chat_manager.delete_session(session_id)
        print("\nCleaned up example session")
        
    except Exception as e:
        print(f"Error running examples: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 