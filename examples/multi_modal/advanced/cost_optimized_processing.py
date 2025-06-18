"""
Example demonstrating cost-optimized multi-modal processing with dynamic model selection.
"""

import asyncio
import base64
from pathlib import Path
from typing import Dict, List, Optional

from multimind.router.multi_modal_router import MultiModalRouter
from multimind.models.advanced import CostOptimizedWrapper
from multimind.types import UnifiedRequest, ModalityInput, ModalityOutput
from multimind.metrics.cost_tracker import CostTracker
from multimind.metrics.performance import PerformanceTracker


class CostOptimizedMultiModalProcessor:
    """Handles cost-optimized processing of multi-modal requests."""
    
    def __init__(
        self,
        router: MultiModalRouter,
        cost_tracker: CostTracker,
        performance_tracker: PerformanceTracker,
        budget: float = 0.1
    ):
        self.router = router
        self.cost_tracker = cost_tracker
        self.performance_tracker = performance_tracker
        self.budget = budget
        
    async def process_request(
        self,
        request: UnifiedRequest,
        optimize_cost: bool = True
    ) -> Dict:
        """Process a multi-modal request with cost optimization."""
        
        # Track start time
        start_time = self.performance_tracker.get_current_time()
        
        try:
            # Process each modality
            results = {}
            total_cost = 0.0
            
            for input_data in request.inputs:
                # Check budget
                if optimize_cost and total_cost >= self.budget:
                    raise ValueError(f"Budget exceeded: {total_cost} > {self.budget}")
                
                # Process modality
                result = await self._process_modality(input_data)
                
                # Track cost
                modality_cost = self.cost_tracker.get_modality_cost(
                    input_data.modality,
                    result
                )
                total_cost += modality_cost
                
                # Store result
                results[input_data.modality] = result
            
            # Track performance
            end_time = self.performance_tracker.get_current_time()
            latency = end_time - start_time
            self.performance_tracker.track_latency("multi_modal", latency)
            
            return {
                "results": results,
                "cost": total_cost,
                "latency": latency
            }
            
        except Exception as e:
            # Track error
            self.performance_tracker.track_error("multi_modal", str(e))
            raise
    
    async def _process_modality(
        self,
        input_data: ModalityInput
    ) -> ModalityOutput:
        """Process a single modality with cost optimization."""
        
        # Get cost-optimized model
        model = self._get_cost_optimized_model(input_data.modality)
        
        # Process with model
        result = await self.router.process_modality(
            input_data,
            model=model
        )
        
        return result
    
    def _get_cost_optimized_model(self, modality: str) -> str:
        """Get the most cost-effective model for a modality."""
        
        # Get available models
        models = self.router.get_available_models(modality)
        
        # Get cost metrics
        cost_metrics = self.cost_tracker.get_modality_metrics(modality)
        
        # Get performance metrics
        perf_metrics = self.performance_tracker.get_modality_metrics(modality)
        
        # Select best model based on cost and performance
        best_model = None
        best_score = float('inf')
        
        for model in models:
            cost = cost_metrics.get(model, {}).get("avg_cost", float('inf'))
            perf = perf_metrics.get(model, {}).get("success_rate", 0.0)
            
            # Calculate score (lower is better)
            score = cost / (perf + 0.1)  # Add small constant to avoid division by zero
            
            if score < best_score:
                best_score = score
                best_model = model
        
        return best_model or models[0]


async def main():
    """Run the cost-optimized multi-modal processing example."""
    
    # Initialize components
    router = MultiModalRouter()
    cost_tracker = CostTracker()
    performance_tracker = PerformanceTracker()
    
    # Create processor
    processor = CostOptimizedMultiModalProcessor(
        router=router,
        cost_tracker=cost_tracker,
        performance_tracker=performance_tracker,
        budget=0.1
    )
    
    # Load sample data
    data_dir = Path("examples/data")
    image_path = data_dir / "sample_image.jpg"
    audio_path = data_dir / "sample_audio.mp3"
    
    # Read files
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()
    
    # Create request
    request = UnifiedRequest(
        inputs=[
            ModalityInput(
                modality="image",
                content=image_data
            ),
            ModalityInput(
                modality="audio",
                content=audio_data
            ),
            ModalityInput(
                modality="text",
                content="Analyze this image and audio content"
            )
        ]
    )
    
    try:
        # Process request
        result = await processor.process_request(
            request,
            optimize_cost=True
        )
        
        # Print results
        print("\nProcessing Results:")
        print(f"Total Cost: ${result['cost']:.4f}")
        print(f"Latency: {result['latency']:.2f}s")
        
        print("\nModality Results:")
        for modality, output in result["results"].items():
            print(f"\n{modality.upper()}:")
            print(f"Content: {output.content[:100]}...")
            print(f"Confidence: {output.confidence:.2f}")
            
        # Print metrics
        print("\nCost Metrics:")
        for modality in ["image", "audio", "text"]:
            metrics = cost_tracker.get_modality_metrics(modality)
            print(f"\n{modality.upper()}:")
            for model, stats in metrics.items():
                print(f"  {model}:")
                print(f"    Avg Cost: ${stats['avg_cost']:.4f}")
                print(f"    Total Cost: ${stats['total_cost']:.4f}")
        
        print("\nPerformance Metrics:")
        for modality in ["image", "audio", "text"]:
            metrics = performance_tracker.get_modality_metrics(modality)
            print(f"\n{modality.upper()}:")
            for model, stats in metrics.items():
                print(f"  {model}:")
                print(f"    Success Rate: {stats['success_rate']:.2%}")
                print(f"    Avg Latency: {stats['avg_latency']:.2f}s")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 