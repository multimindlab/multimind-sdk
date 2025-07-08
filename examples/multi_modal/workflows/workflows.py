"""
Multi-modal workflow examples using MCP.
"""

from typing import Dict, Any, List
from multimind.api.mcp.base import MCPWorkflowAPI
from multimind.api.mcp.registry import WorkflowRegistry
from multimind.router.multi_modal_router import MultiModalRequest

@WorkflowRegistry.register
class ImageCaptionWorkflow(MCPWorkflowAPI):
    """Workflow for generating image captions with analysis."""
    
    def __init__(self, models: Dict[str, Any], integrations: Dict[str, Any]):
        super().__init__(
            name="Image Caption",
            description="Generate detailed captions for images with analysis",
            models=models,
            integrations=integrations
        )
    
    async def execute(self, request: MultiModalRequest) -> Dict[str, Any]:
        """Execute the image caption workflow."""
        # 1. Generate initial caption
        caption_result = await self.models["gpt4v"].process_image(
            request.content["image"]
        )
        
        # 2. Analyze caption with text model
        analysis_result = await self.models["gpt4"].generate(
            f"Analyze this image caption and provide insights: {caption_result['text']}"
        )
        
        # 3. Generate detailed description
        description_result = await self.models["claude"].generate(
            f"Based on this analysis, provide a detailed description of the image: {analysis_result['text']}"
        )
        
        return {
            "caption": caption_result["text"],
            "analysis": analysis_result["text"],
            "description": description_result["text"]
        }

@WorkflowRegistry.register
class AudioTranscriptionWorkflow(MCPWorkflowAPI):
    """Workflow for audio transcription with summarization."""
    
    def __init__(self, models: Dict[str, Any], integrations: Dict[str, Any]):
        super().__init__(
            name="Audio Transcription",
            description="Transcribe audio and generate summaries",
            models=models,
            integrations=integrations
        )
    
    async def execute(self, request: MultiModalRequest) -> Dict[str, Any]:
        """Execute the audio transcription workflow."""
        # 1. Transcribe audio
        transcription_result = await self.models["whisper"].process_audio(
            request.content["audio"]
        )
        
        # 2. Generate summary
        summary_result = await self.models["gpt4"].generate(
            f"Summarize this transcription: {transcription_result['text']}"
        )
        
        # 3. Extract key points
        key_points_result = await self.models["claude"].generate(
            f"Extract key points from this summary: {summary_result['text']}"
        )
        
        return {
            "transcription": transcription_result["text"],
            "summary": summary_result["text"],
            "key_points": key_points_result["text"]
        }

@WorkflowRegistry.register
class MultiModalAnalysisWorkflow(MCPWorkflowAPI):
    """Workflow for complex multi-modal analysis."""
    
    def __init__(self, models: Dict[str, Any], integrations: Dict[str, Any]):
        super().__init__(
            name="Multi-Modal Analysis",
            description="Analyze multiple modalities together",
            models=models,
            integrations=integrations
        )
    
    async def execute(self, request: MultiModalRequest) -> Dict[str, Any]:
        """Execute the multi-modal analysis workflow."""
        results = {}
        
        # Process each modality
        for modality, content in request.content.items():
            if modality == "image":
                results["image_analysis"] = await self.models["gpt4v"].process_image(content)
            elif modality == "audio":
                results["audio_analysis"] = await self.models["whisper"].process_audio(content)
            elif modality == "text":
                results["text_analysis"] = await self.models["gpt4"].generate(content)
        
        # Combine analyses
        combined_prompt = "Analyze these results together:\n"
        for modality, analysis in results.items():
            combined_prompt += f"\n{modality}: {analysis['text']}"
        
        # Generate final analysis
        final_analysis = await self.models["claude"].generate(combined_prompt)
        
        return {
            "modality_analyses": results,
            "combined_analysis": final_analysis["text"]
        }

# Example usage
async def run_workflow_example():
    """Run an example workflow."""
    from multimind.router.multi_modal_router import MultiModalRouter
    
    # Initialize router
    router = MultiModalRouter()
    
    # Create sample request
    request = MultiModalRequest(
        content={
            "image": "base64_encoded_image",
            "text": "Analyze this scene"
        },
        modalities=["image", "text"]
    )
    
    # Get workflow
    workflow = WorkflowRegistry.get_workflow("MultiModalAnalysis")
    
    # Execute workflow
    result = await workflow.execute(request)
    
    print("Workflow execution completed!")
    print("\nResults:")
    for key, value in result.items():
        print(f"\n{key}:")
        print(value)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_workflow_example()) 