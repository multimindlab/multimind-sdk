"""
Cross-modal retrieval workflow using MCP.

This workflow demonstrates how to use MCP to create a cross-modal retrieval system
that can search across different modalities (text, image, audio) and retrieve
relevant content from other modalities.
"""

import asyncio
import base64
from pathlib import Path
import os
from typing import Dict, Any, List, Optional
from multimind.api.mcp.base import MCPWorkflowAPI
from multimind.api.mcp.registry import WorkflowRegistry
from multimind.router.multi_modal_router import MultiModalRequest
from multimind.models.embeddings import EmbeddingModel

class CrossModalRetrievalWorkflow(MCPWorkflowAPI):
    """Workflow for cross-modal retrieval and analysis."""
    
    def __init__(self, models: Dict[str, Any], integrations: Dict[str, Any]):
        super().__init__(
            name="Cross-Modal Retrieval",
            description="Retrieve and analyze content across different modalities",
            models=models,
            integrations=integrations
        )
        self.embedding_model = EmbeddingModel()
    
    async def execute(self, request: MultiModalRequest) -> Dict[str, Any]:
        """Execute the cross-modal retrieval workflow."""
        results = {}
        
        # 1. Generate embeddings for each modality
        embeddings = {}
        for modality, content in request.content.items():
            if modality == "text":
                embeddings[modality] = await self.embedding_model.get_embeddings(content)
            elif modality == "image":
                # Extract text from image first
                image_text = await self.models["gpt4v"].process_image(content)
                embeddings[modality] = await self.embedding_model.get_embeddings(image_text["text"])
            elif modality == "audio":
                # Transcribe audio first
                audio_text = await self.models["whisper"].process_audio(content)
                embeddings[modality] = await self.embedding_model.get_embeddings(audio_text["text"])
        
        # 2. Calculate similarity between modalities
        similarities = {}
        modalities = list(embeddings.keys())
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                similarity = await self._calculate_similarity(
                    embeddings[mod1],
                    embeddings[mod2]
                )
                similarities[f"{mod1}-{mod2}"] = similarity
        
        # 3. Generate cross-modal analysis
        analysis_prompt = "Analyze the relationships between these modalities:\n"
        for mod1, mod2 in similarities:
            analysis_prompt += f"\n{mod1} and {mod2} similarity: {similarities[f'{mod1}-{mod2}']:.2f}"
        
        analysis = await self.models["gpt4"].generate(analysis_prompt)
        
        return {
            "embeddings": embeddings,
            "similarities": similarities,
            "analysis": analysis["text"]
        }
    
    async def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Convert to numpy arrays and reshape
        emb1 = np.array(emb1).reshape(1, -1)
        emb2 = np.array(emb2).reshape(1, -1)
        
        # Calculate cosine similarity
        return float(cosine_similarity(emb1, emb2)[0][0])

async def main():
    """Run the cross-modal retrieval example."""
    from multimind.router.multi_modal_router import MultiModalRouter
    
    # Initialize router
    router = MultiModalRouter()
    
    # Load sample data
    data_dir = Path(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    
    # Load and encode image
    image_path = data_dir / "sample_image.jpg"
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Load and encode audio
    audio_path = data_dir / "sample_audio.mp3"
    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()
    
    # Create request
    request = MultiModalRequest(
        content={
            "image": image_data,
            "audio": audio_data,
            "text": "This is a sample text for cross-modal analysis."
        },
        modalities=["image", "audio", "text"]
    )
    
    # Get workflow
    workflow = WorkflowRegistry.get_workflow("Cross-Modal Retrieval")
    
    # Execute workflow
    result = await workflow.execute(request)
    
    print("Cross-modal retrieval completed!")
    print("\nResults:")
    print("\nSimilarities between modalities:")
    for pair, similarity in result["similarities"].items():
        print(f"{pair}: {similarity:.2f}")
    
    print("\nAnalysis:")
    print(result["analysis"])

if __name__ == "__main__":
    asyncio.run(main()) 