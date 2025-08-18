"""
Advanced examples of custom vector stores and RAG integration.
"""

import asyncio
import os
import numpy as np
from typing import Dict, List, Any, Optional
from multimind import Router, TaskType, TaskConfig, RoutingStrategy, VectorStore, VectorStoreConfig
from multimind.core.provider import ProviderConfig
from multimind.rag import RAGConfig
from multimind.providers.openai import OpenAIProvider

class HierarchicalVectorStore(VectorStore):
    """Vector store with hierarchical data support."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize with hierarchical support."""
        self.config = config
        self.dimension = config.dimension
        self.store: Dict[str, Dict[str, Any]] = {}
        self.hierarchy: Dict[str, List[str]] = {}  # parent_id -> child_ids
    
    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        **kwargs
    ) -> List[str]:
        """Add vectors with hierarchical relationships."""
        vector_ids = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            vector_id = f"vec_{len(self.store)}"
            parent_id = meta.get("parent_id")
            
            if parent_id:
                if parent_id not in self.hierarchy:
                    self.hierarchy[parent_id] = []
                self.hierarchy[parent_id].append(vector_id)
            
            self.store[vector_id] = {
                "vector": vector,
                "metadata": meta
            }
            vector_ids.append(vector_id)
        
        return vector_ids
    
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        include_children: bool = False,
        parent_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search with hierarchical filtering."""
        results = []
        
        # Get vector IDs to search
        vector_ids = set()
        if parent_id:
            vector_ids.add(parent_id)
            if include_children:
                vector_ids.update(self._get_all_children(parent_id))
        else:
            vector_ids = set(self.store.keys())
        
        # Calculate similarities
        for vector_id in vector_ids:
            data = self.store[vector_id]
            vector = data["vector"]
            
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            
            results.append({
                "vector_id": vector_id,
                "distance": float(1 - similarity),
                "metadata": data["metadata"]
            })
        
        # Sort by distance and return top k
        results.sort(key=lambda x: x["distance"])
        return results[:k]
    
    def _get_all_children(self, parent_id: str) -> set:
        """Get all children recursively."""
        children = set()
        if parent_id in self.hierarchy:
            for child_id in self.hierarchy[parent_id]:
                children.add(child_id)
                children.update(self._get_all_children(child_id))
        return children
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        **kwargs
    ) -> bool:
        """Delete vectors and update hierarchy."""
        for vector_id in vector_ids:
            if vector_id in self.store:
                # Remove from parent's children
                for parent_id, children in self.hierarchy.items():
                    if vector_id in children:
                        children.remove(vector_id)
                        if not children:
                            del self.hierarchy[parent_id]
                
                # Remove children
                if vector_id in self.hierarchy:
                    del self.hierarchy[vector_id]
                
                del self.store[vector_id]
        
        return True
    
    async def get_vector(
        self,
        vector_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get vector by ID."""
        return self.store.get(vector_id)
    
    async def update_metadata(
        self,
        vector_id: str,
        metadata: Dict[str, Any],
        **kwargs
    ) -> bool:
        """Update metadata and hierarchy."""
        if vector_id not in self.store:
            return False
        
        old_parent = self.store[vector_id]["metadata"].get("parent_id")
        new_parent = metadata.get("parent_id")
        
        # Update hierarchy if parent changed
        if old_parent != new_parent:
            if old_parent and old_parent in self.hierarchy:
                self.hierarchy[old_parent].remove(vector_id)
                if not self.hierarchy[old_parent]:
                    del self.hierarchy[old_parent]
            
            if new_parent:
                if new_parent not in self.hierarchy:
                    self.hierarchy[new_parent] = []
                self.hierarchy[new_parent].append(vector_id)
        
        self.store[vector_id]["metadata"].update(metadata)
        return True

class MultiModalVectorStore(VectorStore):
    """Vector store for multi-modal content."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize with multi-modal support."""
        self.config = config
        self.dimension = config.dimension
        self.store: Dict[str, Dict[str, Any]] = {}
        self.modal_index: Dict[str, List[str]] = {}  # modality -> vector_ids
    
    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        **kwargs
    ) -> List[str]:
        """Add vectors with modality information."""
        vector_ids = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            vector_id = f"vec_{len(self.store)}"
            modality = meta.get("modality")
            
            if modality:
                if modality not in self.modal_index:
                    self.modal_index[modality] = []
                self.modal_index[modality].append(vector_id)
            
            self.store[vector_id] = {
                "vector": vector,
                "metadata": meta
            }
            vector_ids.append(vector_id)
        
        return vector_ids
    
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        modalities: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search with modality filtering."""
        results = []
        
        # Get vector IDs to search
        vector_ids = set()
        if modalities:
            for modality in modalities:
                if modality in self.modal_index:
                    vector_ids.update(self.modal_index[modality])
        else:
            vector_ids = set(self.store.keys())
        
        # Calculate similarities
        for vector_id in vector_ids:
            data = self.store[vector_id]
            vector = data["vector"]
            
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            
            results.append({
                "vector_id": vector_id,
                "distance": float(1 - similarity),
                "metadata": data["metadata"]
            })
        
        # Sort by distance and return top k
        results.sort(key=lambda x: x["distance"])
        return results[:k]
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        **kwargs
    ) -> bool:
        """Delete vectors and update modal index."""
        for vector_id in vector_ids:
            if vector_id in self.store:
                metadata = self.store[vector_id]["metadata"]
                modality = metadata.get("modality")
                
                if modality and modality in self.modal_index:
                    self.modal_index[modality].remove(vector_id)
                    if not self.modal_index[modality]:
                        del self.modal_index[modality]
                
                del self.store[vector_id]
        
        return True
    
    async def get_vector(
        self,
        vector_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get vector by ID."""
        return self.store.get(vector_id)
    
    async def update_metadata(
        self,
        vector_id: str,
        metadata: Dict[str, Any],
        **kwargs
    ) -> bool:
        """Update metadata and modal index."""
        if vector_id not in self.store:
            return False
        
        old_modality = self.store[vector_id]["metadata"].get("modality")
        new_modality = metadata.get("modality")
        
        # Update modal index if modality changed
        if old_modality != new_modality:
            if old_modality and old_modality in self.modal_index:
                self.modal_index[old_modality].remove(vector_id)
                if not self.modal_index[old_modality]:
                    del self.modal_index[old_modality]
            
            if new_modality:
                if new_modality not in self.modal_index:
                    self.modal_index[new_modality] = []
                self.modal_index[new_modality].append(vector_id)
        
        self.store[vector_id]["metadata"].update(metadata)
        return True

async def main():
    # Initialize provider
    openai_config = ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1"
    )
    openai_provider = OpenAIProvider(openai_config)
    
    # Initialize router
    router = Router()
    router.register_provider("openai", openai_provider)
    
    # Example 1: Hierarchical Vector Store
    print("\nExample 1: Hierarchical Vector Store")
    
    hierarchy_config = VectorStoreConfig(
        dimension=1536
    )
    
    hierarchy_store = HierarchicalVectorStore(hierarchy_config)
    
    # Add hierarchical content
    vectors = [np.random.rand(1536) for _ in range(4)]
    metadata = [
        {
            "text": "Main article about AI",
            "type": "article"
        },
        {
            "text": "Section about machine learning",
            "type": "section",
            "parent_id": "vec_0"
        },
        {
            "text": "Subsection about neural networks",
            "type": "subsection",
            "parent_id": "vec_1"
        },
        {
            "text": "Another section about deep learning",
            "type": "section",
            "parent_id": "vec_0"
        }
    ]
    
    vector_ids = await hierarchy_store.add_vectors(vectors, metadata)
    
    # Search with hierarchy
    query_vector = np.random.rand(1536)
    results = await hierarchy_store.search(
        query_vector,
        parent_id="vec_0",
        include_children=True
    )
    
    print("\nSearch results with hierarchy:")
    for result in results:
        print(f"- {result['metadata']['text']} ({result['metadata']['type']})")
    
    # Example 2: Multi-Modal Vector Store
    print("\nExample 2: Multi-Modal Vector Store")
    
    multimodal_config = VectorStoreConfig(
        dimension=1536
    )
    
    multimodal_store = MultiModalVectorStore(multimodal_config)
    
    # Add multi-modal content
    vectors = [np.random.rand(1536) for _ in range(3)]
    metadata = [
        {
            "text": "Image description: A beautiful sunset over mountains",
            "modality": "image",
            "url": "sunset.jpg"
        },
        {
            "text": "Audio transcript: Interview about AI ethics",
            "modality": "audio",
            "url": "interview.mp3"
        },
        {
            "text": "Video description: Tutorial on machine learning",
            "modality": "video",
            "url": "tutorial.mp4"
        }
    ]
    
    vector_ids = await multimodal_store.add_vectors(vectors, metadata)
    
    # Search with modality filter
    query_vector = np.random.rand(1536)
    results = await multimodal_store.search(
        query_vector,
        modalities=["image", "video"]
    )
    
    print("\nSearch results with modality filter:")
    for result in results:
        print(f"- {result['metadata']['text']} ({result['metadata']['modality']})")
    
    # Example 3: RAG Integration with Custom Stores
    print("\nExample 3: RAG Integration with Custom Stores")
    
    # Initialize RAG pipeline with hierarchical store
    rag_config = RAGConfig(
        vector_store=hierarchy_store,
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002",
        generation_provider="openai",
        generation_model="gpt-4",
        chunk_size=1000,
        chunk_overlap=200,
        max_results=5
    )
    
    # Create RAG pipeline
    pipeline = RAGPipeline(router, rag_config)
    
    # Example documents with hierarchy
    documents = [
        """
        Artificial Intelligence (AI) is transforming industries.
        
        Machine Learning:
        - Supervised learning
        - Unsupervised learning
        - Reinforcement learning
        
        Deep Learning:
        - Neural networks
        - Convolutional networks
        - Recurrent networks
        """
    ]
    
    # Execute RAG pipeline with hierarchical search
    result = await (
        pipeline
        .load_documents(documents)
        .query("What are the main types of machine learning?")
        .generate()
        .execute()
    )
    
    print("\nRAG results with hierarchical store:")
    print(f"Answer: {result.answer}")
    print("\nSources:")
    for source in result.sources:
        print(f"- {source['text'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main()) 