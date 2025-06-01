"""
Vector store module supporting multiple vector database backends.
Provides a unified interface for vector operations across different databases.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import pickle

# Vector Database Imports
import faiss
import chromadb
from chromadb.config import Settings
import weaviate
from weaviate.util import generate_uuid
import qdrant_client
from qdrant_client.http import models as qdrant_models
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import pinecone
from elasticsearch import AsyncElasticsearch
import redis
from redis.commands.search.field import VectorField, TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import psycopg2
from psycopg2.extras import Json
import hnswlib

@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    store_type: str  # Type of vector store to use
    dimension: int  # Vector dimension
    index_params: Dict[str, Any]  # Index-specific parameters
    connection_params: Dict[str, Any]  # Connection parameters
    search_params: Dict[str, Any]  # Search parameters
    custom_params: Dict[str, Any]  # Custom parameters

@dataclass
class SearchResult:
    """Represents a search result."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    document: Dict[str, Any]
    score: float

class VectorStoreType(Enum):
    """Types of vector stores supported."""
    FAISS = "faiss"
    CHROMA = "chroma"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    PINECONE = "pinecone"
    ELASTICSEARCH = "elasticsearch"
    REDIS = "redis"
    POSTGRES = "postgres"

@runtime_checkable
class VectorStoreBackend(Protocol):
    """Protocol defining vector store backend interface."""
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to the store."""
        pass

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from the store."""
        pass

    async def clear(self) -> None:
        """Clear all vectors from the store."""
        pass

    async def persist(self, path: str) -> None:
        """Persist the vector store to disk."""
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "VectorStoreBackend":
        """Load vector store from disk."""
        pass

class FAISSBackend(VectorStoreBackend):
    """FAISS vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = None
        self.metadata = {}
        self.documents = {}
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize FAISS index."""
        index_params = self.config.index_params
        self.index = faiss.IndexFlatL2(self.config.dimension)
        
        if "nlist" in index_params:
            self.index = faiss.IndexIVFFlat(
                self.index,
                self.config.dimension,
                index_params["nlist"]
            )
        
        if "nprobe" in index_params:
            self.index.nprobe = index_params["nprobe"]

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to FAISS index."""
        if not self.index:
            await self.initialize()
        
        vectors_array = np.array(vectors).astype("float32")
        self.index.add(vectors_array)
        
        start_id = len(self.metadata)
        for i, (metadata, doc) in enumerate(zip(metadatas, documents)):
            id = ids[i] if ids else f"vec_{start_id + i}"
            self.metadata[id] = metadata
            self.documents[id] = doc

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search FAISS index."""
        if not self.index:
            return []
        
        query_array = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                id = f"vec_{idx}"
                results.append(SearchResult(
                    id=id,
                    vector=query_vector,  # FAISS doesn't store vectors
                    metadata=self.metadata[id],
                    document=self.documents[id],
                    score=float(1 / (1 + distance))  # Convert distance to similarity
                ))
        
        return results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from FAISS index."""
        # FAISS doesn't support deletion, so we need to rebuild the index
        if not self.index:
            return
        
        # Create new index
        new_index = faiss.IndexFlatL2(self.config.dimension)
        new_metadata = {}
        new_documents = {}
        
        # Rebuild with remaining vectors
        for i, (id, metadata) in enumerate(self.metadata.items()):
            if id not in ids:
                new_metadata[id] = metadata
                new_documents[id] = self.documents[id]
        
        # Update index
        self.index = new_index
        self.metadata = new_metadata
        self.documents = new_documents

    async def clear(self) -> None:
        """Clear FAISS index."""
        self.index = None
        self.metadata = {}
        self.documents = {}

    async def persist(self, path: str) -> None:
        """Persist FAISS index to disk."""
        if not self.index:
            return
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save metadata and documents
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "FAISSBackend":
        """Load FAISS index from disk."""
        path = Path(path)
        
        backend = cls(config)
        
        # Load index
        if (path / "index.faiss").exists():
            backend.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load metadata and documents
        if (path / "metadata.pkl").exists():
            with open(path / "metadata.pkl", "rb") as f:
                backend.metadata = pickle.load(f)
        if (path / "documents.pkl").exists():
            with open(path / "documents.pkl", "rb") as f:
                backend.documents = pickle.load(f)
        
        return backend

class ChromaBackend(VectorStoreBackend):
    """Chroma vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Chroma client and collection."""
        settings = Settings(
            **self.config.connection_params.get("settings", {})
        )
        self.client = chromadb.Client(settings)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.connection_params.get("collection_name", "default"),
            metadata={"dimension": self.config.dimension}
        )

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Chroma collection."""
        if not self.collection:
            await self.initialize()
        
        # Prepare documents and metadatas
        docs = [doc["content"] for doc in documents]
        if not ids:
            ids = [f"doc_{i}" for i in range(len(docs))]
        
        # Add to collection
        self.collection.add(
            embeddings=vectors,
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Chroma collection."""
        if not self.collection:
            return []
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=filter_criteria
        )
        
        # Convert to SearchResult format
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append(SearchResult(
                id=results["ids"][0][i],
                vector=query_vector,  # Chroma doesn't return vectors
                metadata=results["metadatas"][0][i],
                document={"content": results["documents"][0][i]},
                score=results["distances"][0][i] if "distances" in results else 1.0
            ))
        
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Chroma collection."""
        if not self.collection:
            return
        
        self.collection.delete(ids=ids)

    async def clear(self) -> None:
        """Clear Chroma collection."""
        if not self.collection:
            return
        
        self.collection.delete(where={})

    async def persist(self, path: str) -> None:
        """Persist Chroma collection to disk."""
        if not self.client:
            return
        
        # Chroma persists automatically to the configured directory
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "ChromaBackend":
        """Load Chroma collection from disk."""
        backend = cls(config)
        await backend.initialize()
        return backend

class WeaviateBackend(VectorStoreBackend):
    """Weaviate vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.class_name = "Document"
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Weaviate client and schema."""
        self.client = weaviate.Client(
            **self.config.connection_params
        )
        
        # Create schema if it doesn't exist
        if not self.client.schema.contains(self.class_name):
            class_obj = {
                "class": self.class_name,
                "vectorizer": "none",
                "vectorIndexConfig": {
                    "distance": "cosine"
                },
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"]
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"]
                    }
                ]
            }
            self.client.schema.create_class(class_obj)

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Weaviate."""
        if not self.client:
            await self.initialize()
        
        # Prepare batch
        with self.client.batch as batch:
            batch.batch_size = 100
            for i, (vector, metadata, doc) in enumerate(zip(vectors, metadatas, documents)):
                id = ids[i] if ids else generate_uuid()
                data_object = {
                    "content": doc["content"],
                    "metadata": json.dumps(metadata)
                }
                batch.add_data_object(
                    data_object=data_object,
                    class_name=self.class_name,
                    uuid=id,
                    vector=vector
                )

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Weaviate."""
        if not self.client:
            return []
        
        # Prepare query
        query = (
            self.client.query
            .get(self.class_name, ["content", "metadata"])
            .with_near_vector({
                "vector": query_vector
            })
            .with_limit(k)
        )
        
        # Add filters if provided
        if filter_criteria:
            query = query.with_where(filter_criteria)
        
        # Execute query
        results = query.do()
        
        # Convert to SearchResult format
        search_results = []
        for result in results["data"]["Get"][self.class_name]:
            search_results.append(SearchResult(
                id=result["_additional"]["id"],
                vector=query_vector,  # Weaviate doesn't return vectors
                metadata=json.loads(result["metadata"]),
                document={"content": result["content"]},
                score=result["_additional"]["certainty"]
            ))
        
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Weaviate."""
        if not self.client:
            return
        
        for id in ids:
            self.client.data_object.delete(
                id,
                class_name=self.class_name
            )

    async def clear(self) -> None:
        """Clear Weaviate collection."""
        if not self.client:
            return
        
        self.client.schema.delete_class(self.class_name)
        await self.initialize()

    async def persist(self, path: str) -> None:
        """Persist Weaviate to disk."""
        # Weaviate handles persistence internally
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "WeaviateBackend":
        """Load Weaviate from disk."""
        backend = cls(config)
        await backend.initialize()
        return backend

class QdrantBackend(VectorStoreBackend):
    """Qdrant vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.collection_name = "documents"
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Qdrant client and collection."""
        self.client = qdrant_client.QdrantClient(
            **self.config.connection_params
        )
        
        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.config.dimension,
                    distance=qdrant_models.Distance.COSINE
                )
            )

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Qdrant."""
        if not self.client:
            await self.initialize()
        
        # Prepare points
        points = []
        for i, (vector, metadata, doc) in enumerate(zip(vectors, metadatas, documents)):
            id = ids[i] if ids else f"doc_{i}"
            points.append(qdrant_models.PointStruct(
                id=id,
                vector=vector,
                payload={
                    "content": doc["content"],
                    "metadata": metadata
                }
            ))
        
        # Upload points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Qdrant."""
        if not self.client:
            return []
        
        # Prepare search
        search_params = qdrant_models.SearchParams(
            **self.config.search_params
        )
        
        # Execute search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=filter_criteria,
            search_params=search_params
        )
        
        # Convert to SearchResult format
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                id=result.id,
                vector=query_vector,  # Qdrant doesn't return vectors
                metadata=result.payload["metadata"],
                document={"content": result.payload["content"]},
                score=result.score
            ))
        
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Qdrant."""
        if not self.client:
            return
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.PointIdsList(
                points=ids
            )
        )

    async def clear(self) -> None:
        """Clear Qdrant collection."""
        if not self.client:
            return
        
        self.client.delete_collection(self.collection_name)
        await self.initialize()

    async def persist(self, path: str) -> None:
        """Persist Qdrant to disk."""
        # Qdrant handles persistence internally
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "QdrantBackend":
        """Load Qdrant from disk."""
        backend = cls(config)
        await backend.initialize()
        return backend

class MilvusBackend(VectorStoreBackend):
    """Milvus vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.collection = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Milvus connection and collection."""
        # Connect to Milvus
        connections.connect(
            **self.config.connection_params
        )
        
        # Create collection if it doesn't exist
        collection_name = self.config.connection_params.get("collection_name", "documents")
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields=fields)
            self.collection = Collection(
                name=collection_name,
                schema=schema,
                using="default"
            )
            
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(
                field_name="vector",
                index_params=index_params
            )
        else:
            self.collection = Collection(collection_name)

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Milvus."""
        if not self.collection:
            await self.initialize()
        
        # Prepare data
        if not ids:
            ids = [f"doc_{i}" for i in range(len(vectors))]
        
        entities = [
            ids,
            vectors,
            [doc["content"] for doc in documents],
            [json.dumps(metadata) for metadata in metadatas]
        ]
        
        # Insert data
        self.collection.insert(entities)
        self.collection.flush()

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Milvus."""
        if not self.collection:
            return []
        
        # Load collection
        self.collection.load()
        
        # Prepare search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # Execute search
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=k,
            expr=filter_criteria
        )
        
        # Convert to SearchResult format
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(SearchResult(
                    id=hit.id,
                    vector=query_vector,  # Milvus doesn't return vectors
                    metadata=json.loads(hit.entity.get("metadata")),
                    document={"content": hit.entity.get("content")},
                    score=hit.score
                ))
        
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Milvus."""
        if not self.collection:
            return
        
        expr = f'id in {json.dumps(ids)}'
        self.collection.delete(expr)

    async def clear(self) -> None:
        """Clear Milvus collection."""
        if not self.collection:
            return
        
        self.collection.drop()
        await self.initialize()

    async def persist(self, path: str) -> None:
        """Persist Milvus to disk."""
        # Milvus handles persistence internally
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "MilvusBackend":
        """Load Milvus from disk."""
        backend = cls(config)
        await backend.initialize()
        return backend

class PineconeBackend(VectorStoreBackend):
    """Pinecone vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Pinecone client and index."""
        pinecone.init(
            **self.config.connection_params
        )
        
        # Get or create index
        index_name = self.config.connection_params.get("index_name", "documents")
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=self.config.dimension,
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Pinecone."""
        if not self.index:
            await self.initialize()
        
        # Prepare vectors
        if not ids:
            ids = [f"doc_{i}" for i in range(len(vectors))]
        
        vectors_to_upsert = []
        for i, (vector, metadata, doc) in enumerate(zip(vectors, metadatas, documents)):
            vectors_to_upsert.append({
                "id": ids[i],
                "values": vector,
                "metadata": {
                    **metadata,
                    "content": doc["content"]
                }
            })
        
        # Upsert vectors
        self.index.upsert(vectors=vectors_to_upsert)

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Pinecone."""
        if not self.index:
            return []
        
        # Execute search
        results = self.index.query(
            vector=query_vector,
            top_k=k,
            filter=filter_criteria,
            include_metadata=True
        )
        
        # Convert to SearchResult format
        search_results = []
        for match in results.matches:
            search_results.append(SearchResult(
                id=match.id,
                vector=query_vector,  # Pinecone doesn't return vectors
                metadata={k: v for k, v in match.metadata.items() if k != "content"},
                document={"content": match.metadata["content"]},
                score=match.score
            ))
        
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Pinecone."""
        if not self.index:
            return
        
        self.index.delete(ids=ids)

    async def clear(self) -> None:
        """Clear Pinecone index."""
        if not self.index:
            return
        
        self.index.delete(delete_all=True)

    async def persist(self, path: str) -> None:
        """Persist Pinecone to disk."""
        # Pinecone handles persistence internally
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "PineconeBackend":
        """Load Pinecone from disk."""
        backend = cls(config)
        await backend.initialize()
        return backend

class ElasticsearchBackend(VectorStoreBackend):
    """Elasticsearch vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.index_name = "documents"
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Elasticsearch client and index."""
        self.client = AsyncElasticsearch(
            **self.config.connection_params
        )
        
        # Create index if it doesn't exist
        if not await self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "dense_vector",
                            "dims": self.config.dimension,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "content": {
                            "type": "text"
                        },
                        "metadata": {
                            "type": "object"
                        }
                    }
                }
            }
            await self.client.indices.create(
                index=self.index_name,
                body=mapping
            )

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Elasticsearch."""
        if not self.client:
            await self.initialize()
        
        # Prepare bulk request
        actions = []
        for i, (vector, metadata, doc) in enumerate(zip(vectors, metadatas, documents)):
            id = ids[i] if ids else f"doc_{i}"
            action = {
                "_index": self.index_name,
                "_id": id,
                "_source": {
                    "vector": vector,
                    "content": doc["content"],
                    "metadata": metadata
                }
            }
            actions.append(action)
        
        # Bulk index
        await self.client.bulk(operations=actions)

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Elasticsearch."""
        if not self.client:
            return []
        
        # Prepare query
        query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "size": k
        }
        
        # Add filters if provided
        if filter_criteria:
            query["query"]["script_score"]["query"] = {
                "bool": {
                    "filter": [
                        {"term": {k: v}} for k, v in filter_criteria.items()
                    ]
                }
            }
        
        # Execute search
        response = await self.client.search(
            index=self.index_name,
            body=query
        )
        
        # Convert to SearchResult format
        search_results = []
        for hit in response["hits"]["hits"]:
            search_results.append(SearchResult(
                id=hit["_id"],
                vector=query_vector,  # Elasticsearch doesn't return vectors
                metadata=hit["_source"]["metadata"],
                document={"content": hit["_source"]["content"]},
                score=hit["_score"]
            ))
        
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Elasticsearch."""
        if not self.client:
            return
        
        for id in ids:
            await self.client.delete(
                index=self.index_name,
                id=id
            )

    async def clear(self) -> None:
        """Clear Elasticsearch index."""
        if not self.client:
            return
        
        await self.client.indices.delete(index=self.index_name)
        await self.initialize()

    async def persist(self, path: str) -> None:
        """Persist Elasticsearch to disk."""
        # Elasticsearch handles persistence internally
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "ElasticsearchBackend":
        """Load Elasticsearch from disk."""
        backend = cls(config)
        await backend.initialize()
        return backend

class RedisBackend(VectorStoreBackend):
    """Redis vector store backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.index_name = "documents"
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Redis client and index."""
        self.client = redis.Redis(
            **self.config.connection_params
        )
        
        # Create index if it doesn't exist
        try:
            self.client.ft(self.index_name).info()
        except redis.ResponseError:
            # Define schema
            schema = (
                VectorField("vector", "FLOAT32", self.config.dimension, "COSINE"),
                TextField("content"),
                TagField("metadata")
            )
            
            # Create index
            self.client.ft(self.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(
                    prefix=["doc:"],
                    index_type=IndexType.HASH
                )
            )

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Redis."""
        if not self.client:
            await self.initialize()
        
        # Prepare pipeline
        pipe = self.client.pipeline()
        
        for i, (vector, metadata, doc) in enumerate(zip(vectors, metadatas, documents)):
            id = ids[i] if ids else f"doc_{i}"
            key = f"doc:{id}"
            
            # Store document
            pipe.hset(
                key,
                mapping={
                    "vector": np.array(vector).astype(np.float32).tobytes(),
                    "content": doc["content"],
                    "metadata": json.dumps(metadata)
                }
            )
        
        # Execute pipeline
        pipe.execute()

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Redis."""
        if not self.client:
            return []
        
        # Prepare query
        query = f"*=>[KNN {k} @vector $BLOB AS score]"
        params = {
            "BLOB": np.array(query_vector).astype(np.float32).tobytes()
        }
        
        # Add filters if provided
        if filter_criteria:
            filter_query = " ".join(
                f"@metadata:{json.dumps(v)}" for v in filter_criteria.values()
            )
            query = f"({filter_query})=>[KNN {k} @vector $BLOB AS score]"
        
        # Execute search
        results = self.client.ft(self.index_name).search(
            query,
            query_params=params,
            sort_by="score",
            sort_asc=False
        )
        
        # Convert to SearchResult format
        search_results = []
        for doc in results.docs:
            search_results.append(SearchResult(
                id=doc.id,
                vector=query_vector,  # Redis doesn't return vectors
                metadata=json.loads(doc.metadata),
                document={"content": doc.content},
                score=float(doc.score)
            ))
        
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Redis."""
        if not self.client:
            return
        
        # Delete documents
        pipe = self.client.pipeline()
        for id in ids:
            pipe.delete(f"doc:{id}")
        pipe.execute()

    async def clear(self) -> None:
        """Clear Redis index."""
        if not self.client:
            return
        
        self.client.ft(self.index_name).dropindex()
        await self.initialize()

    async def persist(self, path: str) -> None:
        """Persist Redis to disk."""
        # Redis handles persistence internally
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "RedisBackend":
        """Load Redis from disk."""
        backend = cls(config)
        await backend.initialize()
        return backend

class PostgresBackend(VectorStoreBackend):
    """Postgres vector store backend using pgvector."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.conn = None
        self.table_name = "documents"
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Postgres connection and table."""
        self.conn = psycopg2.connect(
            **self.config.connection_params
        )
        
        # Create extension if it doesn't exist
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create table if it doesn't exist
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    vector vector({self.config.dimension}),
                    content TEXT,
                    metadata JSONB
                )
            """)
            
            # Create index
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_vector_idx 
                ON {self.table_name} 
                USING ivfflat (vector vector_cosine_ops)
                WITH (lists = 100)
            """)
        
        self.conn.commit()

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to Postgres."""
        if not self.conn:
            await self.initialize()
        
        # Prepare data
        if not ids:
            ids = [f"doc_{i}" for i in range(len(vectors))]
        
        # Insert data
        with self.conn.cursor() as cur:
            for i, (id, vector, metadata, doc) in enumerate(zip(ids, vectors, metadatas, documents)):
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, vector, content, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET vector = EXCLUDED.vector,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata
                    """,
                    (id, vector, doc["content"], Json(metadata))
                )
        
        self.conn.commit()

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Postgres."""
        if not self.conn:
            return []
        
        # Prepare query
        query = f"""
            SELECT id, content, metadata, 1 - (vector <=> %s) as score
            FROM {self.table_name}
        """
        params = [query_vector]
        
        # Add filters if provided
        if filter_criteria:
            conditions = []
            for key, value in filter_criteria.items():
                conditions.append(f"metadata->>%s = %s")
                params.extend([key, json.dumps(value)])
            query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY vector <=> %s LIMIT %s"
        params.extend([query_vector, k])
        
        # Execute query
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        
        # Convert to SearchResult format
        search_results = []
        for id, content, metadata, score in results:
            search_results.append(SearchResult(
                id=id,
                vector=query_vector,  # Postgres doesn't return vectors
                metadata=metadata,
                document={"content": content},
                score=float(score)
            ))
        
        return search_results

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Postgres."""
        if not self.conn:
            return
        
        with self.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",
                (ids,)
            )
        self.conn.commit()

    async def clear(self) -> None:
        """Clear Postgres table."""
        if not self.conn:
            return
        
        with self.conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table_name}")
        self.conn.commit()

    async def persist(self, path: str) -> None:
        """Persist Postgres to disk."""
        # Postgres handles persistence internally
        pass

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "PostgresBackend":
        """Load Postgres from disk."""
        backend = cls(config)
        await backend.initialize()
        return backend

class VectorStore:
    """Unified vector store interface."""
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize vector store.
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self.backend = self._get_backend()
        self.logger = logging.getLogger(__name__)

    def _get_backend(self) -> VectorStoreBackend:
        """Get appropriate vector store backend."""
        store_type = VectorStoreType(self.config.store_type)
        
        if store_type == VectorStoreType.FAISS:
            return FAISSBackend(self.config)
        elif store_type == VectorStoreType.CHROMA:
            return ChromaBackend(self.config)
        elif store_type == VectorStoreType.WEAVIATE:
            return WeaviateBackend(self.config)
        elif store_type == VectorStoreType.QDRANT:
            return QdrantBackend(self.config)
        elif store_type == VectorStoreType.MILVUS:
            return MilvusBackend(self.config)
        elif store_type == VectorStoreType.PINECONE:
            return PineconeBackend(self.config)
        elif store_type == VectorStoreType.ELASTICSEARCH:
            return ElasticsearchBackend(self.config)
        elif store_type == VectorStoreType.REDIS:
            return RedisBackend(self.config)
        elif store_type == VectorStoreType.POSTGRES:
            return PostgresBackend(self.config)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")

    async def initialize(self) -> None:
        """Initialize vector store backend."""
        await self.backend.initialize()

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to store."""
        await self.backend.add_vectors(vectors, metadatas, documents, ids)

    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search vectors in store."""
        return await self.backend.search(query_vector, k, filter_criteria)

    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from store."""
        await self.backend.delete_vectors(ids)

    async def clear(self) -> None:
        """Clear vector store."""
        await self.backend.clear()

    async def persist(self, path: str) -> None:
        """Persist vector store to disk."""
        await self.backend.persist(path)

    @classmethod
    async def load(cls, path: str, config: VectorStoreConfig) -> "VectorStore":
        """Load vector store from disk."""
        store = cls(config)
        store.backend = await store.backend.load(path, config)
        return store 