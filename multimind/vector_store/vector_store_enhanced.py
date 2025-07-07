"""
Enhanced vector store module with advanced features:
- Plugin registry for easy backend switching
- Live index updates
- Hybrid search (BM25 + vector)
- Scoring fusion
- Advanced persistence
- Metadata indexing
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable, Type, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
from abc import ABC, abstractmethod
import aiofiles
import boto3
from botocore.exceptions import ClientError
import rank_bm25
from sklearn.preprocessing import normalize
import sqlite3
import yaml
import requests

from . import (
    VectorStore, VectorStoreConfig, VectorStoreType, VectorStoreBackend,
    SearchResult, FAISSBackend, ChromaBackend, WeaviateBackend, QdrantBackend,
    MilvusBackend, PineconeBackend, ElasticsearchBackend, RedisBackend,
    PostgresBackend
)

@dataclass
class EnhancedVectorStoreConfig(VectorStoreConfig):
    """Enhanced configuration for vector store."""
    # Plugin registry settings
    plugin_dir: Optional[str] = None  # Directory for custom plugins
    auto_discover_plugins: bool = True  # Auto-discover plugins in plugin_dir
    
    # Live update settings
    enable_live_updates: bool = False  # Enable live index updates
    update_batch_size: int = 100  # Batch size for updates
    update_interval: float = 1.0  # Update interval in seconds
    
    # Hybrid search settings
    enable_hybrid_search: bool = False  # Enable hybrid search
    bm25_weight: float = 0.3  # Weight for BM25 scores
    vector_weight: float = 0.7  # Weight for vector similarity scores
    
    # Scoring fusion settings
    enable_scoring_fusion: bool = False  # Enable scoring fusion
    fusion_method: str = "weighted_sum"  # Fusion method (weighted_sum, reciprocal_rank, etc.)
    fusion_weights: Dict[str, float] = None  # Weights for different scoring methods
    
    # Persistence settings
    persistence_type: str = "local"  # Type of persistence (local, s3, etc.)
    persistence_config: Dict[str, Any] = None  # Persistence configuration
    
    # Metadata indexing settings
    enable_metadata_indexing: bool = False  # Enable metadata indexing
    indexed_metadata_fields: List[str] = None  # Fields to index in metadata
    metadata_index_type: str = "btree"  # Type of metadata index

@dataclass
class HybridSearchResult(SearchResult):
    """Enhanced search result with hybrid scoring."""
    bm25_score: float
    vector_score: float
    fusion_score: float
    metadata_scores: Dict[str, float]

class PluginRegistry:
    """
    Registry for vector store plugins.
    Now supports plugin marketplace: install, uninstall, list available (remote), and activate plugins.
    Usage:
        registry = PluginRegistry()
        registry.list_marketplace_plugins()
        registry.install_plugin('my_plugin')
        registry.activate_plugin('my_plugin')
    """
    
    def __init__(self):
        self._plugins: Dict[str, Type[VectorStoreBackend]] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_plugin(
        self,
        name: str,
        plugin_class: Type[VectorStoreBackend],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a vector store plugin."""
        self._plugins[name] = plugin_class
        self._plugin_configs[name] = config or {}
        self.logger.info(f"Registered plugin: {name}")
    
    def get_plugin(self, name: str) -> Tuple[Type[VectorStoreBackend], Dict[str, Any]]:
        """Get a registered plugin and its configuration."""
        if name not in self._plugins:
            raise ValueError(f"Plugin not found: {name}")
        return self._plugins[name], self._plugin_configs[name]
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self._plugins.keys())
    
    def discover_plugins(self, plugin_dir: str) -> None:
        """Discover and register plugins from a directory."""
        plugin_dir = Path(plugin_dir)
        if not plugin_dir.exists():
            return
        
        for plugin_file in plugin_dir.glob("*.py"):
            try:
                # Import plugin module
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem, plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Register plugin if it has the required attributes
                if hasattr(module, "PLUGIN_NAME") and hasattr(module, "PluginBackend"):
                    self.register_plugin(
                        module.PLUGIN_NAME,
                        module.PluginBackend,
                        getattr(module, "PLUGIN_CONFIG", None)
                    )
            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_file}: {e}")

    def list_marketplace_plugins(self, marketplace_url: str = "https://multimind-plugins.example.com/api/plugins") -> list:
        """List available plugins from the remote marketplace (placeholder URL)."""
        try:
            resp = requests.get(marketplace_url)
            resp.raise_for_status()
            return resp.json().get('plugins', [])
        except Exception as e:
            self.logger.error(f"Failed to fetch marketplace plugins: {e}")
            return []

    def install_plugin(self, name: str, marketplace_url: str = "https://multimind-plugins.example.com/api/plugins") -> bool:
        """Install a plugin from the marketplace (downloads and registers)."""
        try:
            plugins = self.list_marketplace_plugins(marketplace_url)
            plugin_info = next((p for p in plugins if p['name'] == name), None)
            if not plugin_info:
                self.logger.error(f"Plugin {name} not found in marketplace.")
                return False
            # Download plugin file
            resp = requests.get(plugin_info['download_url'])
            resp.raise_for_status()
            plugin_dir = Path(self._plugin_configs.get('plugin_dir', './plugins'))
            plugin_dir.mkdir(exist_ok=True)
            plugin_path = plugin_dir / f"{name}.py"
            with open(plugin_path, 'wb') as f:
                f.write(resp.content)
            self.discover_plugins(str(plugin_dir))
            self.logger.info(f"Installed plugin: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to install plugin {name}: {e}")
            return False

    def uninstall_plugin(self, name: str) -> bool:
        """Uninstall a plugin by removing its file and unregistering."""
        try:
            plugin_dir = Path(self._plugin_configs.get('plugin_dir', './plugins'))
            plugin_path = plugin_dir / f"{name}.py"
            if plugin_path.exists():
                plugin_path.unlink()
            if name in self._plugins:
                del self._plugins[name]
            if name in self._plugin_configs:
                del self._plugin_configs[name]
            self.logger.info(f"Uninstalled plugin: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to uninstall plugin {name}: {e}")
            return False

    def activate_plugin(self, name: str) -> bool:
        """Activate a plugin (make it the current backend, if applicable)."""
        try:
            if name not in self._plugins:
                self.logger.error(f"Plugin {name} not registered.")
                return False
            # In a real system, this would set the active backend or strategy
            self.logger.info(f"Activated plugin: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to activate plugin {name}: {e}")
            return False

class LiveUpdateHandler:
    """Handles live updates to vector store indices."""
    
    def __init__(
        self,
        vector_store: "EnhancedVectorStore",
        batch_size: int = 100,
        update_interval: float = 1.0
    ):
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.update_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.update_task = None
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the live update handler."""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Live update handler started")
    
    async def stop(self) -> None:
        """Stop the live update handler."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Live update handler stopped")
    
    async def queue_update(
        self,
        operation: str,
        vectors: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Queue an update operation."""
        await self.update_queue.put({
            "operation": operation,
            "vectors": vectors,
            "metadatas": metadatas,
            "documents": documents,
            "ids": ids,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _update_loop(self) -> None:
        """Main update loop."""
        batch = []
        last_update = datetime.now()
        
        while self.is_running:
            try:
                # Get update from queue with timeout
                try:
                    update = await asyncio.wait_for(
                        self.update_queue.get(),
                        timeout=self.update_interval
                    )
                    batch.append(update)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if it's full or enough time has passed
                now = datetime.now()
                if (len(batch) >= self.batch_size or
                    (batch and (now - last_update).total_seconds() >= self.update_interval)):
                    await self._process_batch(batch)
                    batch = []
                    last_update = now
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of updates."""
        try:
            # Group updates by operation
            updates_by_op = {}
            for update in batch:
                op = update["operation"]
                if op not in updates_by_op:
                    updates_by_op[op] = []
                updates_by_op[op].append(update)
            
            # Process each operation type
            for op, updates in updates_by_op.items():
                if op == "add":
                    await self._process_add_batch(updates)
                elif op == "delete":
                    await self._process_delete_batch(updates)
                elif op == "update":
                    await self._process_update_batch(updates)
            
            self.logger.info(f"Processed batch of {len(batch)} updates")
        
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            # Requeue failed updates
            for update in batch:
                await self.queue_update(**update)
    
    async def _process_add_batch(self, updates: List[Dict[str, Any]]) -> None:
        """Process a batch of add operations."""
        vectors = []
        metadatas = []
        documents = []
        ids = []
        
        for update in updates:
            vectors.extend(update["vectors"])
            metadatas.extend(update["metadatas"])
            documents.extend(update["documents"])
            if update["ids"]:
                ids.extend(update["ids"])
        
        await self.vector_store.add_vectors(vectors, metadatas, documents, ids)
    
    async def _process_delete_batch(self, updates: List[Dict[str, Any]]) -> None:
        """Process a batch of delete operations."""
        ids = []
        for update in updates:
            ids.extend(update["ids"])
        
        await self.vector_store.delete_vectors(ids)
    
    async def _process_update_batch(self, updates: List[Dict[str, Any]]) -> None:
        """Process a batch of update operations."""
        # Updates are treated as delete + add
        await self._process_delete_batch(updates)
        await self._process_add_batch(updates)

class HybridSearchHandler:
    """Handles hybrid search combining BM25 and vector similarity."""
    
    def __init__(
        self,
        vector_store: "EnhancedVectorStore",
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7
    ):
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25_index = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the hybrid search handler."""
        # Create BM25 index from documents
        documents = await self.vector_store.get_all_documents()
        tokenized_docs = [doc["content"].split() for doc in documents]
        self.bm25_index = rank_bm25.BM25Okapi(tokenized_docs)
    
    async def search(
        self,
        query: str,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[HybridSearchResult]:
        """Perform hybrid search."""
        # Get vector search results
        vector_results = await self.vector_store.search(
            query_vector, k, filter_criteria
        )
        
        # Get BM25 results
        tokenized_query = query.split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Combine results
        results = []
        for result in vector_results:
            doc_idx = self.vector_store.get_document_index(result.id)
            if doc_idx is not None:
                bm25_score = float(bm25_scores[doc_idx])
                fusion_score = (
                    self.bm25_weight * bm25_score +
                    self.vector_weight * result.score
                )
                
                results.append(HybridSearchResult(
                    id=result.id,
                    vector=result.vector,
                    metadata=result.metadata,
                    document=result.document,
                    score=result.score,
                    bm25_score=bm25_score,
                    vector_score=result.score,
                    fusion_score=fusion_score,
                    metadata_scores={}
                ))
        
        # Sort by fusion score
        results.sort(key=lambda x: x.fusion_score, reverse=True)
        return results[:k]

class FusionPerformanceTracker:
    """Tracks performance/feedback for each fusion scoring method."""
    def __init__(self):
        self.metrics = {}
        # metrics: {method: {"success": int, "fail": int, "feedback": [float]}}

    def record(self, method: str, success: bool, feedback: float = None):
        if method not in self.metrics:
            self.metrics[method] = {"success": 0, "fail": 0, "feedback": []}
        if success:
            self.metrics[method]["success"] += 1
        else:
            self.metrics[method]["fail"] += 1
        if feedback is not None:
            self.metrics[method]["feedback"].append(feedback)

    def get_weight(self, method: str) -> float:
        m = self.metrics.get(method, None)
        if not m:
            return 1.0
        total = m["success"] + m["fail"]
        success_rate = m["success"] / total if total > 0 else 1.0
        avg_feedback = np.mean(m["feedback"]) if m["feedback"] else 1.0
        return success_rate * (avg_feedback + 1.0)

    def get_all_weights(self, methods: List[str]) -> Dict[str, float]:
        return {m: self.get_weight(m) for m in methods}

    def submit_feedback(self, method: str, feedback: float):
        """Submit user feedback for a fusion method (1.0=good, 0.0=bad, or any float)."""
        self.record(method, success=True, feedback=feedback)

class ScoringFusionHandler:
    """Handles fusion of multiple scoring methods."""
    
    def __init__(
        self,
        vector_store: "EnhancedVectorStore",
        fusion_method: str = "weighted_sum",
        fusion_weights: Optional[Dict[str, float]] = None
    ):
        self.vector_store = vector_store
        self.fusion_method = fusion_method
        self.fusion_weights = fusion_weights or {
            "vector": 0.6,
            "bm25": 0.2,
            "metadata": 0.2
        }
        self.logger = logging.getLogger(__name__)
        self.performance_tracker = FusionPerformanceTracker()
        # Neural fusion model (if available)
        try:
            import torch
            import torch.nn as nn
            class SimpleFusionNet(nn.Module):
                def __init__(self, n_methods):
                    super().__init__()
                    self.linear = nn.Linear(n_methods, 1)
                def forward(self, x):
                    return self.linear(x)
            class MultiLayerFusionNet(nn.Module):
                def __init__(self, n_methods, hidden_dim=16, n_layers=2):
                    super().__init__()
                    layers = [nn.Linear(n_methods, hidden_dim), nn.ReLU()]
                    for _ in range(n_layers-1):
                        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
                    layers += [nn.Linear(hidden_dim, 1)]
                    self.net = nn.Sequential(*layers)
                def forward(self, x):
                    return self.net(x)
            class AttentionFusionNet(nn.Module):
                def __init__(self, n_methods, hidden_dim=16):
                    super().__init__()
                    self.query = nn.Parameter(torch.randn(1, hidden_dim))
                    self.key = nn.Linear(n_methods, hidden_dim)
                    self.value = nn.Linear(n_methods, 1)
                def forward(self, x):
                    # x: [batch, n_methods]
                    k = self.key(x)  # [batch, hidden_dim]
                    attn = torch.softmax((k * self.query).sum(-1, keepdim=True), dim=0)
                    v = self.value(x)  # [batch, 1]
                    return attn * v
            class TransformerFusionNet(nn.Module):
                def __init__(self, n_methods, hidden_dim=16, n_heads=2, n_layers=2):
                    super().__init__()
                    encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
                    self.embedding = nn.Linear(n_methods, hidden_dim)
                    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                    self.fc = nn.Linear(hidden_dim, 1)
                def forward(self, x):
                    # x: [batch, n_methods] -> [batch, 1, n_methods]
                    x = self.embedding(x).unsqueeze(1)  # [batch, 1, hidden_dim]
                    x = self.encoder(x)  # [batch, 1, hidden_dim]
                    x = self.fc(x.squeeze(1))  # [batch, 1]
                    return x
            self.fusion_net = None
            self.fusion_net_trained = False
            self.SimpleFusionNet = SimpleFusionNet
            self.MultiLayerFusionNet = MultiLayerFusionNet
            self.AttentionFusionNet = AttentionFusionNet
            self.TransformerFusionNet = TransformerFusionNet
        except ImportError:
            self.fusion_net = None
            self.fusion_net_trained = False
            self.SimpleFusionNet = None
            self.MultiLayerFusionNet = None
            self.AttentionFusionNet = None
            self.TransformerFusionNet = None
    
    def fuse_scores(
        self,
        scores: Dict[str, List[float]],
        use_adaptive_weights: bool = True
    ) -> List[float]:
        """Fuse multiple score lists into a single score list (adaptive if enabled, supports neural/attention/transformer fusion)."""
        if not scores:
            return []
        if self.fusion_method == "neural_fusion" and self.fusion_net and self.fusion_net_trained:
            return self._neural_fusion(scores)
        if self.fusion_method == "multi_layer_fusion" and self.fusion_net and self.fusion_net_trained:
            return self._multi_layer_fusion(scores)
        if self.fusion_method == "attention_fusion" and self.fusion_net and self.fusion_net_trained:
            return self._attention_fusion(scores)
        if self.fusion_method == "transformer_fusion" and self.fusion_net and self.fusion_net_trained:
            return self._transformer_fusion(scores)
        if self.fusion_method == "weighted_sum":
            return self._weighted_sum_fusion(scores, use_adaptive_weights=use_adaptive_weights)
        elif self.fusion_method == "reciprocal_rank":
            return self._reciprocal_rank_fusion(scores)
        elif self.fusion_method == "borda_count":
            return self._borda_count_fusion(scores)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _weighted_sum_fusion(
        self,
        scores: Dict[str, List[float]],
        use_adaptive_weights: bool = True
    ) -> List[float]:
        """Fuse scores using weighted sum (adaptive if enabled)."""
        # Normalize scores
        normalized_scores = {}
        for method, score_list in scores.items():
            if score_list:
                normalized_scores[method] = normalize(
                    np.array(score_list).reshape(1, -1)
                ).flatten()
        # Use adaptive weights if enabled
        if use_adaptive_weights:
            methods = list(normalized_scores.keys())
            weights = self.performance_tracker.get_all_weights(methods)
        else:
            weights = self.fusion_weights
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        # Compute weighted sum
        fused_scores = np.zeros(len(next(iter(scores.values()))))
        for method, score_list in normalized_scores.items():
            weight = normalized_weights.get(method, 0.0)
            fused_scores += weight * score_list
        # Add explanation for transparency
        self.logger.info(f"Fusion weights used: {normalized_weights}")
        return fused_scores.tolist()
    
    def _reciprocal_rank_fusion(
        self,
        scores: Dict[str, List[float]]
    ) -> List[float]:
        """Fuse scores using reciprocal rank fusion."""
        n_docs = len(next(iter(scores.values())))
        fused_scores = np.zeros(n_docs)
        
        for method, score_list in scores.items():
            # Get ranks (1-based)
            ranks = np.argsort(np.argsort(-np.array(score_list))) + 1
            # Add reciprocal ranks
            fused_scores += 1.0 / ranks
        
        return fused_scores.tolist()
    
    def _borda_count_fusion(
        self,
        scores: Dict[str, List[float]]
    ) -> List[float]:
        """Fuse scores using Borda count."""
        n_docs = len(next(iter(scores.values())))
        fused_scores = np.zeros(n_docs)
        
        for method, score_list in scores.items():
            # Get ranks (0-based)
            ranks = np.argsort(np.argsort(-np.array(score_list)))
            # Add Borda counts
            fused_scores += (n_docs - ranks - 1)
        
        return fused_scores.tolist()

    def _neural_fusion(self, scores: Dict[str, List[float]]) -> List[float]:
        """Fuse scores using a neural network (requires torch, must be trained)."""
        import torch
        method_names = list(scores.keys())
        score_matrix = np.stack([scores[m] for m in method_names], axis=1)
        x = torch.tensor(score_matrix, dtype=torch.float32)
        with torch.no_grad():
            fused = self.fusion_net(x).squeeze(-1).numpy()
        return fused.tolist()

    def _multi_layer_fusion(self, scores: Dict[str, List[float]]) -> List[float]:
        """Fuse scores using a multi-layer neural network (requires torch, must be trained)."""
        import torch
        method_names = list(scores.keys())
        score_matrix = np.stack([scores[m] for m in method_names], axis=1)
        x = torch.tensor(score_matrix, dtype=torch.float32)
        with torch.no_grad():
            fused = self.fusion_net(x).squeeze(-1).numpy()
        return fused.tolist()

    def _attention_fusion(self, scores: Dict[str, List[float]]) -> List[float]:
        """Fuse scores using an attention-based neural network (requires torch, must be trained)."""
        import torch
        method_names = list(scores.keys())
        score_matrix = np.stack([scores[m] for m in method_names], axis=1)
        x = torch.tensor(score_matrix, dtype=torch.float32)
        with torch.no_grad():
            fused = self.fusion_net(x).squeeze(-1).numpy()
        return fused.tolist()

    def _transformer_fusion(self, scores: Dict[str, List[float]]) -> List[float]:
        """Fuse scores using a transformer-based neural network (requires torch, must be trained)."""
        import torch
        method_names = list(scores.keys())
        score_matrix = np.stack([scores[m] for m in method_names], axis=1)
        x = torch.tensor(score_matrix, dtype=torch.float32)
        with torch.no_grad():
            fused = self.fusion_net(x).squeeze(-1).numpy()
        return fused.tolist()

    def train_neural_fusion(self, scores: List[Dict[str, List[float]]], labels: List[List[float]], method: str = "simple"):
        """Train the neural/attention/transformer fusion model (requires torch). method: 'simple', 'multi_layer', 'attention', 'transformer'"""
        if method == "simple":
            Net = self.SimpleFusionNet
        elif method == "multi_layer":
            Net = self.MultiLayerFusionNet
        elif method == "attention":
            Net = self.AttentionFusionNet
        elif method == "transformer":
            Net = self.TransformerFusionNet
        else:
            raise ValueError("Unknown fusion net type")
        if not Net:
            raise ImportError("PyTorch is required for neural fusion.")
        import torch
        import torch.nn as nn
        import torch.optim as optim
        method_names = list(scores[0].keys())
        n_methods = len(method_names)
        self.fusion_net = Net(n_methods)
        optimizer = optim.Adam(self.fusion_net.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        x = torch.tensor(np.stack([np.stack([s[m] for m in method_names], axis=1) for s in scores]), dtype=torch.float32)
        y = torch.tensor(np.array(labels), dtype=torch.float32)
        for epoch in range(100):
            optimizer.zero_grad()
            out = self.fusion_net(x).squeeze(-1)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
        self.fusion_net_trained = True

    def submit_feedback(self, method: str, feedback: float):
        """Submit user feedback for a fusion method (1.0=good, 0.0=bad, or any float)."""
        self.performance_tracker.submit_feedback(method, feedback)

class PersistenceManager:
    """Manages persistence of vector stores to different backends."""
    
    def __init__(
        self,
        vector_store: "EnhancedVectorStore",
        persistence_type: str = "local",
        persistence_config: Optional[Dict[str, Any]] = None
    ):
        self.vector_store = vector_store
        self.persistence_type = persistence_type
        self.persistence_config = persistence_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize persistence backend
        if persistence_type == "s3":
            self.s3_client = boto3.client(
                "s3",
                **self.persistence_config.get("aws_config", {})
            )
    
    async def save(self, path: str) -> None:
        """Save vector store to persistent storage."""
        try:
            if self.persistence_type == "local":
                await self._save_local(path)
            elif self.persistence_type == "s3":
                await self._save_s3(path)
            else:
                raise ValueError(f"Unsupported persistence type: {self.persistence_type}")
            
            self.logger.info(f"Saved vector store to {path}")
        
        except Exception as e:
            self.logger.error(f"Error saving vector store: {e}")
            raise
    
    async def load(self, path: str) -> None:
        """Load vector store from persistent storage."""
        try:
            if self.persistence_type == "local":
                await self._load_local(path)
            elif self.persistence_type == "s3":
                await self._load_s3(path)
            else:
                raise ValueError(f"Unsupported persistence type: {self.persistence_type}")
            
            self.logger.info(f"Loaded vector store from {path}")
        
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            raise
    
    async def _save_local(self, path: str) -> None:
        """Save vector store to local storage."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store state
        state = {
            "config": self.vector_store.config.__dict__,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        async with aiofiles.open(path / "state.json", "w") as f:
            await f.write(json.dumps(state))
        
        # Save vector store data
        await self.vector_store.persist(str(path / "data"))
    
    async def _load_local(self, path: str) -> None:
        """Load vector store from local storage."""
        path = Path(path)
        
        # Load vector store state
        async with aiofiles.open(path / "state.json", "r") as f:
            state = json.loads(await f.read())
        
        # Update config
        self.vector_store.config = EnhancedVectorStoreConfig(**state["config"])
        
        # Load vector store data
        await self.vector_store.load(str(path / "data"))
    
    async def _save_s3(self, path: str) -> None:
        """Save vector store to S3."""
        # Create temporary directory
        temp_dir = Path("temp_vector_store")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Save to temporary directory
            await self._save_local(str(temp_dir))
            
            # Upload to S3
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    s3_key = f"{path}/{file_path.relative_to(temp_dir)}"
                    self.s3_client.upload_file(
                        str(file_path),
                        self.persistence_config["bucket"],
                        s3_key
                    )
        
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)
    
    async def _load_s3(self, path: str) -> None:
        """Load vector store from S3."""
        # Create temporary directory
        temp_dir = Path("temp_vector_store")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Download from S3
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(
                Bucket=self.persistence_config["bucket"],
                Prefix=path
            ):
                for obj in page.get("Contents", []):
                    s3_key = obj["Key"]
                    local_path = temp_dir / s3_key[len(path) + 1:]
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    self.s3_client.download_file(
                        self.persistence_config["bucket"],
                        s3_key,
                        str(local_path)
                    )
            
            # Load from temporary directory
            await self._load_local(str(temp_dir))
        
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)

class MetadataIndexHandler:
    """Handles metadata indexing and filtering."""
    
    def __init__(
        self,
        vector_store: "EnhancedVectorStore",
        indexed_fields: List[str],
        index_type: str = "btree"
    ):
        self.vector_store = vector_store
        self.indexed_fields = indexed_fields
        self.index_type = index_type
        self.metadata_db = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize metadata indexing."""
        # Create SQLite database for metadata
        self.metadata_db = sqlite3.connect(":memory:")
        cursor = self.metadata_db.cursor()
        
        # Create metadata table
        fields_sql = ", ".join(
            f"{field} TEXT" for field in self.indexed_fields
        )
        cursor.execute(f"""
            CREATE TABLE metadata (
                id TEXT PRIMARY KEY,
                {fields_sql}
            )
        """)
        
        # Create indices
        for field in self.indexed_fields:
            cursor.execute(f"""
                CREATE INDEX idx_{field} ON metadata ({field})
            """)
        
        self.metadata_db.commit()
    
    async def index_metadata(
        self,
        ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Index metadata for documents."""
        if not self.metadata_db:
            await self.initialize()
        
        cursor = self.metadata_db.cursor()
        
        # Prepare data
        data = []
        for id, metadata in zip(ids, metadatas):
            row = [id]
            for field in self.indexed_fields:
                row.append(str(metadata.get(field, "")))
            data.append(row)
        
        # Insert or update metadata
        cursor.executemany(
            f"""
            INSERT OR REPLACE INTO metadata (id, {", ".join(self.indexed_fields)})
            VALUES ({", ".join("?" * (len(self.indexed_fields) + 1))})
            """,
            data
        )
        
        self.metadata_db.commit()
    
    async def search_metadata(
        self,
        filter_criteria: Dict[str, Any]
    ) -> List[str]:
        """Search metadata using filter criteria."""
        if not self.metadata_db:
            return []
        
        cursor = self.metadata_db.cursor()
        
        # Build query
        conditions = []
        params = []
        for field, value in filter_criteria.items():
            if field in self.indexed_fields:
                conditions.append(f"{field} = ?")
                params.append(str(value))
        
        if not conditions:
            return []
        
        # Execute query
        query = f"""
            SELECT id FROM metadata
            WHERE {" AND ".join(conditions)}
        """
        cursor.execute(query, params)
        
        return [row[0] for row in cursor.fetchall()]
    
    async def get_metadata_scores(
        self,
        ids: List[str],
        filter_criteria: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get metadata relevance scores for documents."""
        if not self.metadata_db:
            return {id: 0.0 for id in ids}
        
        cursor = self.metadata_db.cursor()
        
        # Build query
        conditions = []
        params = []
        for field, value in filter_criteria.items():
            if field in self.indexed_fields:
                conditions.append(f"{field} = ?")
                params.append(str(value))
        
        if not conditions:
            return {id: 0.0 for id in ids}
        
        # Execute query
        query = f"""
            SELECT id, COUNT(*) as matches
            FROM metadata
            WHERE {" AND ".join(conditions)}
            GROUP BY id
        """
        cursor.execute(query, params)
        
        # Calculate scores
        scores = {id: 0.0 for id in ids}
        for id, matches in cursor.fetchall():
            if id in scores:
                scores[id] = matches / len(conditions)
        
        return scores

class EnhancedVectorStore(VectorStore):
    """Enhanced vector store with advanced features."""
    
    def __init__(self, config: EnhancedVectorStoreConfig):
        """Initialize enhanced vector store."""
        super().__init__(config)
        
        # Initialize components
        self.plugin_registry = PluginRegistry()
        self.live_update_handler = None
        self.hybrid_search_handler = None
        self.scoring_fusion_handler = None
        self.persistence_manager = None
        self.metadata_index_handler = None
        
        # Register built-in plugins
        self._register_builtin_plugins()
        
        # Initialize components based on config
        if config.enable_live_updates:
            self.live_update_handler = LiveUpdateHandler(
                self,
                config.update_batch_size,
                config.update_interval
            )
        
        if config.enable_hybrid_search:
            self.hybrid_search_handler = HybridSearchHandler(
                self,
                config.bm25_weight,
                config.vector_weight
            )
        
        if config.enable_scoring_fusion:
            self.scoring_fusion_handler = ScoringFusionHandler(
                self,
                config.fusion_method,
                config.fusion_weights
            )
        
        self.persistence_manager = PersistenceManager(
            self,
            config.persistence_type,
            config.persistence_config
        )
        
        if config.enable_metadata_indexing:
            self.metadata_index_handler = MetadataIndexHandler(
                self,
                config.indexed_metadata_fields,
                config.metadata_index_type
            )
    
    def _register_builtin_plugins(self) -> None:
        """Register built-in vector store plugins."""
        self.plugin_registry.register_plugin("faiss", FAISSBackend)
        self.plugin_registry.register_plugin("chroma", ChromaBackend)
        self.plugin_registry.register_plugin("weaviate", WeaviateBackend)
        self.plugin_registry.register_plugin("qdrant", QdrantBackend)
        self.plugin_registry.register_plugin("milvus", MilvusBackend)
        self.plugin_registry.register_plugin("pinecone", PineconeBackend)
        self.plugin_registry.register_plugin("elasticsearch", ElasticsearchBackend)
        self.plugin_registry.register_plugin("redis", RedisBackend)
        self.plugin_registry.register_plugin("postgres", PostgresBackend)
    
    async def initialize(self) -> None:
        """Initialize enhanced vector store."""
        await super().initialize()
        
        # Initialize components
        if self.live_update_handler:
            await self.live_update_handler.start()
        
        if self.hybrid_search_handler:
            await self.hybrid_search_handler.initialize()
        
        if self.metadata_index_handler:
            await self.metadata_index_handler.initialize()
    
    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to store with live updates."""
        if self.live_update_handler:
            await self.live_update_handler.queue_update(
                "add",
                vectors=vectors,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
        else:
            await super().add_vectors(vectors, metadatas, documents, ids)
        
        # Update metadata index
        if self.metadata_index_handler:
            await self.metadata_index_handler.index_metadata(ids or [], metadatas)
    
    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None
    ) -> List[SearchResult]:
        """Enhanced search with hybrid and metadata support."""
        # Get metadata filter results
        metadata_ids = None
        metadata_scores = {}
        if self.metadata_index_handler and filter_criteria:
            metadata_ids = await self.metadata_index_handler.search_metadata(
                filter_criteria
            )
            metadata_scores = await self.metadata_index_handler.get_metadata_scores(
                metadata_ids,
                filter_criteria
            )
        
        # Perform search
        if self.hybrid_search_handler and query_text:
            results = await self.hybrid_search_handler.search(
                query_text,
                query_vector,
                k,
                filter_criteria
            )
        else:
            results = await super().search(
                query_vector,
                k,
                filter_criteria
            )
        
        # Apply metadata filtering
        if metadata_ids is not None:
            results = [
                result for result in results
                if result.id in metadata_ids
            ]
        
        # Apply scoring fusion
        if self.scoring_fusion_handler:
            # Prepare scores for fusion
            scores = {
                "vector": [r.score for r in results],
                "metadata": [
                    metadata_scores.get(r.id, 0.0)
                    for r in results
                ]
            }
            
            # Fuse scores
            fused_scores = self.scoring_fusion_handler.fuse_scores(scores)
            
            # Update result scores
            for result, score in zip(results, fused_scores):
                result.score = float(score)
        
        return results
    
    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors with live updates."""
        if self.live_update_handler:
            await self.live_update_handler.queue_update(
                "delete",
                ids=ids
            )
        else:
            await super().delete_vectors(ids)
    
    async def clear(self) -> None:
        """Clear vector store."""
        await super().clear()
        
        # Clear metadata index
        if self.metadata_index_handler:
            await self.metadata_index_handler.initialize()
    
    async def persist(self, path: str) -> None:
        """Persist vector store using persistence manager."""
        await self.persistence_manager.save(path)
    
    @classmethod
    async def load(cls, path: str, config: EnhancedVectorStoreConfig) -> "EnhancedVectorStore":
        """Load vector store using persistence manager."""
        store = cls(config)
        await store.persistence_manager.load(path)
        return store
    
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the store."""
        # This is a placeholder - implement based on backend
        return []
    
    def get_document_index(self, doc_id: str) -> Optional[int]:
        """Get document index by ID."""
        # This is a placeholder - implement based on backend
        return None 