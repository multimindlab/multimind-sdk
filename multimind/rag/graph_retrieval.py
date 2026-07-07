"""Knowledge-graph retrieval built on GraphRAG's LLM entity/relation extraction.

``GraphRetriever`` builds a document/entity graph with :class:`GraphRAG`
(strict-parsed LLM extraction) and ranks documents by query-entity overlap
plus one-hop neighbor expansion. Given fixed model output the ranking is
deterministic: scores break ties by document insertion order.
"""

from typing import Any, Dict, List, Optional, Union

from ..models.base import BaseLLM
from ..patterns.advanced_patterns import GraphRAG


class GraphRetriever:
    """Retrieves documents via an LLM-extracted knowledge graph.

    Args:
        model: LLM used for entity/relation extraction (parse-strict; malformed
            extraction output raises ``ValueError``).
        documents: Optional initial documents — strings or dicts with
            ``content`` (and optional ``id``/``metadata``).
        entity_weight: Score contribution per document occurrence of a
            query entity.
        neighbor_weight: Score contribution per document occurrence of a
            one-hop neighbor of a query entity.
    """

    def __init__(
        self,
        model: BaseLLM,
        documents: Optional[List[Union[str, Dict[str, Any]]]] = None,
        entity_weight: float = 1.0,
        neighbor_weight: float = 0.5,
        **kwargs,
    ):
        self.model = model
        self.entity_weight = entity_weight
        self.neighbor_weight = neighbor_weight
        self.graph_rag = GraphRAG(model, **kwargs)
        self._doc_order: List[str] = []
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._pending: List[Dict[str, Any]] = [
            self._normalize(doc, i) for i, doc in enumerate(documents or [])
        ]

    @staticmethod
    def _normalize(doc: Union[str, Dict[str, Any]], index: int) -> Dict[str, Any]:
        if isinstance(doc, str):
            return {"id": f"doc_{index}", "content": doc, "metadata": {}}
        if not isinstance(doc, dict) or "content" not in doc:
            raise ValueError(f"Documents must be strings or dicts with 'content', got {doc!r}")
        return {
            "id": str(doc.get("id", f"doc_{index}")),
            "content": doc["content"],
            "metadata": doc.get("metadata", {}),
        }

    async def add_documents(self, documents: List[Union[str, Dict[str, Any]]]) -> None:
        """Extract entities/relations from documents and add them to the graph."""
        offset = len(self._doc_order) + len(self._pending)
        for i, doc in enumerate(documents):
            self._pending.append(self._normalize(doc, offset + i))
        await self.build()

    async def build(self) -> None:
        """Run LLM extraction for all pending documents."""
        while self._pending:
            doc = self._pending.pop(0)
            if doc["id"] in self._documents:
                raise ValueError(f"Duplicate document id: {doc['id']}")
            await self.graph_rag.add_document(doc)
            self._documents[doc["id"]] = doc
            self._doc_order.append(doc["id"])

    async def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return up to ``k`` documents ranked by entity overlap + neighbors.

        Each result is the stored document dict plus a ``score`` field.
        Documents with zero score are omitted.
        """
        await self.build()
        query_entities = await self.graph_rag._extract_entities(query)

        scores: Dict[str, float] = {}
        matched_names = set()
        for entity in query_entities:
            direct_nodes = self.graph_rag._match_entity_nodes(entity)
            for node in direct_nodes:
                matched_names.add(
                    str(self.graph_rag.graph.nodes[node].get("name", "")).strip().lower()
                )
            for doc in self.graph_rag._find_documents_with_entity(entity):
                scores[doc["id"]] = scores.get(doc["id"], 0.0) + self.entity_weight

        # Neighbor expansion: entities one hop from a query entity also pull
        # in their documents, at reduced weight.
        for entity in query_entities:
            for neighbor in self.graph_rag._find_related_entities(entity):
                if str(neighbor.get("name", "")).strip().lower() in matched_names:
                    continue
                for doc in self.graph_rag._find_documents_with_entity(neighbor):
                    scores[doc["id"]] = scores.get(doc["id"], 0.0) + self.neighbor_weight

        # Stable sort over insertion order keeps ties deterministic.
        ranked = sorted(
            (doc_id for doc_id in self._doc_order if scores.get(doc_id, 0.0) > 0.0),
            key=lambda doc_id: -scores[doc_id],
        )
        return [{**self._documents[doc_id], "score": scores[doc_id]} for doc_id in ranked[:k]]
