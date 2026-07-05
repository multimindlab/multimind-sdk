"""
Cost tracking helpers.

The SDK has multiple cost/perf tracking implementations; this one is a small,
stable surface used by examples (e.g. multi-modal cost optimization).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict


class CostTracker:
    """
    Tracks costs per (modality, model_id).

    Expected example surface:
    - get_modality_cost(modality, result) -> float
    - get_modality_metrics(modality) -> dict[model_id] -> {"avg_cost", "total_cost", "count"}
    """

    def __init__(self) -> None:
        # stats[modality][model_id] = {"total_cost": float, "count": int}
        self._stats: DefaultDict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def _ensure(self, modality: str, model_id: str) -> Dict[str, Any]:
        if model_id not in self._stats[modality]:
            self._stats[modality][model_id] = {"total_cost": 0.0, "count": 0}
        return self._stats[modality][model_id]

    def get_modality_cost(self, modality: str, result: Any) -> float:
        """
        Extract + record cost for a modality result.

        Heuristics:
        - If result is a dict, read result["cost"] (default 0.0)
        - model id is read from result["model_id"] or result["model"] (default "unknown")
        """

        model_id = "unknown"
        cost = 0.0

        data: Any = result
        # Support Pydantic models (v1/v2) and plain objects
        if hasattr(result, "model_dump"):
            data = result.model_dump()
        elif hasattr(result, "dict"):
            try:
                data = result.dict()
            except Exception:
                data = result

        if isinstance(data, dict):
            metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            model_id = str(
                data.get("model_id") or data.get("model") or metadata.get("model_id") or "unknown"
            )
            try:
                cost = float(data.get("cost") or metadata.get("cost") or 0.0)
            except (TypeError, ValueError):
                cost = 0.0

        stat = self._ensure(modality, model_id)
        stat["total_cost"] += cost
        stat["count"] += 1

        return cost

    def get_modality_metrics(self, modality: str) -> Dict[str, Dict[str, Any]]:
        """Return metrics keyed by model id."""

        metrics: Dict[str, Dict[str, Any]] = {}
        for model_id, stat in self._stats.get(modality, {}).items():
            count = int(stat.get("count", 0)) or 0
            total_cost = float(stat.get("total_cost", 0.0)) if count else 0.0
            avg_cost = total_cost / count if count else 0.0
            metrics[model_id] = {
                "avg_cost": avg_cost,
                "total_cost": total_cost,
                "count": count,
            }
        return metrics
