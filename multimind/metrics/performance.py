"""
Performance tracking helpers.

This module provides a small, stable API used by examples.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, DefaultDict, Dict


class PerformanceTracker:
    """
    Tracks latency/success per (modality, model_id).

    Expected example surface:
    - get_current_time() -> float
    - track_latency(modality, latency) -> None
    - track_error(modality, error) -> None
    - get_modality_metrics(modality) -> dict[model_id] -> {"success_rate", "avg_latency"}
    """

    def __init__(self) -> None:
        # stats[modality][model_id] = {"success": int, "fail": int, "lat_total": float, "lat_count": int}
        self._stats: DefaultDict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def _ensure(self, modality: str, model_id: str) -> Dict[str, Any]:
        if model_id not in self._stats[modality]:
            self._stats[modality][model_id] = {
                "success": 0,
                "fail": 0,
                "lat_total": 0.0,
                "lat_count": 0,
            }
        return self._stats[modality][model_id]

    def get_current_time(self) -> float:
        return time.time()

    def track_latency(
        self, modality: str, latency: float, model_id: str = "unknown", success: bool = True
    ) -> None:
        stat = self._ensure(modality, model_id)
        if success:
            stat["success"] += 1
        else:
            stat["fail"] += 1
        try:
            lat = float(latency)
        except (TypeError, ValueError):
            lat = 0.0
        stat["lat_total"] += lat
        stat["lat_count"] += 1

    def track_error(self, modality: str, error: str, model_id: str = "unknown") -> None:
        stat = self._ensure(modality, model_id)
        stat["fail"] += 1

    def track_success(self, modality: str, model_id: str = "unknown") -> None:
        """Convenience method for recording a success without latency."""
        stat = self._ensure(modality, model_id)
        stat["success"] += 1

    def get_modality_metrics(self, modality: str) -> Dict[str, Dict[str, Any]]:
        metrics: Dict[str, Dict[str, Any]] = {}
        for model_id, stat in self._stats.get(modality, {}).items():
            success = int(stat.get("success", 0))
            fail = int(stat.get("fail", 0))
            total = success + fail
            lat_count = int(stat.get("lat_count", 0))
            lat_total = float(stat.get("lat_total", 0.0))
            metrics[model_id] = {
                "success_rate": (success / total) if total else 0.0,
                "avg_latency": (lat_total / lat_count) if lat_count else 0.0,
                "success": success,
                "fail": fail,
            }
        return metrics
