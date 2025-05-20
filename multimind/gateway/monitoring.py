"""
Monitoring and metrics module for the MultiMind Gateway
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdic
import asyncio
import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Metrics for model performance and usage"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    last_used: Optional[datetime] = None
    error_count: Dict[str, int] = defaultdict(int)

class ModelHealth(BaseModel):
    """Health status of a model"""
    is_healthy: bool
    last_check: datetime
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None
    uptime_percentage: float = 100.0

class ModelMonitor:
    """Monitor model health, usage, and performance"""

    def __init__(self):
        self.metrics: Dict[str, ModelMetrics] = defaultdict(ModelMetrics)
        self.health: Dict[str, ModelHealth] = {}
        self.rate_limits: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "requests_per_minute": 60,
            "tokens_per_minute": 100000
        })
        self._lock = asyncio.Lock()

    async def track_request(
        self,
        model: str,
        tokens: int,
        cost: float,
        response_time: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Track a model request and its metrics"""
        async with self._lock:
            metrics = self.metrics[model]
            metrics.total_requests += 1
            metrics.total_tokens += tokens
            metrics.total_cost += cos
            metrics.last_used = datetime.now()

            # Update response time (moving average)
            if metrics.avg_response_time == 0:
                metrics.avg_response_time = response_time
            else:
                metrics.avg_response_time = (metrics.avg_response_time * 0.9 +
                                          response_time * 0.1)

            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
                if error:
                    metrics.error_count[error] += 1

    async def check_health(self, model: str, handler) -> ModelHealth:
        """Check the health of a model"""
        try:
            start_time = time.time()
            # Simple health check with a test promp
            response = await handler.generate("test")
            latency = (time.time() - start_time) * 1000  # Convert to ms

            health = ModelHealth(
                is_healthy=True,
                last_check=datetime.now(),
                latency_ms=latency
            )
        except Exception as e:
            health = ModelHealth(
                is_healthy=False,
                last_check=datetime.now(),
                error_message=str(e)
            )

        self.health[model] = health
        return health

    async def get_metrics(self, model: Optional[str] = None) -> Dict:
        """Get metrics for a specific model or all models"""
        if model:
            return {
                "metrics": self.metrics[model],
                "health": self.health.get(model)
            }
        return {
            model: {
                "metrics": metrics,
                "health": self.health.get(model)
            }
            for model, metrics in self.metrics.items()
        }

    def set_rate_limits(
        self,
        model: str,
        requests_per_minute: int,
        tokens_per_minute: in
    ) -> None:
        """Set rate limits for a model"""
        self.rate_limits[model].update({
            "requests_per_minute": requests_per_minute,
            "tokens_per_minute": tokens_per_minute
        })

    async def check_rate_limit(self, model: str, tokens: int) -> bool:
        """Check if a request would exceed rate limits"""
        # Implement rate limiting logic here
        # This is a placeholder for actual rate limiting implementation
        return True

# Global monitor instance
monitor = ModelMonitor()