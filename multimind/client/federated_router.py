from typing import Callable, Dict, Any
import time

class FederatedRouter:
    """
    Routes between local (on-device) and cloud model clients based on context (input size, latency, privacy, etc.).
    Supports custom routing logic via router_fn.
    """
    def __init__(self, local_client, cloud_client, router_fn: Callable[[str, Dict], str] = None):
        self.clients = {"local": local_client, "cloud": cloud_client}
        self.router_fn = router_fn or self.default_router
        self.metrics = {"local": {"latency": [], "count": 0}, "cloud": {"latency": [], "count": 0}}

    def default_router(self, prompt: str, metrics: Dict) -> str:
        # Example: use local for short prompts, cloud for long
        if len(prompt) < 256:
            return "local"
        return "cloud"

    def generate(self, prompt: str, **kwargs):
        # Compute average latency for each client
        avg_latencies = {k: (sum(v["latency"]) / len(v["latency"]) if v["latency"] else float('inf')) for k, v in self.metrics.items()}
        selected = self.router_fn(prompt, {"avg_latencies": avg_latencies, **self.metrics})
        client = self.clients[selected]
        start = time.time()
        result = client.generate(prompt, **kwargs)
        elapsed = time.time() - start
        self.metrics[selected]["latency"].append(elapsed)
        self.metrics[selected]["count"] += 1
        return result

    def register_client(self, name: str, client):
        self.clients[name] = client
        self.metrics[name] = {"latency": [], "count": 0}

# --- Example usage ---
if __name__ == "__main__":
    class DummyClient:
        def generate(self, prompt, **kwargs):
            return f"[{self.__class__.__name__} output for: {prompt}]"
    local = DummyClient()
    cloud = DummyClient()
    router = FederatedRouter(local, cloud)
    print(router.generate("short prompt"))
    print(router.generate("This is a very long prompt that should go to the cloud..." * 20)) 