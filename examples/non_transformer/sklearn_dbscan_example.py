from multimind.llm.non_transformer_llm import NonTransformerLLM
from sklearn.cluster import DBSCAN
import numpy as np
import asyncio

# Toy data (2D points)
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# Train DBSCAN
model = DBSCAN(eps=3, min_samples=2)
model.fit(X)

# Custom wrapper for DBSCAN
class SklearnDBSCANLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
    async def generate(self, prompt: str, **kwargs) -> str:
        # Expect prompt as comma-separated numbers, e.g. "1,2"
        try:
            point = np.array([[float(x.strip()) for x in prompt.split(",")]])
        except Exception:
            return "Invalid input"
        # DBSCAN does not have predict, so use fit_predict for demonstration
        X_full = np.vstack([model.components_, point]) if hasattr(model, 'components_') else point
        loop = asyncio.get_event_loop()
        pred = await loop.run_in_executor(None, lambda: self.model.fit_predict(X_full))
        return str(pred[-1])

llm = SklearnDBSCANLLM(
    model_name="sklearn_dbscan",
    model_instance=model
)

async def main():
    prompt = "0,0"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nPredicted cluster: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 