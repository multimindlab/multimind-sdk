from multimind.llm.non_transformer_llm import NonTransformerLLM
from sklearn.mixture import GaussianMixture
import numpy as np
import asyncio

# Toy data (2D points)
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# Train GMM
model = GaussianMixture(n_components=2, random_state=0)
model.fit(X)

# Custom wrapper for GMM
class SklearnGMMLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
    async def generate(self, prompt: str, **kwargs) -> str:
        # Expect prompt as comma-separated numbers, e.g. "1,2"
        try:
            point = np.array([[float(x.strip()) for x in prompt.split(",")]])
        except Exception:
            return "Invalid input"
        loop = asyncio.get_event_loop()
        pred = await loop.run_in_executor(None, lambda: self.model.predict(point))
        return str(pred[0])

llm = SklearnGMMLLM(
    model_name="sklearn_gmm",
    model_instance=model
)

async def main():
    prompt = "0,0"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nPredicted cluster: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 