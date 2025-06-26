from multimind.llm.non_transformer_llm import NonTransformerLLM
from hmmlearn.hmm import GaussianHMM
import numpy as np
import asyncio

# Toy data: 1D sequence
X = np.array([[0.0], [1.0], [0.5], [1.5], [0.0], [1.0], [0.5], [1.5]])
lengths = [len(X)]

# Train HMM
model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
model.fit(X, lengths)

# Custom wrapper for HMM
class HMMGaussianLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
    async def generate(self, prompt: str, **kwargs) -> str:
        # Expect prompt as comma-separated numbers
        try:
            seq = np.array([[float(x.strip())] for x in prompt.split(",")])
        except Exception:
            return "Invalid input"
        loop = asyncio.get_event_loop()
        states = await loop.run_in_executor(None, lambda: self.model.predict(seq))
        return str(states.tolist())

llm = HMMGaussianLLM(
    model_name="hmm_gaussian",
    model_instance=model
)

async def main():
    prompt = "0.0,1.0,0.5,1.5"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nPredicted states: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 