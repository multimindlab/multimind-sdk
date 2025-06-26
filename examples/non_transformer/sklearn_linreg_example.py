from multimind.llm.non_transformer_llm import NonTransformerLLM
from sklearn.linear_model import LinearRegression
import numpy as np
import asyncio

# Toy training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])  # y = 2x

# Train Linear Regression regressor
model = LinearRegression()
model.fit(X, y)

# Custom wrapper for scikit-learn Linear Regression
class SklearnLinRegLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)

    async def generate(self, prompt: str, **kwargs) -> str:
        # Expect prompt to be a number (as string)
        try:
            x_val = float(prompt)
        except Exception:
            return "Invalid input"
        X_input = np.array([[x_val]])
        loop = asyncio.get_event_loop()
        pred = await loop.run_in_executor(None, lambda: self.model.predict(X_input))
        return str(pred[0])

# Wrap the model
llm = SklearnLinRegLLM(
    model_name="sklearn_linreg_regressor",
    model_instance=model
)

async def main():
    prompt = "7"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nPredicted value: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 