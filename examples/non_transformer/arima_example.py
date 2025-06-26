from multimind.llm.non_transformer_llm import NonTransformerLLM
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import asyncio

# Toy time series data
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Train ARIMA model
model = ARIMA(y, order=(1,1,1)).fit()

# Custom wrapper for ARIMA
class ARIMALLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)

    async def generate(self, prompt: str, **kwargs) -> str:
        # Expect prompt to be a comma-separated series
        try:
            series = np.array([float(x.strip()) for x in prompt.split(",")])
        except Exception:
            return "Invalid input"
        loop = asyncio.get_event_loop()
        pred = await loop.run_in_executor(None, lambda: self.model.apply(series))
        # Forecast next value
        forecast = self.model.forecast(steps=1)
        return str(forecast[0])

# Wrap the model
llm = ARIMALLM(
    model_name="arima_model",
    model_instance=model
)

async def main():
    prompt = "1,2,3,4,5,6,7,8,9,10"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nForecasted next value: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 