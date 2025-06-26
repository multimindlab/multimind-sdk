from multimind.llm.non_transformer_llm import NonTransformerLLM
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import asyncio

# Download VADER lexicon if not already present
nltk.download('vader_lexicon', quiet=True)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Custom wrapper for NLTK VADER
class NLTKSentimentLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
    async def generate(self, prompt: str, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, lambda: self.model.polarity_scores(prompt))
        return str(scores)

llm = NLTKSentimentLLM(
    model_name="nltk_vader_sentiment",
    model_instance=sia
)

async def main():
    prompt = "I love programming but hate bugs."
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nSentiment Scores: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 