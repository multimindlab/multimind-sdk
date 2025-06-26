from multimind.llm.non_transformer_llm import NonTransformerLLM
from textblob import TextBlob
import asyncio

# Custom wrapper for TextBlob
class TextBlobSentimentLLM(NonTransformerLLM):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, None, **kwargs)
    async def generate(self, prompt: str, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        blob = await loop.run_in_executor(None, lambda: TextBlob(prompt))
        sentiment = blob.sentiment
        return f"polarity: {sentiment.polarity}, subjectivity: {sentiment.subjectivity}"

llm = TextBlobSentimentLLM(
    model_name="textblob_sentiment"
)

async def main():
    prompt = "I love programming but hate bugs."
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nSentiment: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 