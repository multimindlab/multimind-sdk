from multimind.llm.non_transformer_llm import NonTransformerLLM
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import asyncio

# Toy training data
texts = [
    "I love programming in Python",
    "Python is great for data science",
    "I dislike bugs in code",
    "Debugging is fun",
    "I enjoy machine learning"
]
labels = [1, 1, 0, 1, 1]  # 1=positive, 0=negative

# Vectorize text
tokenizer = CountVectorizer()
X = tokenizer.fit_transform(texts)
y = np.array(labels)

# Train Logistic Regression classifier
model = LogisticRegression()
model.fit(X, y)

# Custom wrapper for scikit-learn Logistic Regression
class SklearnLogRegLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, vectorizer, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.vectorizer = vectorizer

    async def generate(self, prompt: str, **kwargs) -> str:
        X = self.vectorizer.transform([prompt])
        loop = asyncio.get_event_loop()
        pred = await loop.run_in_executor(None, lambda: self.model.predict(X))
        return str(pred[0])

# Wrap the model
llm = SklearnLogRegLLM(
    model_name="sklearn_logreg_classifier",
    model_instance=model,
    vectorizer=tokenizer
)

async def main():
    prompt = "I hate errors in my code"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nPredicted label: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 