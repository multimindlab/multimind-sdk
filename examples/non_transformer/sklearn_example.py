from multimind.llm.non_transformer_llm import NonTransformerLLM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import asyncio

# NonTransformerLLM is an extension scaffold: subclass it and implement generate.
class SklearnTextClassifierLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, vectorizer, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.vectorizer = vectorizer

    async def generate(self, prompt: str, **kwargs) -> str:
        X = self.vectorizer.transform([prompt])
        return str(self.model.predict(X)[0])

# Toy training data
texts = [
    "I love programming in Python",
    "Python is great for data science",
    "I dislike bugs in code",
    "Debugging is fun",
    "I enjoy machine learning"
]
labels = ["positive", "positive", "negative", "positive", "positive"]

# Train vectorizer and classifier
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
clf = MultinomialNB()
clf.fit(X, labels)

llm = SklearnTextClassifierLLM(
    model_name="sklearn_nb",
    model_instance=clf,
    vectorizer=vectorizer
)

async def main():
    prompt = "I hate errors in my code"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nPredicted label: {result}")

if __name__ == "__main__":
    asyncio.run(main())
