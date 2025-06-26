from multimind.llm.non_transformer_llm import SklearnTextClassifierLLM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import asyncio

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

# Wrap with SklearnTextClassifierLLM
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