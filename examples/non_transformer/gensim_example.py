from multimind.llm.non_transformer_llm import NonTransformerLLM
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import asyncio

# Toy data
documents = [
    ["human", "interface", "computer"],
    ["survey", "user", "computer", "system", "response", "time"],
    ["eps", "user", "interface", "system"],
    ["system", "human", "system", "eps"],
    ["user", "response", "time"],
    ["trees"],
    ["graph", "trees"],
    ["graph", "minors", "trees"],
    ["graph", "minors", "survey"]
]

# Prepare dictionary and corpus
dictionary = Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Train LDA model
lda = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Custom wrapper for Gensim LDA
class GensimLDALLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, dictionary, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.dictionary = dictionary
    async def generate(self, prompt: str, **kwargs) -> str:
        # Expect prompt as space-separated words
        bow = self.dictionary.doc2bow(prompt.split())
        loop = asyncio.get_event_loop()
        topics = await loop.run_in_executor(None, lambda: self.model.get_document_topics(bow))
        return str(topics)

llm = GensimLDALLM(
    model_name="gensim_lda",
    model_instance=lda,
    dictionary=dictionary
)

async def main():
    prompt = "user response time"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nTopic Distribution: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 