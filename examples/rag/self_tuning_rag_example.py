import asyncio
from multimind.patterns.advanced_patterns import SelfImprovingRAG
from multimind.retrieval.retrieval import HybridRetriever
from multimind.memory import TokenAwareMemory

# Mock LLM, Retriever, and PEFTTuner for demonstration
class MockLLM:
    async def generate(self, prompt, **kwargs):
        return "mocked LLM response"

class MockRetriever(HybridRetriever):
    def __init__(self):
        super().__init__(dense_retriever=MockLLM())
        self.top_k = 5
        self.alpha = 0.5
    async def retrieve(self, query, **kwargs):
        return [{"id": "doc1", "content": "mocked doc content"}]

class MockMemory(TokenAwareMemory):
    async def get_relevant_memory(self, query, **kwargs):
        return []
    async def add_conversation_turn(self, query, response, context, metadata):
        pass

class MockPEFTTuner:
    def __init__(self):
        self.model = MockLLM()
        self.output_dir = "/tmp/mock_model"
    def train(self, train_data):
        print(f"[PEFTTuner] Training on {len(train_data)} samples...")
    def save_model(self):
        print("[PEFTTuner] Model saved.")
    def load_model(self, path):
        print(f"[PEFTTuner] Model loaded from {path}.")

async def main():
    # Instantiate mocks
    llm = MockLLM()
    retriever = MockRetriever()
    memory = MockMemory()
    peft_tuner = MockPEFTTuner()

    # Create SelfImprovingRAG with auto-retraining enabled
    rag = SelfImprovingRAG(
        model=llm,
        retriever=retriever,
        memory=memory,
        peft_tuner=peft_tuner,
        retrain_threshold=0.8,  # retrain if quality < 0.8
        retrain_window=3,
        retrain_cooldown=1  # allow retrain every second for demo
    )

    # Simulate queries and feedback
    queries = [
        "What is self-tuning RAG?",
        "How does feedback improve retrieval?",
        "Explain auto-retraining in RAG systems."
    ]
    feedbacks = [
        {"thumbs": "down"},
        {"thumbs": "down"},
        {"thumbs": "down"}  # This should trigger retraining
    ]

    for i, query in enumerate(queries):
        response, metadata = await rag.process_query(query)
        print(f"\nQuery: {query}\nResponse: {response}")
        rag.submit_feedback(query, response, feedbacks[i])
        # Optionally analyze feedback
        analytics = await rag.analyze_feedback()
        print(f"Feedback analytics: {analytics['stats']}")

if __name__ == "__main__":
    asyncio.run(main()) 