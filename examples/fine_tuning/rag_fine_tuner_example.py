# Example usage for demonstration purposes only
from multimind.fine_tuning.rag_fine_tuner import RAGFineTuner

def dummy_rag_pipeline(query):
    return {"context": f"[Context for: {query}]", "answer": f"[Answer for: {query}]"}
def dummy_fine_tune(train_data, **kwargs):
    print(f"Fine-tuning on {len(train_data)} examples.")
    return "fine-tuned-model"
rag_ft = RAGFineTuner(dummy_rag_pipeline, dummy_fine_tune)
queries = ["What is the capital of France?", "Who wrote Hamlet?"]
model = rag_ft.auto_ft_from_rag(queries, n_per_query=2)
print("Resulting model:", model) 