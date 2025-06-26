from multimind.llm.non_transformer_llm import NonTransformerLLM
import spacy
import asyncio

# Load spaCy model
nlp = spacy.blank("en")
# For demonstration, add a simple NER pipe if not present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
    # Add a dummy entity label for demonstration
    ner.add_label("ORG")

# Custom wrapper for spaCy NER
class SpacyNERLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
    async def generate(self, prompt: str, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(None, lambda: self.model(prompt))
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return str(entities)

llm = SpacyNERLLM(
    model_name="spacy_ner",
    model_instance=nlp
)

async def main():
    prompt = "Apple is looking at buying U.K. startup for $1 billion"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nNamed Entities: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 