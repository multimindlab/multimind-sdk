from multimind.llm.non_transformer_llm import NonTransformerLLM
import sklearn_crfsuite
import asyncio

# Toy data: sequence labeling (BIO format)
X_train = [[{'word': 'John'}, {'word': 'loves'}, {'word': 'Mary'}]]
y_train = [['B-PER', 'O', 'B-PER']]

# Feature extraction for CRF
# For demonstration, use word as the only feature
def sent2features(sent):
    return [{'word': token['word']} for token in sent]

X_train_feats = [sent2features(sent) for sent in X_train]

# Train CRF
crf = sklearn_crfsuite.CRF()
crf.fit(X_train_feats, y_train)

# Custom wrapper for CRF
class SklearnCRFLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
    async def generate(self, prompt: str, **kwargs) -> str:
        # Expect prompt as space-separated tokens
        tokens = prompt.split()
        feats = [{'word': t} for t in tokens]
        loop = asyncio.get_event_loop()
        pred = await loop.run_in_executor(None, lambda: self.model.predict([feats]))
        return str(pred[0])

llm = SklearnCRFLLM(
    model_name="sklearn_crf",
    model_instance=crf
)

async def main():
    prompt = "John loves Mary"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nPredicted labels: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 