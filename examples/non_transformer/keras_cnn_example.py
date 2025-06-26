from multimind.llm.non_transformer_llm import NonTransformerLLM
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=8)
y = np.array(labels)

# Build a simple CNN model
model = Sequential([
    Embedding(input_dim=100, output_dim=8, input_length=8),
    Conv1D(16, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, verbose=0)

# Custom wrapper for Keras CNN
class KerasCNNTextClassifierLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, tokenizer, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.tokenizer = tokenizer

    async def generate(self, prompt: str, **kwargs) -> str:
        seq = self.tokenizer.texts_to_sequences([prompt])
        seq = pad_sequences(seq, maxlen=8)
        # Keras predict is sync, so run in executor
        loop = asyncio.get_event_loop()
        pred = await loop.run_in_executor(None, lambda: self.model.predict(seq))
        label = int(pred[0][0] > 0.5)
        return str(label)

# Wrap the model
llm = KerasCNNTextClassifierLLM(
    model_name="keras_cnn",
    model_instance=model,
    tokenizer=tokenizer
)

async def main():
    prompt = "I hate errors in my code"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nPredicted label: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 