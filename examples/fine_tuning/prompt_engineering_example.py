from multimind.fine_tuning.unified_fine_tuner import PromptEngineeringMixin

class MyPromptModel(PromptEngineeringMixin):
    def format_prompt(self, prompt, examples=None, **kwargs):
        # Pseudo-code: concatenate examples and prompt
        if examples:
            return '\n'.join(examples) + '\n' + prompt
        return prompt

model = MyPromptModel()
prompt = "What is the capital of France?"
examples = ["Q: What is the capital of Germany?\nA: Berlin", "Q: What is the capital of Italy?\nA: Rome"]
formatted = model.format_prompt(prompt, examples)
print("Formatted prompt:\n", formatted) 