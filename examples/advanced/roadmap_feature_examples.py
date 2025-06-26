import torch
from multimind.client.model_client import (
    LSTMModelClient, DynamicMoEModelClient, MultiModalClient, ImageModelClient, AudioModelClient, VideoModelClient, CodeModelClient
)
from multimind.llm.non_transformer_llm import MambaLLM
from multimind.agents.react_toolchain import ReasoningChain, ReasoningStep
from multimind.agents.agent_registry import AgentRegistry
from multimind.agents.prompt_correction import PromptCorrectionLayer
from multimind.client.federated_router import FederatedRouter
from multimind.fine_tuning.rag_fine_tuner import RAGFineTuner

# --- DynamicMoEModelClient Example ---
class DummyClient:
    def __init__(self, name): self.name = name
    def generate(self, prompt, **kwargs): return f"[{self.name} output for: {prompt}]"
def router_fn(prompt, metrics):
    if len(prompt) > 10: return "slow"
    return "fast"
moe_client = DynamicMoEModelClient({"fast": DummyClient("fast"), "slow": DummyClient("slow")}, router_fn)
print("DynamicMoEModelClient (short):", moe_client.generate("hi"))
print("DynamicMoEModelClient (long):", moe_client.generate("this is a long prompt"))

# --- MultiModalClient Example ---
mm_client = MultiModalClient(
    text_client=DummyClient("text"),
    image_client=ImageModelClient(),
    audio_client=AudioModelClient(),
    video_client=VideoModelClient(),
    code_client=CodeModelClient()
)
print("MultiModalClient (text):", mm_client.generate("hello", input_type="text"))
print("MultiModalClient (image):", mm_client.generate("image.jpg", input_type="image"))
print("MultiModalClient (audio):", mm_client.generate("audio.wav", input_type="audio"))
print("MultiModalClient (video):", mm_client.generate("video.mp4", input_type="video"))
print("MultiModalClient (code):", mm_client.generate("print('hi')", input_type="code"))

# --- Per-user/session LoRA injection Example ---
class DummyMamba(MambaLLM):
    def __init__(self): pass
    async def generate(self, prompt, *args, **kwargs): return f"[Mamba output for: {prompt}]"
mamba = DummyMamba()
mamba.load_adapter_for("user1", "adapter_path_1")
mamba.load_adapter_for("user2", "adapter_path_2")
print("Adapter for user1:", mamba.get_active_adapter("user1"))
mamba.unload_adapter_for("user1")
print("Adapter for user1 after unload:", mamba.get_active_adapter("user1"))

# --- ReasoningChain Example ---
def step1(x, context=None): return x + " step1"
def step2(x, context=None): return x + " step2"
chain = ReasoningChain([
    ReasoningStep("step1", step1),
    ReasoningStep("step2", step2)
])
print("ReasoningChain result:", chain.run("input"))

# --- AgentRegistry Example ---
registry = AgentRegistry()
def agent_a(query, state=None): return {"response": f"A: {query}", "state": {"last": query}}
def agent_b(query, state=None): return {"response": f"B: {query}", "state": {"last": query}}
registry.register_agent("a", agent_a)
registry.register_agent("b", agent_b)
registry.set_fallback("a", "b")
print("AgentRegistry (a):", registry.run_agent("a", "hello"))
print("AgentRegistry (a, fallback):", registry.run_agent("a", "fail"))

# --- PromptCorrectionLayer Example ---
pcl = PromptCorrectionLayer()
def error_logger(prompt, exc, trace): print(f"Error detected: {exc}")
def simple_correction(prompt, trace): return prompt + " [CORRECTED]"
def adapter_updater(adapter_key, new_path): print(f"Adapter {adapter_key} updated to {new_path}")
pcl.add_error_hook(error_logger)
pcl.add_correction_hook(simple_correction)
pcl.add_adapter_update_hook(adapter_updater)
print("PromptCorrectionLayer monitor:", pcl.monitor("prompt", "[error] hallucination detected", {"step": 1}))
pcl.update_adapter("user123", "lora_adapter_v2")

# --- FederatedRouter Example ---
local = DummyClient("local")
cloud = DummyClient("cloud")
fed_router = FederatedRouter(local, cloud)
print("FederatedRouter (short):", fed_router.generate("short"))
print("FederatedRouter (long):", fed_router.generate("this is a very long prompt" * 20))

# --- RAGFineTuner Example ---
def dummy_rag_pipeline(query): return {"context": f"[Context for: {query}]", "answer": f"[Answer for: {query}]"}
def dummy_fine_tune(train_data, **kwargs): print(f"Fine-tuning on {len(train_data)} examples."); return "fine-tuned-model"
rag_ft = RAGFineTuner(dummy_rag_pipeline, dummy_fine_tune)
queries = ["What is the capital of France?", "Who wrote Hamlet?"]
print("RAGFineTuner result:", rag_ft.auto_ft_from_rag(queries, n_per_query=2)) 