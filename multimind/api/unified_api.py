"""
Unified API endpoint for multi-modal processing with MoE support.
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import asyncio
import logging
import os
import base64
import io
from ..models.base import BaseLLM
from ..models.factory import ModelFactory
from ..models.moe import Expert
from ..types import UnifiedRequest, UnifiedResponse, ModalityInput

logger = logging.getLogger(__name__)

app = FastAPI(title="Unified Multi-Modal API")

API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []


def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> bool:
    """Verify the API key from request header."""
    if not API_KEYS:
        return True
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Reuse a single factory across requests to avoid re-creating model caches.
_MODEL_FACTORY = ModelFactory()

_ROUTER = None
_WORKFLOW_REGISTRY = None


def _get_router():
    global _ROUTER
    if _ROUTER is None:
        from ..router.multi_modal_router import MultiModalRouter
        _ROUTER = MultiModalRouter()
    return _ROUTER


def _get_workflow_registry():
    global _WORKFLOW_REGISTRY
    if _WORKFLOW_REGISTRY is None:
        from .mcp.registry import WorkflowRegistry
        _WORKFLOW_REGISTRY = WorkflowRegistry()
    return _WORKFLOW_REGISTRY


class _TextExpertAdapter(Expert):
    """Expert wrapper around a model instance for text."""

    def __init__(self, expert_id: str, model: BaseLLM, **kwargs):
        super().__init__(expert_id, **kwargs)
        self.model = model

    async def process(self, input_data: Any) -> Any:
        text = input_data.get("text", "") if isinstance(input_data, dict) else input_data
        if isinstance(text, list):
            text = "\n".join(str(t) for t in text)
        return await self.model.generate(str(text))


class _ImageExpertAdapter(Expert):
    """Expert wrapper for image analysis/captioning."""

    def __init__(self, expert_id: str, provider: Any, model: str, default_prompt: str = "Describe this image", **kwargs):
        super().__init__(expert_id, **kwargs)
        self.provider = provider
        self.model = model
        self.default_prompt = default_prompt

    async def process(self, input_data: Any) -> Any:
        # input_data may be {"image": [...], "text": [...]} or just a list of base64 images
        if isinstance(input_data, dict):
            images = input_data.get("image") or []
            prompt = input_data.get("text") or self.default_prompt
        else:
            images = input_data
            prompt = self.default_prompt

        if isinstance(images, str):
            images = [images]
        if not isinstance(images, list) or not images or not images[0]:
            return "No image input provided."

        if isinstance(prompt, list):
            prompt = "\n".join(str(p) for p in prompt) if prompt else self.default_prompt
        prompt = str(prompt) if prompt else self.default_prompt

        try:
            image_bytes = base64.b64decode(images[0])
        except Exception as e:
            return f"Invalid image base64: {e}"

        result = await self.provider.analyze_image(image_bytes, prompt=prompt, model=self.model)
        if getattr(result, "text", None):
            return result.text
        captions = getattr(result, "captions", None)
        if captions:
            return captions[0]
        return "Image analyzed, but no caption produced."


class _AudioExpertAdapter(Expert):
    """Expert wrapper for audio transcription (OpenAI Whisper when configured)."""

    def __init__(self, expert_id: str, openai_client: Any, model: str = "whisper-1", **kwargs):
        super().__init__(expert_id, **kwargs)
        self.client = openai_client
        self.model = model

    async def process(self, input_data: Any) -> Any:
        # input_data may be {"audio": [...], "text": [...]} or just a list of base64 audio blobs
        if isinstance(input_data, dict):
            audios = input_data.get("audio") or []
        else:
            audios = input_data

        if isinstance(audios, str):
            audios = [audios]
        if not isinstance(audios, list) or not audios or not audios[0]:
            return "No audio input provided."

        try:
            audio_bytes = base64.b64decode(audios[0])
        except Exception as e:
            return f"Invalid audio base64: {e}"

        if self.client is None:
            return "Audio transcription not configured (missing OPENAI_API_KEY)."

        bio = io.BytesIO(audio_bytes)
        bio.name = "audio.mp3"
        resp = await self.client.audio.transcriptions.create(
            model=self.model,
            file=bio
        )
        return getattr(resp, "text", None) or str(resp)


def _build_experts(modalities: List[str], router: Any) -> Dict[str, Expert]:
    """Build available experts for modality MoE."""
    experts: Dict[str, Expert] = {}

    for modality in modalities:
        model = None
        model_map = router.modality_registry.get(modality, {})
        if model_map:
            model = next(iter(model_map.values()), None)

        if model is None and modality == "text":
            for provider in _MODEL_FACTORY.available_models():
                try:
                    model = _MODEL_FACTORY.get_model(provider)
                    break
                except Exception:
                    continue

        if modality == "text" and isinstance(model, BaseLLM):
            experts["text_expert"] = _TextExpertAdapter("text_expert", model)

    # Image/audio experts from providers (independent of router registry).
    try:
        from ..core.provider import ProviderConfig
    except Exception:
        ProviderConfig = None

    if ProviderConfig is not None and "image" in modalities:
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from ..providers.openai import OpenAIProvider
                provider = OpenAIProvider(ProviderConfig(api_key=openai_key))
                experts["image_expert"] = _ImageExpertAdapter("image_expert", provider=provider, model="gpt-4o-mini")
            else:
                from ..providers.ollama import OllamaProvider
                provider = OllamaProvider(ProviderConfig(api_base=os.getenv("OLLAMA_BASE_URL")))
                experts["image_expert"] = _ImageExpertAdapter("image_expert", provider=provider, model="llava-phi3:latest")
        except Exception:
            pass

    if "audio" in modalities:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                client = openai.AsyncOpenAI(api_key=openai_key)
                experts["audio_expert"] = _AudioExpertAdapter("audio_expert", openai_client=client)
            except Exception:
                pass

    return experts

@app.post("/v1/process", response_model=UnifiedResponse)
async def process_request(request: UnifiedRequest, authenticated: bool = Depends(verify_api_key)):
    """Process multi-modal request using either MoE or router."""
    try:
        from ..router.multi_modal_router import MultiModalRequest

        router = _get_router()
        workflow_registry = _get_workflow_registry()
        
        # Convert inputs to router format (support multiple inputs per modality)
        content: Dict[str, Any] = {}
        for inp in request.inputs:
            content.setdefault(inp.modality, []).append(inp.content)
        modalities = [input.modality for input in request.inputs]
        
        if request.use_moe:
            # Strict MoE path: do not use router fallback in this branch.
            experts = _build_experts(modalities, router)
            if not experts:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "MoE requested but no experts available. "
                        "Check model registration/API keys for required modalities."
                    ),
                )

            from ..models.moe.unified_moe import UnifiedMoE
            moe_model = UnifiedMoE(mode="modality", experts=experts)
            result = await moe_model.process(content)

            expert_outputs = result.get("expert_outputs") or {}
            routing_weights = result.get("routing_weights") or {}

            # Pull out per-expert text (for transparency)
            image_text = None
            audio_text = None
            if isinstance(expert_outputs, dict):
                img = expert_outputs.get("image_expert")
                aud = expert_outputs.get("audio_expert")
                if isinstance(img, dict):
                    image_text = img.get("output")
                if isinstance(aud, dict):
                    audio_text = aud.get("output")

            # If a text prompt exists and we have a text expert, use it for final synthesis
            prompt_list = content.get("text") if isinstance(content, dict) else None
            prompt = None
            if isinstance(prompt_list, list) and prompt_list:
                prompt = "\n".join(str(p) for p in prompt_list)

            synthesized_text = None
            if prompt and "text_expert" in experts and (image_text or audio_text):
                ctx_parts = []
                if image_text:
                    ctx_parts.append(f"IMAGE:\n{image_text}")
                if audio_text:
                    ctx_parts.append(f"AUDIO:\n{audio_text}")
                ctx_joined = "\n\n".join(ctx_parts)
                synthesis_prompt = (
                    f"{prompt}\n\n"
                    f"Context from other experts:\n\n{ctx_joined}\n\n"
                    "Using the context above, produce the best final answer."
                )
                synthesized_text = await experts["text_expert"].process({"text": synthesis_prompt})

            # Final outputs shape used by examples
            final_text = synthesized_text if synthesized_text else (result.get("output") if isinstance(result.get("output"), str) else str(result.get("output")))
            outputs: Dict[str, Any] = {"text": final_text}
            if image_text is not None:
                outputs["image_text"] = image_text
            if audio_text is not None:
                outputs["audio_text"] = audio_text

            return UnifiedResponse(
                outputs=outputs,
                expert_weights=routing_weights,
                metrics={
                    "processing_type": "moe",
                    "num_experts": len(experts),
                    "expert_outputs": expert_outputs,
                    "text_synthesis_used": synthesized_text is not None,
                    **result.get("metrics", {})
                }
            )
        else:
            # Use router-based processing
            router_request = MultiModalRequest(
                content=content,
                modalities=modalities,
                constraints=request.constraints
            )
            
            if request.workflow:
                # Use MCP workflow
                workflow = workflow_registry.get_workflow(request.workflow)
                result = await workflow.execute(router_request)
            else:
                # Use direct routing
                result = await router.route_request(router_request)
            
            return UnifiedResponse(
                outputs=result,
                metrics={
                    "processing_type": "router",
                    "workflow": request.workflow
                }
            )
            
    except HTTPException:
        # Preserve intended HTTP status codes (e.g., 400 for invalid input).
        raise
    except Exception as e:
        logger.exception("Error processing request")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/v1/models")
async def list_models(authenticated: bool = Depends(verify_api_key)):
    """List available models and their capabilities."""
    router = _get_router()

    models = {}
    for modality, model_dict in router.modality_registry.items():
        models[modality] = list(model_dict.keys())
    return {"models": models}

@app.get("/v1/workflows")
async def list_workflows(authenticated: bool = Depends(verify_api_key)):
    """List available MCP workflows."""
    workflow_registry = _get_workflow_registry()
    return {"workflows": workflow_registry.list_workflows()}

@app.get("/v1/metrics")
async def get_metrics(authenticated: bool = Depends(verify_api_key)):
    """Get performance metrics for models."""
    router = _get_router()

    return {
        "costs": router.cost_tracker.costs,
        "performance": router.performance_metrics.metrics
    } 