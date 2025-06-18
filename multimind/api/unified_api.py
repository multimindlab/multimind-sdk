"""
Unified API endpoint for multi-modal processing with MoE support.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import asyncio
from ..router.multi_modal_router import MultiModalRouter, MultiModalRequest
from ..models.moe import MoEFactory
from ..mcp.api.registry import WorkflowRegistry

app = FastAPI(title="Unified Multi-Modal API")

class ModalityInput(BaseModel):
    """Input for a specific modality."""
    content: Any
    modality: str

class UnifiedRequest(BaseModel):
    """Unified request structure for multi-modal processing."""
    inputs: List[ModalityInput]
    use_moe: bool = Field(default=True, description="Whether to use MoE processing")
    constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Processing constraints (cost, latency, etc.)"
    )
    workflow: Optional[str] = Field(
        default=None,
        description="Optional MCP workflow to use"
    )

class UnifiedResponse(BaseModel):
    """Unified response structure."""
    outputs: Dict[str, Any]
    expert_weights: Optional[Dict[str, float]] = None
    metrics: Dict[str, Any]

# Initialize components
router = MultiModalRouter()
moe_factory = MoEFactory()
workflow_registry = WorkflowRegistry()

@app.post("/v1/process", response_model=UnifiedResponse)
async def process_request(request: UnifiedRequest):
    """Process multi-modal request using either MoE or router."""
    try:
        # Convert inputs to router format
        content = {
            input.modality: input.content
            for input in request.inputs
        }
        modalities = [input.modality for input in request.inputs]
        
        if request.use_moe:
            # Use MoE processing
            moe_config = {
                "experts": {
                    modality: {"model": router.modality_registry[modality]}
                    for modality in modalities
                    if modality in router.modality_registry
                }
            }
            moe_model = moe_factory.create_moe_model(moe_config)
            
            # Process through MoE
            result = await moe_model.process(content)
            
            return UnifiedResponse(
                outputs=result["output"],
                expert_weights=result["expert_weights"],
                metrics={
                    "processing_type": "moe",
                    "num_experts": len(moe_config["experts"])
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
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/v1/models")
async def list_models():
    """List available models and their capabilities."""
    models = {}
    for modality, model_dict in router.modality_registry.items():
        models[modality] = list(model_dict.keys())
    return {"models": models}

@app.get("/v1/workflows")
async def list_workflows():
    """List available MCP workflows."""
    return {"workflows": workflow_registry.list_workflows()}

@app.get("/v1/metrics")
async def get_metrics():
    """Get performance metrics for models."""
    return {
        "costs": router.cost_tracker.costs,
        "performance": router.performance_metrics.metrics
    } 