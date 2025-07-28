"""
Unified API endpoint for multi-modal processing with MoE support.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import asyncio
from ..models.moe import MoEFactory

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
try:
    moe_factory = MoEFactory()
except ImportError:
    moe_factory = None

@app.post("/v1/process", response_model=UnifiedResponse)
async def process_request(request: UnifiedRequest):
    """Process multi-modal request using either MoE or router."""
    try:
        # Import here to avoid circular imports
        from ..router.multi_modal_router import MultiModalRouter, MultiModalRequest
        from .mcp.registry import WorkflowRegistry
        
        # Initialize components
        router = MultiModalRouter()
        workflow_registry = WorkflowRegistry()
        
        # Convert inputs to router format
        content = {
            input.modality: input.content
            for input in request.inputs
        }
        modalities = [input.modality for input in request.inputs]
        
        if request.use_moe:
            # Use MoE processing
            if moe_factory is None:
                raise HTTPException(
                    status_code=400,
                    detail="MoE processing is not available. PyTorch is required for MoE features."
                )
            
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
    # Import here to avoid circular imports
    from ..router.multi_modal_router import MultiModalRouter
    router = MultiModalRouter()
    
    models = {}
    for modality, model_dict in router.modality_registry.items():
        models[modality] = list(model_dict.keys())
    return {"models": models}

@app.get("/v1/workflows")
async def list_workflows():
    """List available MCP workflows."""
    # Import here to avoid circular imports
    from .mcp.registry import WorkflowRegistry
    workflow_registry = WorkflowRegistry()
    return {"workflows": workflow_registry.list_workflows()}

@app.get("/v1/metrics")
async def get_metrics():
    """Get performance metrics for models."""
    # Import here to avoid circular imports
    from ..router.multi_modal_router import MultiModalRouter
    router = MultiModalRouter()
    
    return {
        "costs": router.cost_tracker.costs,
        "performance": router.performance_metrics.metrics
    } 