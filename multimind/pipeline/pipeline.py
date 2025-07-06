"""
Pipeline system for building and executing complex workflows.
"""

from typing import Dict, List, Optional, Union, Any, Callable, TypeVar, Generic
from pydantic import BaseModel, ConfigDict
from enum import Enum
import asyncio
from ..core.router import Router, TaskType
from ..core.provider import GenerationResult, EmbeddingResult, ImageAnalysisResult

T = TypeVar('T')
R = TypeVar('R')

class StageType(Enum):
    """Types of pipeline stages."""
    EMBED = "embed"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    ANALYZE = "analyze"
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATE = "aggregate"

class StageConfig(BaseModel):
    """Configuration for a pipeline stage."""
    type: StageType
    provider: Optional[str] = None
    model: Optional[str] = None
    parameters: Dict[str, Any] = {}
    error_handler: Optional[Callable] = None
    retry_count: int = 0
    timeout: Optional[float] = None

class StageResult(BaseModel):
    """Result from a pipeline stage."""
    stage_type: StageType
    output: Any
    metadata: Dict[str, Any] = {}
    error: Optional[Exception] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class PipelineStage(Generic[T, R]):
    """Represents a stage in the pipeline."""
    
    def __init__(
        self,
        stage_type: StageType,
        handler: Callable[[T], R],
        config: Optional[StageConfig] = None
    ):
        self.stage_type = stage_type
        self.handler = handler
        self.config = config or StageConfig(type=stage_type)
        self.next_stage: Optional['PipelineStage'] = None
    
    async def execute(self, input_data: T) -> R:
        """Execute the stage with retry and error handling."""
        for attempt in range(self.config.retry_count + 1):
            try:
                if self.config.timeout:
                    result = await asyncio.wait_for(
                        self.handler(input_data),
                        timeout=self.config.timeout
                    )
                else:
                    result = await self.handler(input_data)
                return result
            except Exception as e:
                if attempt == self.config.retry_count:
                    if self.config.error_handler:
                        return await self.config.error_handler(e, input_data)
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

class Pipeline:
    """Main pipeline class for building and executing workflows."""
    
    def __init__(self, router: Router):
        """Initialize the pipeline with a router."""
        self.router = router
        self.stages: List[PipelineStage] = []
        self.context: Dict[str, Any] = {}
    
    def stage(
        self,
        stage_type: StageType,
        handler: Optional[Callable] = None,
        **config
    ) -> 'Pipeline':
        """Add a stage to the pipeline."""
        if handler is None:
            handler = self._get_default_handler(stage_type)
        
        stage_config = StageConfig(type=stage_type, **config)
        stage = PipelineStage(stage_type, handler, stage_config)
        
        if self.stages:
            self.stages[-1].next_stage = stage
        self.stages.append(stage)
        
        return self
    
    def _get_default_handler(self, stage_type: StageType) -> Callable:
        """Get the default handler for a stage type."""
        handlers = {
            StageType.EMBED: self._default_embed_handler,
            StageType.RETRIEVE: self._default_retrieve_handler,
            StageType.GENERATE: self._default_generate_handler,
            StageType.ANALYZE: self._default_analyze_handler,
            StageType.TRANSFORM: self._default_transform_handler,
            StageType.FILTER: self._default_filter_handler,
            StageType.AGGREGATE: self._default_aggregate_handler
        }
        return handlers[stage_type]
    
    async def _default_embed_handler(self, input_data: str) -> EmbeddingResult:
        """Default handler for embedding stage."""
        return await self.router.route(
            TaskType.EMBEDDINGS,
            input_data,
            model="text-embedding-ada-002"
        )
    
    async def _default_retrieve_handler(self, input_data: List[float]) -> List[Dict[str, Any]]:
        """Default handler for retrieve stage."""
        # This would typically interact with a vector DB
        # For now, return empty list
        return []
    
    async def _default_generate_handler(self, input_data: Dict[str, Any]) -> GenerationResult:
        """Default handler for generate stage."""
        return await self.router.route(
            TaskType.TEXT_GENERATION,
            input_data.get("prompt", ""),
            model=input_data.get("model", "gpt-3.5-turbo")
        )
    
    async def _default_analyze_handler(self, input_data: bytes) -> ImageAnalysisResult:
        """Default handler for analyze stage."""
        return await self.router.route(
            TaskType.IMAGE_ANALYSIS,
            input_data,
            prompt="Analyze this image in detail."
        )
    
    async def _default_transform_handler(self, input_data: Any) -> Any:
        """Default handler for transform stage."""
        return input_data
    
    async def _default_filter_handler(self, input_data: List[Any]) -> List[Any]:
        """Default handler for filter stage."""
        return input_data
    
    async def _default_aggregate_handler(self, input_data: List[Any]) -> Any:
        """Default handler for aggregate stage."""
        return input_data
    
    async def run(self, input_data: Any) -> Any:
        """Execute the pipeline with the given input."""
        if not self.stages:
            return input_data
        
        current_input = input_data
        for stage in self.stages:
            try:
                current_input = await stage.execute(current_input)
            except Exception as e:
                if stage.config.error_handler:
                    current_input = await stage.config.error_handler(e, current_input)
                else:
                    raise
        
        return current_input
    
    def set_context(self, key: str, value: Any) -> 'Pipeline':
        """Set a value in the pipeline context."""
        self.context[key] = value
        return self
    
    def get_context(self, key: str) -> Any:
        """Get a value from the pipeline context."""
        return self.context.get(key)

class PipelineBuilder:
    """Builder class for creating pre-built pipelines."""
    
    def __init__(self, router: Router):
        """Initialize the pipeline builder with a router."""
        self.router = router
    
    def qa_retrieval(self) -> Pipeline:
        """Create a QA retrieval pipeline."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.EMBED,
            provider="openai",
            model="text-embedding-ada-002"
        ).stage(
            StageType.RETRIEVE,
            parameters={"top_k": 5}
        ).stage(
            StageType.GENERATE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "answer_with_context"}
        )
    
    def code_review(self) -> Pipeline:
        """Create a code review pipeline."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.ANALYZE,
            provider="openai",
            model="gpt-4"
        ).stage(
            StageType.GENERATE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "code_review"}
        )
    
    def image_analysis(self) -> Pipeline:
        """Create an image analysis pipeline."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.ANALYZE,
            provider="openai",
            model="gpt-4-vision-preview"
        ).stage(
            StageType.GENERATE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "image_description"}
        )
    
    def text_summarization(self) -> Pipeline:
        """Create a text summarization pipeline."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.GENERATE,
            provider="openai",
            model="gpt-3.5-turbo",
            parameters={"template": "summarize"}
        ).stage(
            StageType.TRANSFORM,
            parameters={"format": "bullet_points"}
        )
    
    def content_generation(self) -> Pipeline:
        """Create a content generation pipeline with SEO optimization."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.GENERATE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "content_outline"}
        ).stage(
            StageType.TRANSFORM,
            parameters={"format": "markdown"}
        ).stage(
            StageType.GENERATE,
            provider="openai",
            model="gpt-4",
            parameters={"template": "seo_optimize"}
        )
    
    def data_analysis(self) -> Pipeline:
        """Create a data analysis pipeline with visualization suggestions."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.ANALYZE,
            provider="openai",
            model="gpt-4",
            parameters={"template": "data_analysis"}
        ).stage(
            StageType.GENERATE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "visualization_suggestions"}
        ).stage(
            StageType.TRANSFORM,
            parameters={"format": "json"}
        )
    
    def multi_modal_qa(self) -> Pipeline:
        """Create a multi-modal QA pipeline that can handle both text and images."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.ANALYZE,
            provider="openai",
            model="gpt-4-vision-preview"
        ).stage(
            StageType.EMBED,
            provider="openai",
            model="text-embedding-ada-002"
        ).stage(
            StageType.RETRIEVE,
            parameters={"top_k": 3}
        ).stage(
            StageType.GENERATE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "multi_modal_answer"}
        )
    
    def code_generation(self) -> Pipeline:
        """Create a code generation pipeline with testing and documentation."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.GENERATE,
            provider="openai",
            model="gpt-4",
            parameters={"template": "code_generation"}
        ).stage(
            StageType.TRANSFORM,
            parameters={"format": "python"}
        ).stage(
            StageType.GENERATE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "generate_tests"}
        ).stage(
            StageType.GENERATE,
            provider="openai",
            model="gpt-3.5-turbo",
            parameters={"template": "generate_docs"}
        )
    
    def sentiment_analysis(self) -> Pipeline:
        """Create a sentiment analysis pipeline with aspect extraction."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.ANALYZE,
            provider="openai",
            model="gpt-3.5-turbo",
            parameters={"template": "sentiment_analysis"}
        ).stage(
            StageType.TRANSFORM,
            parameters={"format": "json"}
        ).stage(
            StageType.GENERATE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "aspect_extraction"}
        )
    
    def document_processing(self) -> Pipeline:
        """Create a document processing pipeline with entity extraction and summarization."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.ANALYZE,
            provider="openai",
            model="gpt-4",
            parameters={"template": "entity_extraction"}
        ).stage(
            StageType.TRANSFORM,
            parameters={"format": "json"}
        ).stage(
            StageType.GENERATE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "document_summary"}
        )
    
    def translation_pipeline(self) -> Pipeline:
        """Create a translation pipeline with style preservation and cultural adaptation."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.ANALYZE,
            provider="openai",
            model="gpt-4",
            parameters={"template": "style_analysis"}
        ).stage(
            StageType.GENERATE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "translation"}
        ).stage(
            StageType.TRANSFORM,
            parameters={"format": "text"}
        ).stage(
            StageType.GENERATE,
            provider="openai",
            model="gpt-3.5-turbo",
            parameters={"template": "cultural_adaptation"}
        )
    
    def research_assistant(self) -> Pipeline:
        """Create a research assistant pipeline with literature review and synthesis."""
        pipeline = Pipeline(self.router)
        return pipeline.stage(
            StageType.EMBED,
            provider="openai",
            model="text-embedding-ada-002"
        ).stage(
            StageType.RETRIEVE,
            parameters={"top_k": 10}
        ).stage(
            StageType.ANALYZE,
            provider="claude",
            model="claude-3-sonnet",
            parameters={"template": "literature_analysis"}
        ).stage(
            StageType.GENERATE,
            provider="openai",
            model="gpt-4",
            parameters={"template": "research_synthesis"}
        ) 