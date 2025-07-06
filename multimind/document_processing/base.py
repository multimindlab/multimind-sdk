"""
Base classes and interfaces for document processing.
"""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

class DocumentProcessingError(Exception):
    """Exception raised for document processing errors."""
    pass

@dataclass
class Document:
    """Represents a document."""
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str

@dataclass
class DocumentConfig:
    """Configuration for document processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    clean_text: bool = True
    custom_params: Dict[str, Any] = None

@runtime_checkable
class DocumentLoader(Protocol):
    """Protocol defining document loader interface."""
    async def load(self, path: Path) -> List[Document]:
        """Load documents from a path."""
        pass

    async def load_batch(self, paths: List[Path]) -> List[Document]:
        """Load multiple documents from paths."""
        pass

class BaseDocumentProcessor(ABC):
    """Base class for document processors."""
    
    def __init__(self, config: Optional[DocumentConfig] = None):
        """Initialize the document processor.
        
        Args:
            config: Configuration for document processing
        """
        self.config = config or DocumentConfig()
    
    @abstractmethod
    async def process(self, document: Document) -> Document:
        """Process a single document.
        
        Args:
            document: Document to process
            
        Returns:
            Processed document
        """
        pass
    
    @abstractmethod
    async def process_batch(self, documents: List[Document]) -> List[Document]:
        """Process multiple documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed documents
        """
        pass
    
    async def validate_document(self, document: Document) -> bool:
        """Validate a document before processing.
        
        Args:
            document: Document to validate
            
        Returns:
            True if document is valid, False otherwise
        """
        return (
            document.id is not None and
            document.content is not None and
            len(document.content.strip()) > 0
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return {
            "config": self.config.__dict__,
            "processor_type": self.__class__.__name__
        }

@runtime_checkable
class DocumentProcessor(Protocol):
    """Protocol defining document processor interface."""
    async def process(self, document: Document) -> Document:
        """Process a document."""
        pass

    async def process_batch(self, documents: List[Document]) -> List[Document]:
        """Process multiple documents."""
        pass

class DocumentType(Enum):
    """Types of documents supported."""
    PDF = "pdf"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    DOCX = "docx"
    CSV = "csv"
    JSON = "json" 