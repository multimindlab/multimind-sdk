"""
Base classes and interfaces for document processing.
"""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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