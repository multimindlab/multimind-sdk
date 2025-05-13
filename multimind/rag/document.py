"""
Document processing utilities for RAG system.
"""

from typing import List, Dict, Any, Optional, Union
import re
from dataclasses import dataclass
import tiktoken
from pathlib import Path

@dataclass
class Document:
    """A document with text content and metadata."""
    
    text: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate document after initialization."""
        if not isinstance(self.text, str):
            raise ValueError("Document text must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("Document metadata must be a dictionary")

class DocumentProcessor:
    """Process documents for RAG system."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer: Optional[str] = None
    ):
        """Initialize document processor.
        
        Args:
            chunk_size: Maximum size of text chunks in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            tokenizer: Name of tokenizer to use (default: cl100k_base for GPT models)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer or "cl100k_base")
        
    def _count_tokens(self, text: str) -> int:
        """Count number of tokens in text."""
        return len(self.tokenizer.encode(text))
        
    def _split_text(
        self,
        text: str,
        separator: str = "\n"
    ) -> List[str]:
        """Split text into chunks based on separator."""
        # Split by separator
        segments = text.split(separator)
        
        # Process segments
        chunks = []
        current_chunk = []
        current_size = 0
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            segment_size = self._count_tokens(segment)
            
            # If segment is too large, split it further
            if segment_size > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                    
                # Split large segment into smaller chunks
                words = segment.split()
                temp_chunk = []
                temp_size = 0
                
                for word in words:
                    word_size = self._count_tokens(word + " ")
                    if temp_size + word_size > self.chunk_size:
                        if temp_chunk:
                            chunks.append(" ".join(temp_chunk))
                        temp_chunk = [word]
                        temp_size = word_size
                    else:
                        temp_chunk.append(word)
                        temp_size += word_size
                        
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                continue
                
            # Add segment to current chunk if it fits
            if current_size + segment_size <= self.chunk_size:
                current_chunk.append(segment)
                current_size += segment_size
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                current_chunk = [segment]
                current_size = segment_size
                
        # Add final chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))
            
        return chunks
        
    def process_document(
        self,
        document: Union[str, Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process a document into chunks.
        
        Args:
            document: Document text or Document object
            metadata: Optional metadata to add to chunks
            
        Returns:
            List of Document chunks
        """
        # Handle input types
        if isinstance(document, str):
            text = document
            doc_metadata = metadata or {}
        else:
            text = document.text
            doc_metadata = {**document.metadata, **(metadata or {})}
            
        # Clean text
        text = self._clean_text(text)
        
        # Split into chunks
        chunks = self._split_text(text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **doc_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            documents.append(Document(text=chunk, metadata=chunk_metadata))
            
        return documents
        
    def process_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process a file into document chunks.
        
        Args:
            file_path: Path to file
            metadata: Optional metadata to add to chunks
            
        Returns:
            List of Document chunks
            
        Raises:
            ValueError: If file type is not supported
        """
        file_path = Path(file_path)
        
        # Read file based on extension
        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_path.suffix == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_path.suffix == ".pdf":
            try:
                import PyPDF2
            except ImportError:
                raise ImportError(
                    "PyPDF2 is required for PDF processing. "
                    "Install with: pip install PyPDF2"
                )
                
            text = ""
            with open(file_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        # Add file metadata
        file_metadata = {
            "source": str(file_path),
            "file_type": file_path.suffix[1:],
            "file_name": file_path.name
        }
        if metadata:
            file_metadata.update(metadata)
            
        return self.process_document(text, file_metadata)
        
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text by removing extra whitespace and normalizing."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Normalize newlines
        text = re.sub(r"\n+", "\n", text)
        
        # Remove special characters
        text = re.sub(r"[^\w\s.,!?-]", "", text)
        
        return text.strip() 