"""
Splitter module for MultiMind SDK.

This module provides text splitting capabilities for document processing.
"""

from typing import List, Dict, Any, Optional
import re

class TextSplitter:
    """Basic text splitter for document processing."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
            if start >= len(text):
                break
        
        return chunks
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

class DocumentSplitter:
    """Advanced document splitter with metadata preservation."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.text_splitter = TextSplitter(chunk_size, overlap)
    
    def split_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a document into chunks with metadata."""
        content = document.get('content', '')
        chunks = self.text_splitter.split_text(content)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_doc = document.copy()
            chunk_doc['content'] = chunk
            chunk_doc['chunk_id'] = i
            chunk_doc['total_chunks'] = len(chunks)
            result.append(chunk_doc)
        
        return result

__all__ = [
    "TextSplitter",
    "DocumentSplitter"
] 