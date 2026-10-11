"""
Data structures for basic indexing techniques.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import hashlib
from abc import ABC, abstractmethod


@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    chunk_id: str
    document_id: str
    content: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'content': self.content,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'metadata': self.metadata,
            'embedding': self.embedding,
            'tokens': self.tokens
        }


@dataclass
class IndexStats:
    """Statistics about the index"""
    total_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    average_chunk_length: float = 0.0
    vocabulary_size: int = 0
    index_size_mb: float = 0.0
    creation_time: float = 0.0