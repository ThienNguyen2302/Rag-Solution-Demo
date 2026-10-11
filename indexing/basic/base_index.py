"""
Abstract base class for all index types.
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from .datastructures import DocumentChunk, IndexStats


class BaseIndex(ABC):
    """Abstract base class for all index types"""

    def __init__(self, index_type):
        self.index_type = index_type
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: Dict[str, DocumentChunk] = {}
        self.stats = IndexStats()

    @abstractmethod
    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add a document to the index and return chunk IDs"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search the index and return (chunk_id, score) tuples"""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Remove a document from the index"""
        pass

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata"""
        return self.documents.get(document_id)

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a chunk by ID"""
        return self.chunks.get(chunk_id)

    def get_stats(self) -> IndexStats:
        """Get index statistics"""
        return self.stats