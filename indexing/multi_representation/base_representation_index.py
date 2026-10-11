"""
Abstract base class for representation-specific indexes.
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from .datastructures import MultiRepresentationResult


class BaseRepresentationIndex(ABC):
    """Abstract base class for representation-specific indexes"""

    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.index = None  # Will be set to specific index implementation
        self.is_built = False

    @abstractmethod
    def build_index(self, documents: Dict[str, str],
                   metadatas: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Build the index from documents"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search the representation index"""
        pass

    @abstractmethod
    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add a single document"""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        pass

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        return self.documents.get(document_id)