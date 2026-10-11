"""
Abstract base class for ColBERT-style embedders.
"""

from typing import List
from abc import ABC, abstractmethod


class BaseColBERTEmbedder(ABC):
    """Abstract base class for ColBERT-style embedders"""

    @abstractmethod
    def embed_tokens(self, texts: List[str]) -> List[List[dict]]:
        """Embed texts into sequences of token embeddings"""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        pass