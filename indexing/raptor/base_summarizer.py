"""
Abstract base class for summarization models used in RAPTOR.
"""

from typing import List, Optional
from abc import ABC, abstractmethod


class BaseSummarizer(ABC):
    """Abstract base class for summarization models"""

    @abstractmethod
    def summarize(self, texts: List[str], max_length: Optional[int] = None) -> List[str]:
        """Summarize a list of texts"""
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass