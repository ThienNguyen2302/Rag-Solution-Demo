"""
Data structures for ColBERT technique.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ColBERTTokenEmbedding:
    """Represents a token embedding in ColBERT"""
    token_id: int
    token_text: str
    embedding: List[float]  # Contextualized embedding vector
    start_pos: int          # Character start position in original text
    end_pos: int            # Character end position in original text


@dataclass
class ColBERTVector:
    """Represents a ColBERT vector (sequence of token embeddings)"""
    vector_id: str
    tokens: List[ColBERTTokenEmbedding]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ColBERTIndexStats:
    """Statistics about the ColBERT index"""
    total_documents: int = 0
    total_vectors: int = 0
    total_tokens: int = 0
    avg_tokens_per_vector: float = 0.0
    embedding_dimension: int = 0
    index_size_mb: float = 0.0