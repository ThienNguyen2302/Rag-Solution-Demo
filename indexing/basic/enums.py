"""
Enumerations for basic indexing techniques.
"""

from enum import Enum


class IndexType(Enum):
    """Types of indexes supported"""
    INVERTED = "inverted"      # Keyword-based inverted index
    VECTOR = "vector"          # Embedding-based vector index
    HYBRID = "hybrid"          # Combination of inverted and vector
    GRAPH = "graph"            # Graph-based index (for relationships)
    METADATA = "metadata"      # Metadata-focused index


class ChunkingStrategy(Enum):
    """Document chunking strategies"""
    FIXED_SIZE = "fixed_size"      # Fixed number of words/tokens
    SENTENCE_BASED = "sentence"    # Split by sentences
    PARAGRAPH_BASED = "paragraph"  # Split by paragraphs
    SEMANTIC = "semantic"          # Semantic-based chunking (requires model)
    RECURSIVE = "recursive"        # Recursively split by separators
    SLIDING_WINDOW = "sliding_window"  # Overlapping windows