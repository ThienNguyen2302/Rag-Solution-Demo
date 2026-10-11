"""
Vector index implementation using embeddings for semantic search.
"""

import re
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from .base_index import BaseIndex
from .datastructures import DocumentChunk


class VectorIndex(BaseIndex):
    """
    Vector index implementation using embeddings for semantic search.

    Stores document embeddings and uses cosine similarity for retrieval.
    """

    def __init__(self, embedding_dimension: int = 384):
        super().__init__("vector")
        self.embedding_dimension = embedding_dimension
        # Vector storage: chunk_id -> embedding vector
        self.vectors: Dict[str, List[float]] = {}
        # For simplicity, we'll use cosine similarity search
        # In practice, would use FAISS, Annoy, or similar for efficiency

    def _simple_embedding(self, text: str) -> List[float]:
        """Simple embedding function (placeholder for real embedding model)"""
        # This is a very basic embedding for demonstration only
        # Real implementation would use SBERT, BERT, or similar
        embedding = [0.0] * self.embedding_dimension

        # Simple features
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return embedding

        # Length feature
        embedding[0] = min(len(text) / 1000.0, 1.0)

        # Word count feature
        embedding[1] = min(len(words) / 100.0, 1.0)

        # Character distribution features
        for i, char in enumerate(text.lower()[:min(len(text), 50)]):
            if i + 2 < self.embedding_dimension:
                embedding[i + 2] = ord(char) / 128.0

        # Word features (hash-based)
        for i, word in enumerate(words[:20]):  # Limit to first 20 words
            word_hash = hash(word) % 10000
            idx = (i * 50 + word_hash % 50) % self.embedding_dimension
            if idx < self.embedding_dimension:
                embedding[idx] = min((embedding[idx] + 0.1), 1.0)

        # Normalize vector
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add document to vector index"""
        if metadata is None:
            metadata = {}

        # Store document
        self.documents[document_id] = {
            'content': content,
            'metadata': metadata,
            'chunks': []
        }

        # Create chunks (simple fixed-size chunking for demo)
        chunk_id = hashlib.md5(f"{document_id}:vector".encode()).hexdigest()[:12]

        # Generate embedding
        embedding = self._simple_embedding(content)

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            start_index=0,
            end_index=len(content),
            metadata=metadata,
            embedding=embedding
        )

        self.chunks[chunk_id] = chunk
        self.vectors[chunk_id] = embedding
        self.documents[document_id]['chunks'].append(chunk_id)

        # Update stats
        self.stats.total_documents = len(self.documents)
        self.stats.total_chunks = len(self.chunks)
        # Estimate index size (vectors + overhead)
        self.stats.index_size_mb = (len(self.vectors) * self.embedding_dimension * 4) / (1024 * 1024)  # 4 bytes per float

        return [chunk_id]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using cosine similarity"""
        if not self.vectors:
            return []

        # Generate query embedding
        query_embedding = self._simple_embedding(query)

        # Calculate similarities
        similarities: List[Tuple[str, float]] = []
        for chunk_id, embedding in self.vectors.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((chunk_id, similarity))

        # Sort by similarity (descending) and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]

        # Convert to (document_id, score) for consistency with other indexes
        # In a real implementation, you might want to return chunk_ids
        document_scores: Dict[str, float] = {}
        for chunk_id, score in top_results:
            chunk = self.chunks.get(chunk_id)
            if chunk:
                doc_id = chunk.document_id
                if doc_id not in document_scores or score > document_scores[doc_id]:
                    document_scores[doc_id] = score

        # Sort documents by score
        sorted_documents = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_documents[:top_k]

    def delete_document(self, document_id: str) -> bool:
        """Remove document from index"""
        if document_id not in self.documents:
            return False

        # Remove chunks and vectors
        chunk_ids = self.documents[document_id]['chunks']
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
            if chunk_id in self.vectors:
                del self.vectors[chunk_id]

        # Remove document
        del self.documents[document_id]

        # Update stats
        self.stats.total_documents = len(self.documents)
        self.stats.total_chunks = len(self.chunks)
        self.stats.index_size_mb = (len(self.vectors) * self.embedding_dimension * 4) / (1024 * 1024)

        return True