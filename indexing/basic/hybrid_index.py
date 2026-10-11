"""
Hybrid index combining inverted and vector indexes for best of both worlds.
"""

from typing import List, Dict, Any, Optional, Tuple
from .inverted_index import InvertedIndex
from .vector_index import VectorIndex
from .base_index import BaseIndex
from .datastructures import DocumentChunk


class HybridIndex(BaseIndex):
    """
    Hybrid index combining inverted and vector indexes for best of both worlds.
    """

    def __init__(self, embedding_dimension: int = 384):
        super().__init__("hybrid")
        self.inverted_index = InvertedIndex()
        self.vector_index = VectorIndex(embedding_dimension)
        self.alpha = 0.5  # Weight for combining scores (0.0 = pure inverted, 1.0 = pure vector)

    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add document to both indexes"""
        if metadata is None:
            metadata = {}

        # Add to both indexes
        inv_chunk_ids = self.inverted_index.add_document(document_id, content, metadata)
        vec_chunk_ids = self.vector_index.add_document(document_id, content, metadata)

        # Store document info
        self.documents[document_id] = {
            'content': content,
            'metadata': metadata,
            'chunks': inv_chunk_ids  # Use inverted index chunks as primary
        }

        # Update chunks dictionary (merge from both)
        self.chunks.update(self.inverted_index.chunks)
        self.chunks.update(self.vector_index.chunks)

        # Update stats
        self._update_stats()

        return inv_chunk_ids  # Return inverted index chunk IDs

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using hybrid approach (combined scores)"""
        # Get results from both indexes
        inv_results = self.inverted_index.search(query, top_k * 2)  # Get more to allow for re-ranking
        vec_results = self.vector_index.search(query, top_k * 2)

        # Combine scores
        combined_scores: Dict[str, float] = {}

        # Add inverted index scores (normalized)
        if inv_results:
            max_inv_score = max(score for _, score in inv_results) if inv_results else 1.0
            for doc_id, score in inv_results:
                normalized_score = score / max_inv_score if max_inv_score > 0 else 0
                combined_scores[doc_id] = (1 - self.alpha) * normalized_score

        # Add vector index scores (normalized)
        if vec_results:
            max_vec_score = max(score for _, score in vec_results) if vec_results else 1.0
            for doc_id, score in vec_results:
                normalized_score = score / max_vec_score if max_vec_score > 0 else 0
                if doc_id in combined_scores:
                    combined_scores[doc_id] += self.alpha * normalized_score
                else:
                    combined_scores[doc_id] = self.alpha * normalized_score

        # Sort and return top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def delete_document(self, document_id: str) -> bool:
        """Remove document from both indexes"""
        inv_result = self.inverted_index.delete_document(document_id)
        vec_result = self.vector_index.delete_document(document_id)

        # Remove from our documents dict
        if document_id in self.documents:
            del self.documents[document_id]

        # Update chunks (rebuild from indexes)
        self.chunks.clear()
        self.chunks.update(self.inverted_index.chunks)
        self.chunks.update(self.vector_index.chunks)

        # Update stats
        self._update_stats()

        return inv_result and vec_result

    def _update_stats(self):
        """Update index statistics"""
        self.stats.total_documents = len(self.documents)
        self.stats.total_chunks = len(self.chunks)
        self.stats.total_tokens = sum(len(chunk.tokens) for chunk in self.chunks.values() if chunk.tokens)
        if self.stats.total_chunks > 0:
            self.stats.average_chunk_length = self.stats.total_tokens / self.stats.total_chunks
        self.stats.vocabulary_size = len(self.inverted_index.inverted_index)
        self.stats.index_size_mb = (
            (len(self.vector_index.vectors) * self.vector_index.embedding_dimension * 4) / (1024 * 1024)
        )