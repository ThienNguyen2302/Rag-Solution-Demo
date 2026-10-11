"""
ColBERT index implementation for efficient and effective retrieval
using late interaction between contextualized token embeddings.
"""

from typing import List, Dict, Any, Optional, Tuple
from .enums import MatchScoreType
from .datastructures import ColBERTTokenEmbedding, ColBERTVector, ColBERTIndexStats
from .base_embedder import BaseColBERTEmbedder
import re
import math
import hashlib
from collections import defaultdict


class ColBERTIndex:
    """
    ColBERT index implementation for efficient and effective retrieval
    using late interaction between contextualized token embeddings.
    """

    def __init__(self,
                 embedder: Optional[BaseColBERTEmbedder] = None,
                 embedding_dimension: int = 128,
                 score_type: MatchScoreType = MatchScoreType.MAXSIM,
                 doc_maxlen: int = 180,
                 query_maxlen: int = 32,
                 dim: int = 64):
        """
        Initialize ColBERT index.

        Args:
            embedder: ColBERT-style embedder (defaults to SimpleColBERTEmbedder)
            embedding_dimension: Dimension of token embeddings
            score_type: Type of matching score to use
            doc_maxlen: Maximum document length in tokens
            query_maxlen: Maximum query length in tokens
            dim: Projected dimension for efficiency (in real ColBERT)
        """
        self.embedder = embedder or SimpleColBERTEmbedder(embedding_dimension)
        self.embedding_dimension = embedding_dimension
        self.score_type = score_type
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.dim = dim  # For projection (simplified)

        self.vectors: Dict[str, ColBERTVector] = {}  # vector_id -> ColBERTVector
        self.document_vectors: Dict[str, List[str]] = {}  # document_id -> [vector_ids]
        self.stats = ColBERTIndexStats()
        self.stats.embedding_dimension = embedding_dimension

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())

    def _compute_maxsim(self, query_embeddings: List[List[float]],
                       doc_embeddings: List[List[float]]) -> float:
        """
        Compute MaxSim similarity between query and document token embeddings.
        For each query token, find the document token with maximum similarity,
        then sum these maxima.
        """
        if not query_embeddings or not doc_embeddings:
            return 0.0

        total_score = 0.0
        for q_emb in query_embeddings:  # For each query token
            max_sim = -float('inf')
            for d_emb in doc_embeddings:  # Find best matching document token
                # Compute cosine similarity
                dot_product = sum(a * b for a, b in zip(q_emb, d_emb))
                norm_q = math.sqrt(sum(a * a for a in q_emb))
                norm_d = math.sqrt(sum(b * b for b in d_emb))
                if norm_q > 0 and norm_d > 0:
                    similarity = dot_product / (norm_q * norm_d)
                else:
                    similarity = 0.0
                max_sim = max(max_sim, similarity)  # Keep the maximum
            total_score += max_sim if max_sim != -float('inf') else 0.0  # Add to total

        return total_score

    def _compute_dot_product(self, query_embeddings: List[List[float]],
                            doc_embeddings: List[List[float]]) -> float:
        """Compute dot product similarity"""
        if not query_embeddings or not doc_embeddings:
            return 0.0

        # Flatten and compute average dot product
        q_flat = [emb for token_embs in query_embeddings for emb in token_embs]
        d_flat = [emb for token_embs in doc_embeddings for emb in token_embs]

        if not q_flat or not d_flat:
            return 0.0

        # Use average vectors for simplicity
        q_avg = [sum(col) / len(col) for col in zip(*q_flat)] if q_flat else [0.0] * self.embedding_dimension
        d_avg = [sum(col) / len(col) for col in zip(*d_flat)] if d_flat else [0.0] * self.embedding_dimension

        dot_product = sum(a * b for a, b in zip(q_avg, d_avg))
        return dot_product

    def _compute_cosine(self, query_embeddings: List[List[float]],
                       doc_embeddings: List[List[float]]) -> float:
        """Compute cosine similarity"""
        if not query_embeddings or not doc_embeddings:
            return 0.0

        # Flatten and compute average vectors
        q_flat = [emb for token_embs in query_embeddings for emb in token_embs]
        d_flat = [emb for token_embs in doc_embeddings for emb in token_embs]

        if not q_flat or not d_flat:
            return 0.0

        q_avg = [sum(col) / len(col) for col in zip(*q_flat)] if q_flat else [0.0] * self.embedding_dimension
        d_avg = [sum(col) / len(col) for col in zip(*d_flat)] if d_flat else [0.0] * self.embedding_dimension

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(q_avg, d_avg))
        norm_q = math.sqrt(sum(a * a for a in q_avg))
        norm_d = math.sqrt(sum(b * b for b in d_avg))
        if norm_q > 0 and norm_d > 0:
            return dot_product / (norm_q * norm_d)
        else:
            return 0.0

    def _compute_similarity(self, query_embeddings: List[List[float]],
                           doc_embeddings: List[List[float]]) -> float:
        """Compute similarity based on score type"""
        if self.score_type == MatchScoreType.MAXSIM:
            return self._compute_maxsim(query_embeddings, doc_embeddings)
        elif self.score_type == MatchScoreType.DOT_PRODUCT:
            return self._compute_dot_product(query_embeddings, doc_embeddings)
        elif self.score_type == MatchScoreType.COSINE:
            return self._compute_cosine(query_embeddings, doc_embeddings)
        else:
            return self._compute_maxsim(query_embeddings, doc_embeddings)  # Default

    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add a document to the ColBERT index.

        Args:
            document_id: Unique identifier for the document
            content: Text content of the document
            metadata: Optional metadata dictionary

        Returns:
            List of vector IDs created from this document
        """
        if metadata is None:
            metadata = {}

        # Truncate document if too long
        tokens = self._tokenize_text(content)
        if len(tokens) > self.doc_maxlen:
            tokens = tokens[:self.doc_maxlen]
            content = ' '.join(tokens)  # Reconstruct truncated content

        # Embed the document
        token_embeddings_list = self.embedder.embed_tokens([content])
        if not token_embeddings_list or not token_embeddings_list[0]:
            return []

        token_embeddings = token_embeddings_list[0]

        # Create vector ID
        vector_id = hashlib.md5(f"{document_id}:{content[:50]}".encode()).hexdigest()[:12]

        # Convert to ColBERTTokenEmbedding objects
        colbert_tokens = []
        for i, token_emb in enumerate(token_embeddings):
            # Find approximate position in text (simplified)
            words = re.findall(r'\b\w+\b', content)
            char_pos = 0
            token_start = 0
            token_end = 0
            for j, word in enumerate(words):
                if j == i:
                    token_start = char_pos
                    token_end = char_pos + len(word)
                    break
                char_pos += len(word) + 1  # +1 for space

            colbert_tokens.append(ColBERTTokenEmbedding(
                token_id=i,
                token_text=words[i] if i < len(words) else "",
                embedding=token_emb,
                start_pos=token_start,
                end_pos=token_end
            ))

        # Create ColBERT vector
        vector = ColBERTVector(
            vector_id=vector_id,
            tokens=colbert_tokens,
            metadata={
                **metadata,
                "document_id": document_id
            }
        )

        # Store vector
        self.vectors[vector_id] = vector

        # Update document mapping
        if document_id not in self.document_vectors:
            self.document_vectors[document_id] = []
        self.document_vectors[document_id].append(vector_id)

        # Update stats
        self._update_stats()

        return [vector_id]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search the ColBERT index for relevant documents.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of (document_id, score) tuples sorted by relevance
        """
        if not self.vectors:
            return []

        # Truncate query if too long
        query_tokens = self._tokenize_text(query)
        if len(query_tokens) > self.query_maxlen:
            query_tokens = query_tokens[:self.query_maxlen]
            query = ' '.join(query_tokens)

        # Embed the query
        query_embeddings_list = self.embedder.embed_tokens([query])
        if not query_embeddings_list or not query_embeddings_list[0]:
            return []
        query_embeddings = query_embeddings_list[0]

        # Compute similarities
        document_scores: Dict[str, float] = defaultdict(float)

        for vector_id, vector in self.vectors.items():
            doc_embeddings = [token.embedding for token in vector.tokens]
            score = self._compute_similarity(query_embeddings, doc_embeddings)
            document_id = vector.metadata.get("document_id", "unknown")
            document_scores[document_id] += score  # Accumulate score if multiple vectors per doc

        # Sort and return top-k
        sorted_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def search_with_vectors(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search and return results with vector-level details.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of result dictionaries with document and vector info
        """
        if not self.vectors:
            return []

        # Truncate query if too long
        query_tokens = self._tokenize_text(query)
        if len(query_tokens) > self.query_maxlen:
            query_tokens = query_tokens[:self.query_maxlen]
            query = ' '.join(query_tokens)

        # Embed the query
        query_embeddings_list = self.embedder.embed_tokens([query])
        if not query_embeddings_list or not query_embeddings_list[0]:
            return []
        query_embeddings = query_embeddings_list[0]

        # Compute similarities per vector
        vector_scores: List[Tuple[str, float]] = []  # (vector_id, score)

        for vector_id, vector in self.vectors.items():
            doc_embeddings = [token.embedding for token in vector.tokens]
            score = self._compute_similarity(query_embeddings, doc_embeddings)
            vector_scores.append((vector_id, score))

        # Sort vectors by score
        vector_scores.sort(key=lambda x: x[1], reverse=True)
        top_vectors = vector_scores[:top_k * 2]  # Get more to allow for deduplication

        # Group by document and accumulate scores
        document_results: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "document_id": "",
            "score": 0.0,
            "vectors": [],
            "metadata": {}
        })

        for vector_id, score in top_vectors:
            vector = self.vectors.get(vector_id)
            if not vector:
                continue

            document_id = vector.metadata.get("document_id", "unknown")
            doc_result = document_results[document_id]
            doc_result["document_id"] = document_id
            doc_result["score"] += score
            doc_result["metadata"] = vector.metadata

            # Add vector info
            doc_result["vectors"].append({
                "vector_id": vector.vector_id,
                "score": score,
                "token_count": len(vector.tokens),
                "sample_tokens": [
                    {
                        "text": token.token_text,
                        "position": f"{token.start_pos}-{token.end_pos}"
                    }
                    for token in vector.tokens[:3]  # First 3 tokens as sample
                ]
            })

        # Convert to list and sort by score
        results = list(document_results.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_document(self, document_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            document_id: ID of document to remove

        Returns:
            True if document was found and removed, False otherwise
        """
        if document_id not in self.document_vectors:
            return False

        # Remove vectors associated with this document
        vector_ids = self.document_vectors[document_id]
        for vector_id in vector_ids:
            if vector_id in self.vectors:
                del self.vectors[vector_id]

        # Remove document mapping
        del self.document_vectors[document_id]

        # Update stats
        self._update_stats()
        return True

    def _update_stats(self):
        """Update index statistics"""
        self.stats.total_documents = len(self.document_vectors)
        self.stats.total_vectors = len(self.vectors)
        self.stats.total_tokens = sum(
            len(vector.tokens) for vector in self.vectors.values()
        )
        if self.stats.total_vectors > 0:
            self.stats.avg_tokens_per_vector = self.stats.total_tokens / self.stats.total_vectors
        else:
            self.stats.avg_tokens_per_vector = 0.0

        # Estimate index size (vectors only for simplicity)
        if self.vectors:
            # Approximate: each token embedding is embedding_dimension * 4 bytes (float32)
            self.stats.index_size_mb = (
                self.stats.total_tokens * self.embedding_dimension * 4
            ) / (1024 * 1024)
        else:
            self.stats.index_size_mb = 0.0

    def get_stats(self) -> ColBERTIndexStats:
        """Get index statistics"""
        return self.stats

    def get_document_vectors(self, document_id: str) -> List[ColBERTVector]:
        """Get all vectors for a document"""
        vector_ids = self.document_vectors.get(document_id, [])
        return [self.vectors[vid] for vid in vector_ids if vid in self.vectors]

    def get_vector(self, vector_id: str) -> Optional[ColBERTVector]:
        """Get a specific vector by ID"""
        return self.vectors.get(vector_id)


# Example usage and demonstration
if __name__ == "__main__":
    # Create ColBERT index
    colbert_index = ColBERTIndex(
        embedding_dimension=128,
        score_type=MatchScoreType.MAXSIM,
        doc_maxlen=100,
        query_maxlen=20
    )

    # Sample documents
    documents = [
        ("doc1", "The quick brown fox jumps over the lazy dog. This is a sentence about animals. Foxes are known for their agility and red fur.",
         {"category": "animals"}),
        ("doc2", "Python is a popular programming language for data science and machine learning. It has many libraries like NumPy, Pandas, and TensorFlow. Created by Guido van Rossum.",
         {"category": "technology"}),
        ("doc3", "Climate change is causing rising sea levels and extreme weather events. Scientists recommend reducing carbon emissions to mitigate these effects. The IPCC reports show alarming trends.",
         {"category": "environment"}),
        ("doc4", "To bake a chocolate cake, you need flour, sugar, cocoa powder, eggs, and butter. Mix the dry ingredients, then add wet ingredients, and bake at 350°F for 30 minutes. Let cool before serving.",
         {"category": "cooking"}),
        ("doc5", "The theory of relativity, proposed by Albert Einstein, revolutionized our understanding of space, time, and gravity. It includes special relativity and general relativity. E=mc^2 is the famous equation.",
         {"category": "physics"})
    ]

    print("Adding documents to ColBERT index...")
    all_vector_ids = []
    for doc_id, content, metadata in documents:
        vector_ids = colbert_index.add_document(doc_id, content, metadata)
        all_vector_ids.extend(vector_ids)
        print(f"Added document '{doc_id}' with {len(vector_ids)} vectors")

    # Show stats
    stats = colbert_index.get_stats()
    print(f"\nColBERT Index Statistics:")
    print(f"  Total documents: {stats.total_documents}")
    print(f"  Total vectors: {stats.total_vectors}")
    print(f"  Total tokens: {stats.total_tokens}")
    print(f"  Avg tokens per vector: {stats.avg_tokens_per_vector:.2f}")
    print(f"  Embedding dimension: {stats.embedding_dimension}")
    print(f"  Index size: {stats.index_size_mb:.2f} MB")

    # Test searches
    test_queries = [
        "What is Python used for?",
        "How to bake a cake?",
        "Climate change effects",
        "Theory of relativity Einstein",
        "quick brown fox",
        "Who invented Python?",
        "What are the effects of climate change?",
        "chocolate cake ingredients"
    ]

    print(f"\nSearch Results:")
    print("=" * 50)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = colbert_index.search(query, top_k=3)

        if results:
            for i, (doc_id, score) in enumerate(results, 1):
                doc = colbert_index.get_document_vectors(doc_id)[0] if colbert_index.get_document_vectors(doc_id) else None
                metadata = doc.metadata if doc else {}
                print(f"  {i}. Document: {doc_id} (Score: {score:.4f})")
                if metadata:
                    print(f"     Category: {metadata.get('category', 'N/A')}")
        else:
            print("  No results found")

    # Demonstrate detailed search
    print(f"\nDetailed Search Example (with vector info):")
    print("-" * 50)
    query = "What is Python used for?"
    detailed_results = colbert_index.search_with_vectors(query, top_k=2)

    for i, result in enumerate(detailed_results, 1):
        print(f"  {i}. Document: {result['document_id']} (Total Score: {result['score']:.4f})")
        print(f"     Vectors: {len(result['vectors'])}")
        for j, vector in enumerate(result['vectors'][:2]):  # Show first 2 vectors
            print(f"       Vector {j+1}: Score={vector['score']:.4f}, Tokens={vector['token_count']}")
            sample_tokens = [f"{t['text']}({t['position']})" for t in vector['sample_tokens']]
            print(f"         Sample tokens: {', '.join(sample_tokens)}")

    # Demonstrate deletion
    print(f"\nDeleting document 'doc2'...")
    success = colbert_index.delete_document("doc2")
    print(f"Deletion successful: {success}")

    # Search again to verify
    print(f"\nSearching for 'Python' after deletion:")
    results = colbert_index.search("Python", top_k=3)
    if results:
        for doc_id, score in results:
            print(f"  Document: {doc_id} (Score: {score:.4f})")
    else:
        print("  No results found (as expected)")