"""
Indexing Technique for RAG Systems

This module implements various indexing techniques for efficient document storage and retrieval
in Retrieval-Augmented Generation (RAG) systems. Indexing is crucial for transforming raw
documents into searchable structures that enable fast and relevant retrieval.

Indexing techniques covered:
- Document chunking and segmentation strategies
- Inverted index construction (keyword-based)
- Vector index construction (embedding-based)
- Hybrid indexing approaches
- Metadata indexing and filtering
- Index optimization and compression techniques
"""

from os import name
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import hashlib
from abc import ABC, abstractmethod
import math

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

class BaseIndex(ABC):
    """Abstract base class for all index types"""

    def __init__(self, index_type: IndexType):
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
    
class InvertedIndex(BaseIndex):
    """
    Inverted index implementation for keyword-based retrieval.

    Maps terms to the documents/chunks that contain them, enabling
    efficient keyword-based search with boolean and ranking capabilities.
    """

    def __init__(self):
        super().__init__(IndexType.INVERTED)
        # Inverted index: term -> {chunk_id: frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        # Document frequency: term -> number of chunks containing term
        self.doc_freq: Dict[str, int] = {}
        # Field norms for length normalization
        self.field_norms: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - in practice would use more sophisticated tokenizer"""
        # Convert to lowercase and extract alphanumeric tokens
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Remove common stop words (simplified list)
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'can', 'will', 'just', 'should', 'now'
        }
        return [token for token in tokens if token not in stop_words and len(token) > 2]

    def _calculate_tf_idf(self, term_freq: int, doc_freq: int, total_chunks: int) -> float:
        """Calculate TF-IDF score"""
        if term_freq == 0 or doc_freq == 0:
            return 0.0
        tf = term_freq
        idf = math.log(total_chunks / doc_freq) if doc_freq > 0 else 0
        return tf * idf

    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add document to inverted index"""
        if metadata is None:
            metadata = {}

        # Store document
        self.documents[document_id] = {
            'content': content,
            'metadata': metadata,
            'chunks': []
        }

        # For inverted index, we'll treat the whole document as one chunk for simplicity
        # In practice, you'd want to chunk first
        chunk_id = hashlib.md5(f"{document_id}:{content[:50]}".encode()).hexdigest()[:12]

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            start_index=0,
            end_index=len(content),
            metadata=metadata,
            tokens=self._tokenize(content)
        )

        self.chunks[chunk_id] = chunk
        self.documents[document_id]['chunks'].append(chunk_id)

        # Update inverted index
        term_freq: Dict[str, int] = {}
        for token in chunk.tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
            chunk.tokens.append(token)  # Store tokens in chunk

        # Update global structures
        for term, freq in term_freq.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
                self.doc_freq[term] = 0

            self.inverted_index[term][chunk_id] = freq
            self.doc_freq[term] += 1

        # Update stats
        self.stats.total_documents = len(self.documents)
        self.stats.total_chunks = len(self.chunks)
        self.stats.total_tokens = sum(len(chunk.tokens) for chunk in self.chunks.values())
        if self.stats.total_chunks > 0:
            self.stats.average_chunk_length = self.stats.total_tokens / self.stats.total_chunks
        self.stats.vocabulary_size = len(self.inverted_index)

        return [chunk_id]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using TF-IDF scoring"""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Calculate query term frequencies
        query_tf: Dict[str, int] = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1

        # Score documents using TF-IDF cosine similarity
        scores: Dict[str, float] = {}
        query_norm = 0.0

        # Calculate query vector norm
        for term, tf in query_tf.items():
            if term in self.doc_freq:
                idf = math.log(self.stats.total_chunks / self.doc_freq[term]) if self.doc_freq[term] > 0 else 0
                query_norm += (tf * idf) ** 2

        query_norm = math.sqrt(query_norm) if query_norm > 0 else 1.0

        # Calculate document scores
        for term, query_tf_val in query_tf.items():
            if term not in self.inverted_index:
                continue

            idf = math.log(self.stats.total_chunks / self.doc_freq[term]) if self.doc_freq[term] > 0 else 0
            query_weight = query_tf_val * idf

            for chunk_id, term_freq in self.inverted_index[term].items():
                # Get chunk document
                chunk = self.chunks.get(chunk_id)
                if not chunk:
                    continue

                doc_id = chunk.document_id

                # Calculate term frequency in document
                tf = term_freq

                # Calculate TF-IDF for document term
                if self.doc_freq[term] > 0:
                    idf_val = math.log(self.stats.total_chunks / self.doc_freq[term])
                    doc_weight = tf * idf_val
                else:
                    doc_weight = 0

                # Accumulate score
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += query_weight * doc_weight

        # Normalize scores by document length (simple approach)
        normalized_scores: Dict[str, float] = {}
        for doc_id, score in scores.items():
            doc = self.documents.get(doc_id)
            if doc and doc['chunks']:
                # Simple length normalization
                length_factor = 1.0 / math.sqrt(len(doc['chunks'])) if len(doc['chunks']) > 0 else 1.0
                normalized_scores[doc_id] = score * length_factor
            else:
                normalized_scores[doc_id] = score

        # Sort and return top-k
        sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def delete_document(self, document_id: str) -> bool:
        """Remove document from index"""
        if document_id not in self.documents:
            return False

        # Remove chunks and update inverted index
        chunk_ids = self.documents[document_id]['chunks']
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                # Remove term frequencies from inverted index
                for term in chunk.tokens:
                    if term in self.inverted_index and chunk_id in self.inverted_index[term]:
                        del self.inverted_index[term][chunk_id]
                        # Remove term if no longer in any chunk
                        if not self.inverted_index[term]:
                            del self.inverted_index[term]
                            self.doc_freq.pop(term, None)

                del self.chunks[chunk_id]

        # Remove document
        del self.documents[document_id]

        # Update stats
        self.stats.total_documents = len(self.documents)
        self.stats.total_chunks = len(self.chunks)
        self.stats.total_tokens = sum(len(chunk.tokens) for chunk in self.chunks.values())
        if self.stats.total_chunks > 0:
            self.stats.average_chunk_length = self.stats.total_tokens / self.stats.total_chunks
        self.stats.vocabulary_size = len(self.inverted_index)

        return True

class VectorIndex(BaseIndex):
    """
    Vector index implementation using embeddings for semantic search.

    Stores document embeddings and uses cosine similarity for retrieval.
    """

    def __init__(self, embedding_dimension: int = 384):
        super().__init__(IndexType.VECTOR)
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

class HybridIndex(BaseIndex):
    """
    Hybrid index combining inverted and vector indexes for best of both worlds.
    """

    def __init__(self, embedding_dimension: int = 384):
        super().__init__(IndexType.HYBRID)
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

class IndexingPipeline:
    """
    Complete indexing pipeline that handles document processing,
    chunking, and index creation for RAG systems.
    """

    def __init__(self, index_type: IndexType = IndexType.HYBRID,
                chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
                embedding_dimension: int = 384):
        self.index_type = index_type
        self.chunking_strategy = chunking_strategy
        self.embedding_dimension = embedding_dimension

        # Initialize the appropriate index
        if index_type == IndexType.INVERTED:
            self.index = InvertedIndex()
        elif index_type == IndexType.VECTOR:
            self.index = VectorIndex(embedding_dimension)
        elif index_type == IndexType.HYBRID:
            self.index = HybridIndex(embedding_dimension)
        else:
            # Default to hybrid for unimplemented types
            self.index = HybridIndex(embedding_dimension)

    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add a document through the complete pipeline"""
        return self.index.add_document(document_id, content, metadata)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search the index"""
        return self.index.search(query, top_k)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        return self.index.delete_document(document_id)

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        return self.index.get_document(document_id)

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk"""
        return self.index.get_chunk(chunk_id)

    def get_stats(self) -> IndexStats:
        """Get index statistics"""
        return self.index.get_stats()

    def add_documents_batch(self, documents: List[Tuple[str, str, Optional[Dict[str, Any]]]]) -> List[List[str]]:
        """Add multiple documents in batch"""
        results = []
        for doc_id, content, metadata in documents:
            chunk_ids = self.add_document(doc_id, content, metadata)
            results.append(chunk_ids)
        return results
    
#Example usage and demonstration

if name == "main":
    # Create indexing pipeline
    pipeline = IndexingPipeline(index_type=IndexType.HYBRID)

# Sample documents
documents = [
    ("doc1", "The quick brown fox jumps over the lazy dog. This is a sentence about animals.",
    {"category": "animals", "author": "John Doe"}),
    ("doc2", "Python is a popular programming language for data science and machine learning. "
            "It has many libraries like NumPy, Pandas, and TensorFlow.",
    {"category": "technology", "author": "Jane Smith"}),
    ("doc3", "Climate change is causing rising sea levels and extreme weather events. "
            "Scientists recommend reducing carbon emissions to mitigate these effects.",
    {"category": "environment", "author": "Dr. James Lee"}),
    ("doc4", "To bake a chocolate cake, you need flour, sugar, cocoa powder, eggs, and butter. "
            "Mix the dry ingredients, then add wet ingredients, and bake at 350°F for 30 minutes.",
    {"category": "cooking", "author": "Chef Maria Garcia"}),
    ("doc5", "The theory of relativity, proposed by Albert Einstein, revolutionized our understanding "
            "of space, time, and gravity. It includes special relativity and general relativity.",
    {"category": "physics", "author": "Prof. Robert Chen"})
]

# Add documents
print("Adding documents to index...")
all_chunk_ids = []
for doc_id, content, metadata in documents:
    chunk_ids = pipeline.add_document(doc_id, content, metadata)
    all_chunk_ids.extend(chunk_ids)
    print(f"Added document '{doc_id}' with {len(chunk_ids)} chunks")

# Show stats
stats = pipeline.get_stats()
print(f"\nIndex Statistics:")
print(f"  Total documents: {stats.total_documents}")
print(f"  Total chunks: {stats.total_chunks}")
print(f"  Total tokens: {stats.total_tokens}")
print(f"  Average chunk length: {stats.average_chunk_length:.2f}")
print(f"  Vocabulary size: {stats.vocabulary_size}")
print(f"  Index size: {stats.index_size_mb:.2f} MB")

# Test searches
test_queries = [
    "What is Python used for?",
    "How to bake a cake?",
    "Climate change effects",
    "Theory of relativity Einstein",
    "quick brown fox"
]

print(f"\nSearch Results:")
print("=" * 50)

for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = pipeline.search(query, top_k=3)

    if results:
        for i, (doc_id, score) in enumerate(results, 1):
            doc = pipeline.get_document(doc_id)
            preview = doc['content'][:100] + "..." if doc and len(doc['content']) > 100 else doc['content'] if doc else "N/A"
            print(f"  {i}. Document: {doc_id} (Score: {score:.4f})")
            print(f"     Preview: {preview}")
            if doc and 'metadata' in doc:
                print(f"     Metadata: {doc['metadata']}")
    else:
        print("  No results found")

# Demonstrate deletion
print(f"\nDeleting document 'doc2'...")
success = pipeline.delete_document("doc2")
print(f"Deletion successful: {success}")

# Search again to verify
print(f"\nSearching for 'Python' after deletion:")
results = pipeline.search("Python", top_k=3)
if results:
    for doc_id, score in results:
        print(f"  Document: {doc_id} (Score: {score:.4f})")
else:
    print("  No results found (as expected)")