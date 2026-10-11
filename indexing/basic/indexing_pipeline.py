"""
Complete indexing pipeline that handles document processing,
chunking, and index creation for RAG systems.
"""

from typing import List, Dict, Any, Optional, Tuple
from .enums import IndexType, ChunkingStrategy
from .base_index import BaseIndex
from .inverted_index import InvertedIndex
from .vector_index import VectorIndex
from .hybrid_index import HybridIndex


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

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk"""
        chunk = self.index.get_chunk(chunk_id)
        return chunk.to_dict() if chunk else None

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = self.index.get_stats()
        return {
            'total_documents': stats.total_documents,
            'total_chunks': stats.total_chunks,
            'total_tokens': stats.total_tokens,
            'average_chunk_length': stats.average_chunk_length,
            'vocabulary_size': stats.vocabulary_size,
            'index_size_mb': stats.index_size_mb,
            'creation_time': stats.creation_time
        }

    def add_documents_batch(self, documents: List[Tuple[str, str, Optional[Dict[str, Any]]]]) -> List[List[str]]:
        """Add multiple documents in batch"""
        results = []
        for doc_id, content, metadata in documents:
            chunk_ids = self.add_document(doc_id, content, metadata)
            results.append(chunk_ids)
        return results


# Example usage and demonstration
if __name__ == "__main__":
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
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Average chunk length: {stats['average_chunk_length']:.2f}")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")
    print(f"  Index size: {stats['index_size_mb']:.2f} MB")

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