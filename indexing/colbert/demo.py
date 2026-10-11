"""
Example usage and demonstration of ColBERT indexing.
"""

from .colbert_index import ColBERTIndex
from .enums import MatchScoreType


if __name__ == "__main__":
    # Create ColBERT index with MaxSim scoring (the core ColBERT innovation)
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

    # Show statistics
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

    # Demonstrate different scoring methods
    print(f"\nComparison of Scoring Methods:")
    print("-" * 50)
    query = "Python programming"

    for score_type in [MatchScoreType.MAXSIM, MatchScoreType.DOT_PRODUCT, MatchScoreType.COSINE]:
        temp_index = ColBERTIndex(
            embedding_dimension=128,
            score_type=score_type,
            doc_maxlen=100,
            query_maxlen=20
        )

        # Add documents
        for doc_id, content, metadata in documents:
            temp_index.add_document(doc_id, content, metadata)

        # Search
        results = temp_index.search(query, top_k=2)
        print(f"{score_type.value:12} | Top result: {results[0][0] if results else 'None'} (Score: {results[0][1]:.4f}" if results else "No results")