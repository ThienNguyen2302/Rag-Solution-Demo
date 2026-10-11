"""
Example usage and demonstration of multi-representation indexing.
"""

from .multi_representation_index import MultiRepresentationIndex
from .helper_functions import create_default_multi_representation_configs
from .enums import FusionStrategy


if __name__ == "__main__":
    # Create default representation configurations
    configs = create_default_multi_representation_configs()

    # Create multi-representation index
    mri = MultiRepresentationIndex(
        representation_configs=configs,
        fusion_strategy=FusionStrategy.WEIGHTED_SUM,
    )

    # Sample documents
    documents = {
        "doc1": "The quick brown fox jumps over the lazy dog. This is a sentence about animals. Foxes are known for their agility and red fur.",
        "doc2": "Python is a popular programming language for data science and machine learning. It has many libraries like NumPy, Pandas, and TensorFlow. Created by Guido van Rossum.",
        "doc3": "Climate change is causing rising sea levels and extreme weather events. Scientists recommend reducing carbon emissions to mitigate these effects. The IPCC reports show alarming trends.",
        "doc4": "To bake a chocolate cake, you need flour, sugar, cocoa powder, eggs, and butter. Mix the dry ingredients, then add wet ingredients, and bake at 350°F for 30 minutes. Let cool before serving.",
        "doc5": "The theory of relativity, proposed by Albert Einstein, revolutionized our understanding of space, time, and gravity. It includes special relativity and general relativity. E=mc^2 is the famous equation.",
    }

    print("Adding documents to multi-representation index...")
    all_chunk_ids = {}
    for doc_id, content in documents.items():
        chunk_ids = mri.add_document(doc_id, content)
        all_chunk_ids[doc_id] = chunk_ids
        print(f"Added document '{doc_id}' to {len(chunk_ids)} representations")

    # Show statistics
    stats = mri.get_stats()
    print("\nIndex Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    for rep_name, rep_info in stats["representations"].items():
        print(f"  Representation '{rep_name}': {rep_info['type']} (enabled: {rep_info['enabled']})")

    # Test search queries
    test_queries = [
        "What is Python used for?",
        "How to bake a cake?",
        "Climate change effects",
        "Theory of relativity Einstein",
        "quick brown fox",
        "Who invented Python?",
        "What are the effects of climate change?",
        "chocolate cake ingredients",
    ]

    print("\nSearch Results (Multi-Representation Fusion):")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = mri.search(query, top_k=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. Document: {result.document_id} (Score: {result.score:.4f})")
                print(f"     Representations used: {', '.join(result.representations_used)}")

                # Show top 2 representation scores
                sorted_rep_scores = sorted(
                    result.representation_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:2]
                for rep_name, score in sorted_rep_scores:
                    print(f"       {rep_name}: {score:.2f}")

                if result.metadata:
                    preview = str(list(result.metadata.values())[0])[:50] if result.metadata else ""
                    print(f"     Metadata preview: {preview}")
        else:
            print("  No results found")

    # Demonstrate deletion
    print("\nDeleting document 'doc2'...")
    success = mri.delete_document("doc2")
    print(f"Deletion successful: {success}")

    # Search again to verify
    print("\nSearching for 'Python' after deletion:")
    results = mri.search("Python", top_k=3)
    if results:
        for result in results:
            print(f"  Document: {result.document_id} (Score: {result.score:.4f})")
    else:
        print("  No results found (as expected)")