"""
Semantic Routing Technique for RAG Systems

This module implements semantic routing, which uses embedding-based similarity
to route queries to appropriate data sources or processing paths based on
semantic meaning rather than keyword matching.

Semantic routing typically involves:
- Embedding queries and candidate destinations
- Computing similarity scores
- Routing to the most semantically similar destination
- Clustering-based routing approaches
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class DataSource(Enum):
    """Available data sources for routing"""
    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"
    DATABASE = "database"
    API_ENDPOINT = "api_endpoint"
    DOCUMENT_STORE = "document_store"
    CODE_REPOSITORY = "code_repository"
    CONVERSATION_HISTORY = "conversation_history"


@dataclass
class RoutingCandidate:
    """Represents a potential routing destination"""
    source: DataSource
    description: str
    embedding: Optional[np.ndarray] = None
    examples: List[str] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


class SemanticRouter:
    """
    Implements semantic routing for RAG systems.

    Uses sentence embeddings to compute semantic similarity between
    queries and predefined routing candidates.
    """

    def __init__(self, embedding_dimension: int = 384):
        """
        Initialize the semantic router.

        Args:
            embedding_dimension: Dimension of embeddings to use
        """
        self.embedding_dimension = embedding_dimension
        self.candidates: List[RoutingCandidate] = []

        # Initialize default candidates with example queries
        self._initialize_default_candidates()

    def _initialize_default_candidates(self):
        """Set up default routing candidates with example queries"""
        self.candidates = [
            RoutingCandidate(
                source=DataSource.KNOWLEDGE_BASE,
                description="Factual information, definitions, explanations",
                examples=[
                    "What is machine learning?",
                    "Explain the theory of relativity",
                    "Who invented the telephone?",
                    "How does photosynthesis work?",
                    "Define algorithm",
                    "What are the symptoms of diabetes?"
                ]
            ),
            RoutingCandidate(
                source=DataSource.WEB_SEARCH,
                description="Current events, news, recent information",
                examples=[
                    "Latest news about climate change",
                    "Today's stock market performance",
                    "Recent developments in AI",
                    "Breaking news from Ukraine",
                    "Current weather in New York",
                    "Recent sports scores"
                ]
            ),
            RoutingCandidate(
                source=DataSource.DATABASE,
                description="Structured data queries, aggregations, transactions",
                examples=[
                    "Count users by country",
                    "Average sales per month",
                    "Find customers who spent over $1000",
                    "Show me the top 10 products by revenue",
                    "What is the maximum temperature recorded?",
                    "List all orders from last week"
                ]
            ),
            RoutingCandidate(
                source=DataSource.API_ENDPOINT,
                description="External service calls, computations, conversions",
                examples=[
                    "Convert 100 dollars to euros",
                    "Translate hello to Spanish",
                    "Calculate mortgage payment",
                    "Get current exchange rate",
                    "Compute tax on income",
                    "Validate email address format"
                ]
            ),
            RoutingCandidate(
                source=DataSource.DOCUMENT_STORE,
                description="Document retrieval, file search, content lookup",
                examples=[
                    "Find the user manual for installation",
                    "Search for safety guidelines",
                    "Look up the API documentation",
                    "Find the troubleshooting guide",
                    "Search for benefit policy document",
                    "Locate the configuration file"
                ]
            ),
            RoutingCandidate(
                source=DataSource.CODE_REPOSITORY,
                description="Code snippets, programming examples, technical implementations",
                examples=[
                    "How to implement binary search in Python",
                    "Show me a React hook for form validation",
                    "Example of REST API with Node.js",
                    "SQL query to join two tables",
                    "CSS flexbox centering example",
                    "Debugging JavaScript asynchronous code"
                ]
            ),
            RoutingCandidate(
                source=DataSource.CONVERSATION_HISTORY,
                description="Previous chat context, follow-up questions, clarification",
                examples=[
                    "Can you elaborate on that?",
                    "What did you mean by earlier?",
                    "Going back to your previous point",
                    "Following up on our discussion",
                    "As mentioned before",
                    "To repeat what I asked earlier"
                ]
            )
        ]

        # Generate embeddings for examples (simplified - in practice would use real embedding model)
        self._generate_example_embeddings()

    def _generate_example_embeddings(self):
        """
        Generate embeddings for example queries.
        In a real implementation, this would use a sentence transformer model.
        For this example, we'll create random embeddings as placeholders.
        """
        np.random.seed(42)  # For reproducible results

        for candidate in self.candidates:
            # Create embedding as average of example embeddings
            example_embeddings = []
            for example in candidate.examples:
                # Simple hash-based embedding (not semantic, just for demonstration)
                # In practice, use: model.encode([example])[0]
                embedding = self._simple_text_embedding(example)
                example_embeddings.append(embedding)

            if example_embeddings:
                candidate.embedding = np.mean(example_embeddings, axis=0)
            else:
                candidate.embedding = np.random.rand(self.embedding_dimension)

    def _simple_text_embedding(self, text: str) -> np.ndarray:
        """
        Create a simple text embedding (placeholder for real embedding model).
        This is just for demonstration - real implementation would use SBERT or similar.
        """
        # Very simple feature extraction for demo purposes
        features = np.zeros(self.embedding_dimension)

        # Length feature
        features[0] = len(text) / 100.0  # Normalize length

        # Word count feature
        features[1] = len(text.split()) / 20.0

        # Character frequency features (simplified)
        for i, char in enumerate(text.lower()[:min(len(text), 50)]):
            if i < self.embedding_dimension - 2:
                features[i + 2] = ord(char) / 128.0

        # Add some randomization based on text content
        hash_val = hash(text) % 10000
        features[-1] = hash_val / 10000.0

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def route_query(self, query: str) -> Tuple[DataSource, float]:
        """
        Route a query to the most semantically similar data source.

        Args:
            query: The user's query string

        Returns:
            Tuple of (DataSource, confidence_score)
        """
        # Generate embedding for the query
        query_embedding = self._simple_text_embedding(query)

        # Compute similarity with each candidate
        similarities = []
        for candidate in self.candidates:
            if candidate.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, candidate.embedding)
                similarities.append((candidate.source, similarity))
            else:
                similarities.append((candidate.source, 0.0))

        # Find the best match
        best_source, best_score = max(similarities, key=lambda x: x[1])

        return best_source, best_score

    def get_route_explanation(self, query: str) -> Dict[str, Any]:
        """
        Get detailed explanation of semantic routing decision.

        Args:
            query: The user's query string

        Returns:
            Dict containing routing decision, scores, and explanation
        """
        query_embedding = self._simple_text_embedding(query)

        # Compute similarities with all candidates
        candidate_scores = []
        for candidate in self.candidates:
            if candidate.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, candidate.embedding)
                candidate_scores.append({
                    'source': candidate.source.value,
                    'description': candidate.description,
                    'similarity': similarity,
                    'examples': candidate.examples[:2]  # Show first 2 examples
                })

        # Sort by similarity score
        candidate_scores.sort(key=lambda x: x['similarity'], reverse=True)

        best_candidate = candidate_scores[0] if candidate_scores else None

        return {
            'query': query,
            'selected_source': best_candidate['source'] if best_candidate else None,
            'confidence_score': best_candidate['similarity'] if best_candidate else 0.0,
            'all_candidates': candidate_scores,
            'explanation': f"Routed to {best_candidate['source'] if best_candidate else 'unknown'} "
                          f"with {best_candidate['similarity']:.3f} confidence" if best_candidate else "No candidates available"
        }

    def add_candidate(self, source: DataSource, description: str, examples: List[str]):
        """
        Add a new routing candidate.

        Args:
            source: The data source
            description: Description of when to use this source
            examples: Example queries that should route to this source
        """
        candidate = RoutingCandidate(
            source=source,
            description=description,
            examples=examples
        )

        # Generate embedding for the new candidate
        np.random.seed(hash(str(source) + str(examples)) % 10000)
        example_embeddings = []
        for example in examples:
            embedding = self._simple_text_embedding(example)
            example_embeddings.append(embedding)

        if example_embeddings:
            candidate.embedding = np.mean(example_embeddings, axis=0)
        else:
            candidate.embedding = np.random.rand(self.embedding_dimension)

        self.candidates.append(candidate)

    def update_candidate_examples(self, source: DataSource, new_examples: List[str]):
        """
        Update examples for an existing candidate and regenerate its embedding.

        Args:
            source: The data source to update
            new_examples: New list of example queries
        """
        for candidate in self.candidates:
            if candidate.source == source:
                candidate.examples = new_examples
                # Regenerate embedding
                example_embeddings = []
                for example in new_examples:
                    embedding = self._simple_text_embedding(example)
                    example_embeddings.append(embedding)

                if example_embeddings:
                    candidate.embedding = np.mean(example_embeddings, axis=0)
                else:
                    candidate.embedding = np.random.rand(self.embedding_dimension)
                break


# Example usage
if __name__ == "__main__":
    router = SemanticRouter()

    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Latest news about AI developments",
        "Count the number of users in the database",
        "Convert 100 USD to EUR",
        "Find the PDF manual for installation",
        "Explain how photosynthesis works",
        "Can you tell me more about that?",
        "Show me a Python example of binary search",
        "What's the weather like today?",
        "Looking at the Q3 sales report..."
    ]

    print("Semantic Routing Examples:")
    print("=" * 50)

    for query in test_queries:
        source, confidence = router.route_query(query)
        explanation = router.get_route_explanation(query)

        print(f"\nQuery: {query}")
        print(f"Routed to: {source.value}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Explanation: {explanation['explanation']}")

        # Show top 3 alternatives
        print("Top alternatives:")
        for i, candidate in enumerate(explanation['all_candidates'][:3]):
            print(f"  {i+1}. {candidate['source']}: {candidate['similarity']:.3f}")