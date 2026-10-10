"""
Multi-Representation Indexing Technique for RAG Systems

This module implements multi-representation indexing, which stores multiple
representations of the same document to capture different aspects and improve
retrieval effectiveness in Retrieval-Augmented Generation (RAG) systems.

Multi-representation indexing strategies:
- Different chunking granularities (sentence, paragraph, document)
- Different indexing approaches (keyword, vector, hybrid) per representation
- Different document views (title, abstract, full text, summary)
- Different transformation representations (questions answered, hypothetical answers)
- Fusion techniques for combining results from multiple representations

This approach improves recall by ensuring that queries matching any
representation can retrieve the relevant document.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import hashlib
from abc import ABC, abstractmethod
import math
import numpy as np


class RepresentationType(Enum):
    """Types of document representations"""
    FULL_TEXT = "full_text"           # Complete document
    SUMMARY = "summary"               # Document summary
    TITLE_ABSTRACT = "title_abstract" # Title and abstract only
    QUESTIONS = "questions"           # Questions the document answers
    HYPOTHETICAL_ANSWERS = "hypothetical_answers"  # Hypothetical answers (HyDE-style)
    KEY_PHRASES = "key_phrases"       # Important phrases/keywords
    ENTITIES = "entities"             # Named entities
    STRUCTURED = "structured"         # Structured data/views


class FusionStrategy(Enum):
    """Strategies for fusing results from multiple representations"""
    WEIGHTED_SUM = "weighted_sum"         # Weighted sum of scores
    MAX_SCORE = "max_score"               # Take maximum score per document
    RANK_FUSION = "rank_fusion"           # Fusion based on rankings (RRF)
    DISTRIBUTION_BASED = "distribution_based"  # Score distribution-based fusion
    LEARNING_BASED = "learning_based"     # Learned fusion (requires training)


@dataclass
class RepresentationConfig:
    """Configuration for a specific representation"""
    name: str                           # Unique identifier for this representation
    type: RepresentationType            # Type of representation
    index_type: str                     # Index type to use (inverted, vector, hybrid)
    chunking_strategy: Optional[str] = None  # Chunking strategy for this representation
    weight: float = 1.0                 # Weight in fusion (default 1.0)
    enabled: bool = True                # Whether this representation is active
    transform_function: Optional[Callable[[str], str]] = None  # Function to transform content
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional configuration


@dataclass
class MultiRepresentationResult:
    """Result from multi-representation search"""
    document_id: str
    score: float
    representation_scores: Dict[str, float]  # Scores from each representation
    representations_used: List[str]          # Which representations contributed
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRepresentationIndex(ABC):
    """Abstract base class for representation-specific indexes"""

    def __init__(self, name: str, config: RepresentationConfig):
        self.name = name
        self.config = config
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.index: Optional[Any] = None  # Will be set to specific index implementation
        self.is_built = False

    @abstractmethod
    def build_index(self, documents: Dict[str, str],
                   metadatas: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Build the index from documents"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search the representation index"""
        pass

    @abstractmethod
    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add a single document"""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        pass

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        return self.documents.get(document_id)


class MultiRepresentationIndex:
    """
    Multi-representation index that maintains multiple indexes for
    different representations of the same document collection.
    """

    def __init__(self,
                 representation_configs: List[RepresentationConfig],
                 fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM,
                 fusion_weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-representation index.

        Args:
            representation_configs: List of representation configurations
            fusion_strategy: How to combine scores from different representations
            fusion_weights: Weights for each representation (if using weighted sum)
        """
        self.representations: Dict[str, BaseRepresentationIndex] = {}
        self.fusion_strategy = fusion_strategy
        self.fusion_weights = fusion_weights or {}
        self.documents: Dict[str, str] = {}  # document_id -> original content
        self.document_metadata: Dict[str, Dict[str, Any]] = {}  # document_id -> metadata

        # Initialize each representation
        for config in representation_configs:
            if config.enabled:
                rep_index = self._create_representation_index(config)
                self.representations[config.name] = rep_index

    def _create_representation_index(self, config: RepresentationConfig) -> BaseRepresentationIndex:
        """Create an index instance for a representation configuration"""
        # Import the index classes from our existing implementation
        # In a real implementation, these would be properly imported
        from indexing_technique import InvertedIndex, VectorIndex, HybridIndex, BaseIndex

        # Create a wrapper that adapts our existing indexes to the representation interface
        class RepresentationIndexWrapper(BaseRepresentationIndex):
            def __init__(self, name: str, config: RepresentationConfig):
                super().__init__(name, config)
                # Initialize the actual index based on config
                if config.index_type == "inverted":
                    self.index = InvertedIndex()
                elif config.index_type == "vector":
                    self.index = VectorIndex()
                elif config.index_type == "hybrid":
                    self.index = HybridIndex()
                else:
                    # Default to hybrid
                    self.index = HybridIndex()

                # Apply chunking strategy if specified (would need implementation)
                # For now, we note it but actual chunking would be in add_document
                self.chunking_strategy = config.chunking_strategy

            def build_index(self, documents: Dict[str, str],
                           metadatas: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
                """Build index from documents"""
                if metadatas is None:
                    metadatas = {}

                for doc_id, content in documents.items():
                    metadata = metadatas.get(doc_id, {})
                    # Apply transformation if specified
                    if config.transform_function:
                        content = config.transform_function(content)
                    self.add_document(doc_id, content, metadata)

                self.is_built = True

            def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
                """Search the representation"""
                if not self.is_built:
                    return []

                # Apply transformation to query if specified
                search_query = query
                if config.transform_function:
                    search_query = config.transform_function(query)

                return self.index.search(search_query, top_k)

            def add_document(self, document_id: str, content: str,
                            metadata: Optional[Dict[str, Any]] = None) -> List[str]:
                """Add single document"""
                if metadata is None:
                    metadata = {}

                # Apply transformation if specified
                if config.transform_function:
                    content = config.transform_function(content)

                # Store document
                self.documents[document_id] = {
                    'content': content,
                    'metadata': metadata
                }

                # Add to index
                chunk_ids = self.index.add_document(document_id, content, metadata)
                return chunk_ids

            def delete_document(self, document_id: str) -> bool:
                """Delete document"""
                if document_id not in self.documents:
                    return False

                del self.documents[document_id]
                return self.index.delete_document(document_id)

        return RepresentationIndexWrapper(config.name, config)

    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Add a document to all representations.

        Returns:
            Dict mapping representation name to list of chunk IDs
        """
        if metadata is None:
            metadata = {}

        # Store original document
        self.documents[document_id] = content
        self.document_metadata[document_id] = metadata or {}

        # Add to each representation
        results = {}
        for rep_name, rep_index in self.representations.items():
            config = rep_index.config
            rep_content = content
            rep_metadata = metadata.copy()

            # Apply representation-specific transformation
            if config.transform_function:
                rep_content = config.transform_function(content)

            # Add to this representation's index
            chunk_ids = rep_index.add_document(document_id, rep_content, rep_metadata)
            results[rep_name] = chunk_ids

        return results

    def search(self, query: str, top_k: int = 10) -> List[MultiRepresentationResult]:
        """
        Search across all representations and fuse results.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of MultiRepresentationResult objects sorted by fused score
        """
        # Get results from each representation
        representation_results: Dict[str, List[Tuple[str, float]]] = {}
        for rep_name, rep_index in self.representations.items():
            if not rep_index.is_built:
                continue

            # Apply query transformation if specified
            search_query = query
            if rep_index.config.transform_function:
                search_query = rep_index.config.transform_function(query)

            results = rep_index.search(search_query, top_k * 2)  # Get extra for fusion
            representation_results[rep_name] = results

        # Fuse results based on strategy
        if self.fusion_strategy == FusionStrategy.WEIGHTED_SUM:
            fused_results = self._weighted_sum_fusion(representation_results, top_k)
        elif self.fusion_strategy == FusionStrategy.MAX_SCORE:
            fused_results = self._max_score_fusion(representation_results, top_k)
        elif self.fusion_strategy == FusionStrategy.RANK_FUSION:
            fused_results = self._rank_fusion_fusion(representation_results, top_k)
        elif self.fusion_strategy == FusionStrategy.DISTRIBUTION_BASED:
            fused_results = self._distribution_based_fusion(representation_results, top_k)
        else:
            # Default to weighted sum
            fused_results = self._weighted_sum_fusion(representation_results, top_k)

        return fused_results

    def _weighted_sum_fusion(self,
                            representation_results: Dict[str, List[Tuple[str, float]]],
                            top_k: int) -> List[MultiRepresentationResult]:
        """Fuse using weighted sum of scores"""
        # Collect all document scores
        doc_scores: Dict[str, float] = {}
        doc_rep_scores: Dict[str, Dict[str, float]] = {}
        doc_rep_used: Dict[str, List[str]] = {}

        for rep_name, results in representation_results.items():
            # Get weight for this representation
            weight = self.fusion_weights.get(rep_name, 1.0)

            # Normalize scores for this representation (0-1 range)
            if results:
                max_score = max(score for _, score in results) if results else 1.0
                min_score = min(score for _, score in results) if results else 0.0
                score_range = max_score - min_score if max_score > min_score else 1.0
            else:
                max_score = 1.0
                min_score = 0.0
                score_range = 1.0

            for doc_id, score in results:
                normalized_score = (score - min_score) / score_range if score_range > 0 else 0.0
                weighted_score = normalized_score * weight

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_rep_scores[doc_id] = {}
                    doc_rep_used[doc_id] = []

                doc_scores[doc_id] += weighted_score
                doc_rep_scores[doc_id][rep_name] = score
                if rep_name not in doc_rep_used[doc_id]:
                    doc_rep_used[doc_id].append(rep_name)

        results: List[MultiRepresentationResult] = []
        for doc_id, score in doc_scores.items():
            result = MultiRepresentationResult(
                document_id=doc_id,
                score=score,
                representation_scores=doc_rep_scores.get(doc_id, {}),
                representations_used=doc_rep_used.get(doc_id, []),
                metadata=self.document_metadata.get(doc_id, {}),
            )
            results.append(result)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _max_score_fusion(self,
                         representation_results: Dict[str, List[Tuple[str, float]]],
                         top_k: int) -> List[MultiRepresentationResult]:
        """Fuse by taking the maximum score per document."""
        doc_max_scores: Dict[str, float] = {}
        doc_rep_scores: Dict[str, Dict[str, float]] = {}
        doc_rep_used: Dict[str, List[str]] = {}

        for rep_name, results in representation_results.items():
            for doc_id, score in results:
                if doc_id not in doc_max_scores or score > doc_max_scores[doc_id]:
                    doc_max_scores[doc_id] = score

                if doc_id not in doc_rep_scores:
                    doc_rep_scores[doc_id] = {}
                    doc_rep_used[doc_id] = []

                doc_rep_scores[doc_id][rep_name] = score
                if rep_name not in doc_rep_used[doc_id]:
                    doc_rep_used[doc_id].append(rep_name)

        results: List[MultiRepresentationResult] = []
        for doc_id, score in doc_max_scores.items():
            result = MultiRepresentationResult(
                document_id=doc_id,
                score=score,
                representation_scores=doc_rep_scores.get(doc_id, {}),
                representations_used=doc_rep_used.get(doc_id, []),
                metadata=self.document_metadata.get(doc_id, {}),
            )
            results.append(result)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _rank_fusion_fusion(self,
                           representation_results: Dict[str, List[Tuple[str, float]]],
                           top_k: int) -> List[MultiRepresentationResult]:
        """Fuse using reciprocal rank fusion (RRF)."""
        k = 60
        doc_scores: Dict[str, float] = {}
        doc_rep_scores: Dict[str, Dict[str, float]] = {}
        doc_rep_used: Dict[str, List[str]] = {}

        for rep_name, results in representation_results.items():
            for rank, (doc_id, score) in enumerate(results, 1):
                if doc_id not in doc_rep_scores:
                    doc_rep_scores[doc_id] = {}
                    doc_rep_used[doc_id] = []

                doc_rep_scores[doc_id][rep_name] = score
                if rep_name not in doc_rep_used[doc_id]:
                    doc_rep_used[doc_id].append(rep_name)

                rrf_score = 1.0 / (k + rank)
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += rrf_score

        results: List[MultiRepresentationResult] = []
        for doc_id, score in doc_scores.items():
            result = MultiRepresentationResult(
                document_id=doc_id,
                score=score,
                representation_scores=doc_rep_scores.get(doc_id, {}),
                representations_used=doc_rep_used.get(doc_id, []),
                metadata=self.document_metadata.get(doc_id, {}),
            )
            results.append(result)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _distribution_based_fusion(self,
                                  representation_results: Dict[str, List[Tuple[str, float]]],
                                  top_k: int) -> List[MultiRepresentationResult]:
        """Fuse using score distribution normalization (Z-score)."""
        doc_z_scores: Dict[str, float] = {}
        doc_rep_scores: Dict[str, Dict[str, float]] = {}
        doc_rep_used: Dict[str, List[str]] = {}

        rep_stats: Dict[str, Dict[str, float]] = {}
        for rep_name, results in representation_results.items():
            if not results:
                continue

            scores = [score for _, score in results]
            mean_score = sum(scores) / len(scores)
            variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
            std_score = math.sqrt(variance) if variance > 0 else 1.0

            rep_stats[rep_name] = {"mean": mean_score, "std": std_score}

            for doc_id, score in results:
                if doc_id not in doc_rep_scores:
                    doc_rep_scores[doc_id] = {}
                    doc_rep_used[doc_id] = []

                doc_rep_scores[doc_id][rep_name] = score
                if rep_name not in doc_rep_used[doc_id]:
                    doc_rep_used[doc_id].append(rep_name)

        for rep_name, results in representation_results.items():
            if rep_name not in rep_stats:
                continue

            mean_score = rep_stats[rep_name]["mean"]
            std_score = rep_stats[rep_name]["std"]

            for doc_id, score in results:
                if std_score > 0:
                    z_score = (score - mean_score) / std_score
                else:
                    z_score = 0.0

                weight = self.fusion_weights.get(rep_name, 1.0)
                weighted_z = z_score * weight

                if doc_id not in doc_z_scores:
                    doc_z_scores[doc_id] = 0.0
                doc_z_scores[doc_id] += weighted_z

        results: List[MultiRepresentationResult] = []
        for doc_id, score in doc_z_scores.items():
            result = MultiRepresentationResult(
                document_id=doc_id,
                score=score,
                representation_scores=doc_rep_scores.get(doc_id, {}),
                representations_used=doc_rep_used.get(doc_id, []),
                metadata=self.document_metadata.get(doc_id, {}),
            )
            results.append(result)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def delete_document(self, document_id: str) -> bool:
        """Delete document from all representations."""
        if document_id not in self.documents:
            return False

        success = True
        for rep_index in self.representations.values():
            if not rep_index.delete_document(document_id):
                success = False

        del self.documents[document_id]
        if document_id in self.document_metadata:
            del self.document_metadata[document_id]

        return success

    def get_representation_index(self, name: str) -> Optional[BaseRepresentationIndex]:
        """Get a specific representation index by name."""
        return self.representations.get(name)

    def get_all_representations(self) -> List[str]:
        """Get list of all representation names."""
        return list(self.representations.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all representations."""
        stats = {
            "total_documents": len(self.documents),
            "representations": {},
        }

        for rep_name, rep_index in self.representations.items():
            if hasattr(rep_index, "index") and hasattr(rep_index.index, "get_stats"):
                rep_stats = rep_index.index.get_stats()
                stats["representations"][rep_name] = {
                    "type": rep_index.config.index_type,
                    "enabled": rep_index.config.enabled,
                    "stats": rep_stats,
                }
            else:
                stats["representations"][rep_name] = {
                    "type": rep_index.config.index_type,
                    "enabled": rep_index.config.enabled,
                    "stats": "Not available",
                }

        return stats


def create_default_multi_representation_configs() -> List[RepresentationConfig]:
    """Create a standard set of representation configurations."""
    return [
        RepresentationConfig(
            name="full_text_inverted",
            type=RepresentationType.FULL_TEXT,
            index_type="inverted",
            weight=1.0,
            metadata={"description": "Full text with keyword indexing"},
        ),
        RepresentationConfig(
            name="full_text_vector",
            type=RepresentationType.FULL_TEXT,
            index_type="vector",
            weight=1.0,
            metadata={"description": "Full text with semantic indexing"},
        ),
        RepresentationConfig(
            name="title_vector",
            type=RepresentationType.TITLE_ABSTRACT,
            index_type="vector",
            weight=1.2,
            metadata={"description": "Title-focused semantic search"},
            transform_function=lambda text: _extract_title(text),
        ),
        RepresentationConfig(
            name="summary_hybrid",
            type=RepresentationType.SUMMARY,
            index_type="hybrid",
            weight=1.1,
            metadata={"description": "Summary with hybrid indexing"},
            transform_function=lambda text: _extract_summary(text),
        ),
        RepresentationConfig(
            name="questions_vector",
            type=RepresentationType.QUESTIONS,
            index_type="vector",
            weight=1.0,
            metadata={"description": "Questions the document answers"},
            transform_function=lambda text: _generate_questions(text),
        ),
    ]


def _extract_title(text: str) -> str:
    """Extract title from text (simplified)."""
    lines = text.strip().split("\n")
    if lines:
        for line in lines[:3]:
            stripped = line.strip()
            if len(stripped) < 100 and len(stripped) > 0:
                return stripped
    return lines[0] if lines else text[:100]


def _extract_summary(text: str) -> str:
    """Extract summary from text (simplified - first few sentences)."""
    sentences = re.split(r"[.!?]+", text)
    summary_sentences = [sentence.strip() for sentence in sentences if sentence.strip()][:3]
    return ". ".join(summary_sentences) + ("." if summary_sentences else "")


def _generate_questions(text: str) -> str:
    """Generate questions that the text answers (simplified)."""
    questions: List[str] = []

    if " is " in text.lower() or " are " in text.lower():
        lines = text.split(".")
        for line in lines:
            lower_line = line.lower()
            if " is " in lower_line or " are " in lower_line:
                if " is " in lower_line:
                    subject = line.split(" is ")[0].strip()
                    questions.append(f"What is {subject}?")
                if " are " in lower_line:
                    subject = line.split(" are ")[0].strip()
                    questions.append(f"What are {subject}?")

    if " how to " in text.lower():
        lines = text.split(".")
        for line in lines:
            lower_line = line.lower()
            if " how to " in lower_line:
                action = line.split(" how to ", 1)[1].strip()
                questions.append(f"How to {action}?")

    if not questions:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if words:
            key_words = [word for word in words if len(word) > 4][:3]
            if key_words:
                questions.append(f"What is {key_words[0]}?")
                questions.append(f"How does {key_words[0]} work?")
                if len(key_words) > 1:
                    questions.append(
                        f"What is the relationship between {key_words[0]} and {key_words[1]}?"
                    )

    return " ".join(questions)


if __name__ == "__main__":
    configs = create_default_multi_representation_configs()
    mri = MultiRepresentationIndex(
        representation_configs=configs,
        fusion_strategy=FusionStrategy.WEIGHTED_SUM,
    )

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

    stats = mri.get_stats()
    print("\nIndex Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    for rep_name, rep_info in stats["representations"].items():
        print(f"  Representation '{rep_name}': {rep_info['type']} (enabled: {rep_info['enabled']})")

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

    print("\nDeleting document 'doc2'...")
    success = mri.delete_document("doc2")
    print(f"Deletion successful: {success}")

    print("\nSearching for 'Python' after deletion:")
    results = mri.search("Python", top_k=3)
    if results:
        for result in results:
            print(f"  Document: {result.document_id} (Score: {result.score:.4f})")
    else:
        print("  No results found (as expected)")