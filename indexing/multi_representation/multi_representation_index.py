"""
Multi-representation index that maintains multiple indexes for
different representations of the same document collection.
"""

from typing import List, Dict, Any, Optional, Tuple
from .enums import RepresentationType, FusionStrategy
from .datastructures import RepresentationConfig, MultiRepresentationResult
from .base_representation_index import BaseRepresentationIndex


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
        # For now, we'll import from the basic indexing module
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'basic'))
        from inverted_index import InvertedIndex
        from vector_index import VectorIndex
        from hybrid_index import HybridIndex

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
            std_score = (variance ** 0.5) if variance > 0 else 1.0

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
            # Try to get stats from the underlying index if available
            if hasattr(rep_index, "index") and hasattr(rep_index.index, "get_stats"):
                try:
                    rep_stats = rep_index.index.get_stats()
                    stats["representations"][rep_name] = {
                        "type": rep_index.config.index_type,
                        "enabled": rep_index.config.enabled,
                        "stats": rep_stats,
                    }
                except AttributeError:
                    stats["representations"][rep_name] = {
                        "type": rep_index.config.index_type,
                        "enabled": rep_index.config.enabled,
                        "stats": "Not available",
                    }
            else:
                stats["representations"][rep_name] = {
                    "type": rep_index.config.index_type,
                    "enabled": rep_index.config.enabled,
                    "stats": "Not available",
                }

        return stats