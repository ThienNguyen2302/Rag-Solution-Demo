"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) for RAG Systems

This module implements the RAPTOR technique, which builds a hierarchical tree structure
by recursively summarizing chunks of text at increasing levels of abstraction. This
enables retrieval at different levels of detail and improves performance on queries
requiring global context understanding.

RAPTOR Process:
1. Chunk documents into base-level nodes
2. Recursively cluster and summarize nodes to create parent nodes
3. Continue until reaching a root node or desired tree depth
4. Enable retrieval at any level of the tree
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
from collections import defaultdict
import heapq


class NodeType(Enum):
    """Types of nodes in the RAPTOR tree"""
    LEAF = "leaf"          # Original document chunks
    SUMMARY = "summary"    # Summarized parent nodes
    ROOT = "root"          # Top-level node(s)


@dataclass
class RaptorNode:
    """Represents a node in the RAPTOR tree"""
    node_id: str
    content: str
    summary: Optional[str]  # Abstractive summary (None for leaf nodes)
    level: int              # 0 = leaf level, higher = more abstract
    node_type: NodeType
    children: List[str] = field(default_factory=list)  # Child node IDs
    parent: Optional[str] = None  # Parent node ID
    embedding: Optional[List[float]] = None  # Vector representation
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # Retrieval score


@dataclass
class RaptorTree:
    """Represents the complete RAPTOR tree structure"""
    nodes: Dict[str, RaptorNode] = field(default_factory=dict)
    leaf_nodes: List[str] = field(default_factory=list)
    root_nodes: List[str] = field(default_factory=list)
    max_level: int = 0

    def get_node(self, node_id: str) -> Optional[RaptorNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def get_nodes_at_level(self, level: int) -> List[RaptorNode]:
        """Get all nodes at a specific level"""
        return [node for node in self.nodes.values() if node.level == level]

    def get_leaf_descendants(self, node_id: str) -> List[str]:
        """Get all leaf descendant nodes of a given node"""
        node = self.get_node(node_id)
        if not node:
            return []

        if node.node_type == NodeType.LEAF:
            return [node_id]

        descendants = []
        for child_id in node.children:
            descendants.extend(self.get_leaf_descendants(child_id))
        return descendants


class BaseSummarizer(ABC):
    """Abstract base class for summarization models"""

    @abstractmethod
    def summarize(self, texts: List[str], max_length: Optional[int] = None) -> List[str]:
        """Summarize a list of texts"""
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass


class SimpleSummarizer(BaseSummarizer):
    """Simple extractive summarizer for demonstration (replace with real model)"""

    def __init__(self, max_summary_length: int = 100):
        self.max_summary_length = max_summary_length

    def _extract_key_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Extract key sentences using simple scoring"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + ('.' if sentences else '')

        # Simple scoring: prefer sentences with key terms
        # In practice, would use TF-IDF, TextRank, or neural models
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Position bias: prefer beginning and end
            position_score = 1.0 - abs(0.5 - i / max(len(sentences) - 1, 1)) * 0.5
            # Length penalty: prefer medium length sentences
            length_score = min(len(sentence.split()) / 20.0, 1.0) if len(sentence.split()) > 0 else 0
            # Keyword bonus: sentences with numbers or proper nouns
            keyword_bonus = 0.2 if any(c.isdigit() for c in sentence) else 0.0
            keyword_bonus += 0.2 if any(c.isupper() and c.isalpha() for c in sentence) else 0.0

            total_score = position_score * 0.4 + length_score * 0.3 + keyword_bonus * 0.3
            scored_sentences.append((total_score, i, sentence))

        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        selected = sorted(scored_sentences[:max_sentences], key=lambda x: x[1])  # Sort by original order
        result = '. '.join([s[2] for s in selected])
        return result + ('.' if result and not result.endswith('.') else '')

    def summarize(self, texts: List[str], max_length: Optional[int] = None) -> List[str]:
        """Summarize texts using extractive approach"""
        max_len = max_length or self.max_summary_length
        summaries = []

        for text in texts:
            if len(text) <= max_len:
                summaries.append(text)
            else:
                # For longer texts, extract key sentences
                summary = self._extract_key_sentences(text, max_sentences=3)
                # Truncate if still too long
                if len(summary) > max_len:
                    summary = summary[:max_len].rsplit(' ', 1)[0] + '...'
                summaries.append(summary)

        return summaries

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Simple embedding function (placeholder)"""
        embeddings = []
        dim = 384  # Standard embedding dimension

        for text in texts:
            # Very simple embedding - in practice use SBERT, BERT, etc.
            embedding = [0.0] * dim
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

            if not words:
                embeddings.append(embedding)
                continue

            # Bag-of-words style embedding with positional encoding
            for i, word in enumerate(words[:50]):  # Limit to first 50 words
                word_hash = hash(word) % 1000
                idx = (i * 7 + word_hash) % dim
                embedding[idx] = min(embedding[idx] + 0.1, 1.0)

            # Add some basic text features
            if len(embedding) > 0:
                embedding[0] = min(len(text) / 1000.0, 1.0)  # Length feature
            if len(embedding) > 1:
                embedding[1] = min(len(words) / 100.0, 1.0)   # Word count feature

            # Normalize
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

            embeddings.append(embedding)

        return embeddings


class RaptorIndex:
    """
    RAPTOR index implementation that builds a hierarchical tree of summaries
    for improved retrieval in RAG systems.
    """

    def __init__(self,
                 summarizer: Optional[BaseSummarizer] = None,
                 branching_factor: int = 5,
                 max_levels: int = 3,
                 threshold_similarity: float = 0.7,
                 chunk_size: int = 500):
        """
        Initialize RAPTOR index.

        Args:
            summarizer: Summarization model to use (defaults to SimpleSummarizer)
            branching_factor: Number of children per parent node
            max_levels: Maximum tree depth to build
            threshold_similarity: Similarity threshold for clustering
            chunk_size: Target size for leaf chunks (characters)
        """
        self.summarizer = summarizer or SimpleSummarizer()
        self.branching_factor = branching_factor
        self.max_levels = max_levels
        self.threshold_similarity = threshold_similarity
        self.chunk_size = chunk_size

        self.tree = RaptorTree()
        self._node_counter = 0

    def _get_next_node_id(self) -> str:
        """Generate unique node ID"""
        self._node_counter += 1
        return f"node_{self._node_counter:06d}"

    def _chunk_document(self, text: str) -> List[str]:
        """Simple fixed-size chunking (can be replaced with smarter chunking)"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending near the boundary
                search_start = max(start + self.chunk_size - 100, start)
                search_end = min(start + self.chunk_size + 100, len(text))
                search_text = text[search_start:search_end]

                # Find last sentence ending
                last_period = max(
                    search_text.rfind('.'),
                    search_text.rfind('!'),
                    search_text.rfind('?')
                )

                if last_period != -1:
                    end = search_start + last_period + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end

        return chunks

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _cluster_nodes(self, node_ids: List[str]) -> List[List[str]]:
        """Cluster nodes based on similarity using simple greedy approach"""
        if not node_ids:
            return []

        # Get embeddings for all nodes
        nodes = [self.tree.get_node(nid) for nid in node_ids if self.tree.get_node(nid)]
        if not nodes:
            return [[nid] for nid in node_ids]

        # Simple clustering: assign each node to the closest cluster center
        clusters: List[List[str]] = []
        cluster_centers: List[str] = []  # Node IDs representing cluster centers

        for node in nodes:
            if not node.embedding:
                # If no embedding, put in its own cluster
                clusters.append([node.node_id])
                continue

            # Find closest cluster center
            best_cluster_idx = -1
            best_similarity = -1.0

            for i, center_id in enumerate(cluster_centers):
                center_node = self.tree.get_node(center_id)
                if center_node and center_node.embedding:
                    similarity = self._cosine_similarity(node.embedding, center_node.embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster_idx = i

            # If similarity is good enough, add to existing cluster
            if best_cluster_idx != -1 and best_similarity >= self.threshold_similarity:
                clusters[best_cluster_idx].append(node.node_id)
            else:
                # Start new cluster
                clusters.append([node.node_id])
                cluster_centers.append(node.node_id)

        # Limit cluster size to branching factor
        final_clusters = []
        for cluster in clusters:
            if len(cluster) <= self.branching_factor:
                final_clusters.append(cluster)
            else:
                # Split large clusters
                for i in range(0, len(cluster), self.branching_factor):
                    final_clusters.append(cluster[i:i + self.branching_factor])

        return final_clusters

    def _create_parent_nodes(self, child_node_ids: List[str], level: int) -> List[str]:
        """Create parent nodes from child nodes by clustering and summarizing"""
        if not child_node_ids:
            return []

        # Cluster child nodes
        clusters = self._cluster_nodes(child_node_ids)
        parent_node_ids = []

        for cluster in clusters:
            if not cluster:
                continue

            # Get content from child nodes in cluster
            child_contents = []
            for child_id in cluster:
                child_node = self.tree.get_node(child_id)
                if child_node:
                    # Use summary if available (for higher levels), otherwise content
                    content = child_node.summary if child_node.summary else child_node.content
                    child_contents.append(content)

            if not child_contents:
                continue

            # Create summary for the cluster
            combined_text = " ".join(child_contents)
            summaries = self.summarizer.summarize([combined_text])
            summary = summaries[0] if summaries else combined_text[:200] + "..."

            # Generate embedding for the summary
            embeddings = self.summarizer.embed([summary])
            embedding = embeddings[0] if embeddings else None

            # Create parent node
            parent_id = self._get_next_node_id()
            parent_node = RaptorNode(
                node_id=parent_id,
                content=combined_text,  # Keep original content for reference
                summary=summary,
                level=level,
                node_type=NodeType.SUMMARY if level < self.max_levels else NodeType.ROOT,
                embedding=embedding,
                metadata={"cluster_size": len(cluster)}
            )

            self.tree.nodes[parent_id] = parent_node

            # Set up parent-child relationships
            for child_id in cluster:
                child_node = self.tree.get_node(child_id)
                if child_node:
                    child_node.parent = parent_id
                    parent_node.children.append(child_id)

            parent_node_ids.append(parent_id)

        return parent_node_ids

    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add a document to the RAPTOR index.

        Args:
            document_id: Unique identifier for the document
            content: Text content of the document
            metadata: Optional metadata dictionary

        Returns:
            List of leaf node IDs created from this document
        """
        if metadata is None:
            metadata = {}

        # Chunk the document into leaf nodes
        chunks = self._chunk_document(content)
        leaf_node_ids = []

        # Create leaf nodes (level 0)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i:03d}"

            # Generate embedding for the chunk
            embeddings = self.summarizer.embed([chunk])
            embedding = embeddings[0] if embeddings else None

            leaf_node = RaptorNode(
                node_id=chunk_id,
                content=chunk,
                summary=None,  # Leaf nodes don't have summaries
                level=0,
                node_type=NodeType.LEAF,
                embedding=embedding,
                metadata={
                    **metadata,
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_count": len(chunks)
                }
            )

            self.tree.nodes[chunk_id] = leaf_node
            self.tree.leaf_nodes.append(chunk_id)
            leaf_node_ids.append(chunk_id)

        # Build the tree hierarchy
        current_level_nodes = leaf_node_ids
        current_level = 0

        while current_level < self.max_levels and current_level_nodes:
            # Create parent nodes for current level
            parent_node_ids = self._create_parent_nodes(current_level_nodes, current_level + 1)

            if not parent_node_ids:
                break

            # Move up to next level
            current_level_nodes = parent_node_ids
            current_level += 1

        # Update root nodes (nodes with no parent)
        self.tree.root_nodes = [
            node_id for node_id, node in self.tree.nodes.items()
            if node.parent is None
        ]

        # Update max level
        if self.tree.nodes:
            self.tree.max_level = max(node.level for node in self.tree.nodes.values())

        return leaf_node_ids

    def search(self, query: str, top_k: int = 10,
              search_levels: Optional[List[int]] = None) -> List[Tuple[str, float]]:
        """
        Search the RAPTOR tree for relevant content.

        Args:
            query: Search query text
            top_k: Number of results to return
            search_levels: Specific levels to search (None = search all levels)

        Returns:
            List of (node_id, score) tuples sorted by relevance
        """
        if not self.tree.nodes:
            return []

        # Generate query embedding
        query_embeddings = self.summarizer.embed([query])
        if not query_embeddings:
            return []
        query_embedding = query_embeddings[0]

        # Determine which levels to search
        if search_levels is None:
            search_levels = list(range(self.tree.max_level + 1))

        # Score nodes at each level
        node_scores: Dict[str, float] = {}

        for level in search_levels:
            nodes_at_level = self.tree.get_nodes_at_level(level)
            for node in nodes_at_level:
                if node.embedding:
                    similarity = self._cosine_similarity(query_embedding, node.embedding)
                    # Boost score for higher levels (more abstract) for global queries
                    level_bonus = 1.0 + (level * 0.1)  # Small boost for higher levels
                    node_scores[node.node_id] = similarity * level_bonus
                else:
                    # Fallback to text matching if no embedding
                    query_lower = query.lower()
                    content_lower = (node.summary or node.content).lower()
                    # Simple word overlap score
                    query_words = set(re.findall(r'\b[a-zA-Z]+\b', query_lower))
                    content_words = set(re.findall(r'\b[a-zA-Z]+\b', content_lower))
                    if query_words:
                        overlap = len(query_words.intersection(content_words)) / len(query_words)
                        node_scores[node.node_id] = overlap * 0.5  # Lower weight for text matching

        # Sort by score and return top-k
        sorted_results = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def search_with_context(self, query: str, top_k: int = 10,
                           include_children: bool = True,
                           include_parent: bool = True) -> List[Dict[str, Any]]:
        """
        Search and return results with contextual information.

        Args:
            query: Search query text
            top_k: Number of results to return
            include_children: Include child nodes in results
            include_parent: Include parent node in results

        Returns:
            List of result dictionaries with node info and context
        """
        results = self.search(query, top_k * 2)  # Get more to allow for filtering
        final_results = []

        for node_id, score in results:
            if len(final_results) >= top_k:
                break

            node = self.tree.get_node(node_id)
            if not node:
                continue

            result = {
                "node_id": node.node_id,
                "content": node.content,
                "summary": node.summary,
                "level": node.level,
                "node_type": node.node_type.value,
                "score": score,
                "metadata": node.metadata
            }

            # Add parent context if requested
            if include_parent and node.parent:
                parent_node = self.tree.get_node(node.parent)
                if parent_node:
                    result["parent"] = {
                        "node_id": parent_node.node_id,
                        "summary": parent_node.summary,
                        "content": parent_node.content[:200] + "..." if len(parent_node.content) > 200 else parent_node.content,
                        "level": parent_node.level
                    }

            # Add children context if requested
            if include_children and node.children:
                children_info = []
                for child_id in node.children[:3]:  # Limit to first 3 children
                    child_node = self.tree.get_node(child_id)
                    if child_node:
                        children_info.append({
                            "node_id": child_node.node_id,
                            "summary": child_node.summary,
                            "content": child_node.content[:100] + "..." if len(child_node.content) > 100 else child_node.content,
                            "level": child_node.level
                        })
                result["children"] = children_info

            final_results.append(result)

        return final_results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAPTOR tree"""
        if not self.tree.nodes:
            return {
                "total_nodes": 0,
                "leaf_nodes": 0,
                "summary_nodes": 0,
                "root_nodes": 0,
                "max_level": 0,
                "avg_branching_factor": 0.0
            }

        nodes_by_type = defaultdict(int)
        nodes_by_level = defaultdict(int)
        total_children = 0
        nodes_with_children = 0

        for node in self.tree.nodes.values():
            nodes_by_type[node.node_type.value] += 1
            nodes_by_level[node.level] += 1

            if node.children:
                total_children += len(node.children)
                nodes_with_children += 1

        avg_branching = total_children / max(nodes_with_children, 1)

        return {
            "total_nodes": len(self.tree.nodes),
            "leaf_nodes": nodes_by_type["leaf"],
            "summary_nodes": nodes_by_type["summary"],
            "root_nodes": nodes_by_type["root"],
            "max_level": self.tree.max_level,
            "nodes_by_level": dict(nodes_by_level),
            "avg_branching_factor": round(avg_branching, 2)
        }


# Example usage and demonstration
if __name__ == "__main__":
    # Create RAPTOR index
    raptor = RaptorIndex(
        branching_factor=5,
        max_levels=3,
        threshold_similarity=0.6,
        chunk_size=400
    )

    # Sample documents
    documents = [
        ("doc1", "The quick brown fox jumps over the lazy dog. This is a sentence about animals. Foxes are known for their agility and red fur. They are omnivorous mammals belonging to the family Canidae.",
         {"category": "animals"}),
        ("doc2", "Python is a popular programming language for data science and machine learning. It has many libraries like NumPy, Pandas, and TensorFlow. Created by Guido van Rossum, Python emphasizes code readability with its notable use of significant whitespace.",
         {"category": "technology"}),
        ("doc3", "Climate change is causing rising sea levels and extreme weather events. Scientists recommend reducing carbon emissions to mitigate these effects. The IPCC reports show alarming trends in global temperatures and weather patterns.",
         {"category": "environment"}),
        ("doc4", "To bake a chocolate cake, you need flour, sugar, cocoa powder, eggs, and butter. Mix the dry ingredients, then add wet ingredients, and bake at 350°F for 30 minutes. Let cool before serving.",
         {"category": "cooking"}),
        ("doc5", "The theory of relativity, proposed by Albert Einstein, revolutionized our understanding of space, time, and gravity. It includes special relativity and general relativity. E=mc^2 is the famous equation representing mass-energy equivalence.",
         {"category": "physics"})
    ]

    print("Adding documents to RAPTOR index...")
    all_leaf_ids = []
    for doc_id, content, metadata in documents:
        leaf_ids = raptor.add_document(doc_id, content, metadata)
        all_leaf_ids.extend(leaf_ids)
        print(f"Added document '{doc_id}' with {len(leaf_ids)} leaf nodes")

    # Show stats
    stats = raptor.get_stats()
    print(f"\nRAPTOR Tree Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Leaf nodes: {stats['leaf_nodes']}")
    print(f"  Summary nodes: {stats['summary_nodes']}")
    print(f"  Root nodes: {stats['root_nodes']}")
    print(f"  Max level: {stats['max_level']}")
    print(f"  Average branching factor: {stats['avg_branching_factor']}")
    print(f"  Nodes by level: {stats['nodes_by_level']}")

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
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = raptor.search_with_context(query, top_k=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. Node: {result['node_id']} (Score: {result['score']:.4f})")
                print(f"     Level: {result['level']} ({result['node_type']})")
                if result['summary']:
                    print(f"     Summary: {result['summary']}")
                else:
                    print(f"     Content: {result['content'][:100]}...")

                if result.get('parent'):
                    parent = result['parent']
                    print(f"     Parent Level {parent['level']}: {parent['summary'][:100]}...")

                if result.get('children'):
                    print(f"     Children: {len(result['children'])} nodes")
                    for child in result['children'][:2]:
                        print(f"       - Level {child['level']}: {child['summary'][:50]}...")
        else:
            print("  No results found")

    # Demonstrate tree traversal
    print(f"\nTree Structure Example:")
    print("-" * 30)
    if raptor.tree.root_nodes:
        root_id = raptor.tree.root_nodes[0]
        def print_node_tree(node_id, indent=0):
            node = raptor.tree.get_node(node_id)
            if not node:
                return
            prefix = "  " * indent
            node_type = "🌲" if node.node_type == NodeType.ROOT else "📄" if node.node_type == NodeType.LEAF else "📝"
            content_preview = (node.summary or node.content)[:50]
            print(f"{prefix}{node_type} {node.node_id} (L{node.level}): {content_preview}...")
            for child_id in node.children:
                print_node_tree(child_id, indent + 1)

        print_node_tree(root_id)