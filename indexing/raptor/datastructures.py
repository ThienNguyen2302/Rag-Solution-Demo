"""
Data structures for RAPTOR technique.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from .enums import NodeType


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