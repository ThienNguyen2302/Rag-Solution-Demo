"""
Enumerations for RAPTOR technique.
"""

from enum import Enum


class NodeType(Enum):
    """Types of nodes in the RAPTOR tree"""
    LEAF = "leaf"          # Original document chunks
    SUMMARY = "summary"    # Summarized parent nodes
    ROOT = "root"          # Top-level node(s)