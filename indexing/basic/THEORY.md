# Basic Indexing Techniques for RAG Systems

## Overview

This module implements fundamental indexing techniques essential for Retrieval-Augmented Generation (RAG) systems. Proper indexing transforms raw documents into searchable structures that enable fast and relevant retrieval, which is critical for RAG performance.

## Core Concepts

### 1. Inverted Index
- **Purpose**: Keyword-based retrieval using term-document mappings
- **How it works**: Maps each unique term to the documents/chunks containing it, along with term frequencies
- **Best for**: Exact keyword matching, boolean queries, TF-IDF based ranking
- **Limitations**: Poor semantic understanding, struggles with synonyms and morphological variations

### 2. Vector Index
- **Purpose**: Semantic search using dense embeddings
- **How it works**: Converts text to fixed-length vectors using embedding models, retrieves via similarity search (cosine/dot product)
- **Best for**: Semantic similarity, conceptual matching, cross-lingual retrieval
- **Limitations**: Requires embedding models, less interpretable, may miss exact keyword matches

### 3. Hybrid Index
- **Purpose**: Combines strengths of inverted and vector indexes
- **How it works**: Maintains both indexes separately, combines scores at query time
- **Best for**: Balanced keyword-semantic retrieval, robust performance across query types
- **Trade-offs**: Higher storage and computational cost, requires score fusion strategy

### 4. Document Chunking Strategies
Proper chunking is crucial for effective indexing:

- **Fixed Size**: Split by character/token count (simple but may break semantic units)
- **Sentence-based**: Split at sentence boundaries (preserves linguistic units)
- **Paragraph-based**: Split at paragraph boundaries (preserves topical units)
- **Semantic**: Split by semantic coherence (requires models, most meaningful)
- **Recursive**: Split by separators in order of importance (markdown, code structure)
- **Sliding Window**: Overlapping chunks (preserves context at boundaries)

## Implementation Notes

This implementation provides:
- Abstract base class (`BaseIndex`) defining the indexing interface
- Concrete implementations for each index type
- Unified pipeline (`IndexingPipeline`) for easy switching between techniques
- Standardized data structures (`DocumentChunk`, `IndexStats`)
- Tokenization and basic embedding functions (placeholders for production models)

## When to Use Each Technique

- **Inverted Index**: When keyword precision is paramount (legal, technical documentation)
- **Vector Index**: When semantic understanding is more important than exact matches (conversational QA)
- **Hybrid Index**: For general-purpose RAG systems requiring both capabilities
- **Chunking Strategy**: Match to document structure and retrieval goals