# Multi-Representation Indexing for RAG Systems

## Overview

Multi-representation indexing is an advanced technique that stores multiple representations of the same document to capture different aspects and improve retrieval effectiveness in Retrieval-Augmented Generation (RAG) systems. This approach recognizes that different queries may match different aspects of a document, and by maintaining various representations, we can improve recall without sacrificing precision.

## Core Concept

The fundamental idea is to create multiple views or representations of each document, index each representation appropriately (potentially with different indexing strategies), and then fuse the results during query time. This is particularly effective because:

1. **Different queries match different aspects**: A query about "Python programming" might match the full text, while "Who created Python?" might match better with a question-answer representation
2. **Different indexing strategies suit different content**: Keyword indexing works well for exact terms, while vector indexing excels at semantic similarity
3. **Compensates for representation limitations**: No single representation captures all relevant aspects of a document

## Common Representation Types

### 1. Full Text Representation
- **Content**: Complete document text
- **Best for**: Comprehensive coverage, detailed information retrieval
- **Indexing**: Often uses both inverted (keyword) and vector (semantic) indexes

### 2. Summary Representation
- **Content**: Condensed version of the document
- **Best for**: Quick topic matching, filtering irrelevant documents
- **Indexing**: Vector indexing often works well for semantic summary matching

### 3. Title-Abstract Representation
- **Content**: Title and abstract/summary sections only
- **Best for**: Quick topical filtering, academic paper retrieval
- **Rationale**: Titles and abstracts often contain the most salient information

### 4. Question-Based Representation
- **Content**: Questions that the document answers (generated or extracted)
- **Best for**: Query matching in QA systems, conversational retrieval
- **Rationale**: Directly aligns with user query format

### 5. Entity-Based Representation
- **Content**: Named entities (people, places, organizations, etc.)
- **Best for**: Entity-focused queries, fact-checking, disambiguation
- **Indexing**: Often works well with inverted indexes for exact entity matching

### 6. Key Phrase Representation
- **Content**: Important phrases/keywords extracted from the document
- **Best for**: Topic modeling, concept matching, tag-based retrieval
- **Rationale**: Captures the essence without full text noise

## Fusion Strategies

After retrieving results from each representation, we need to combine (fuse) them effectively:

### 1. Weighted Sum
- **How**: Normalize scores from each representation, apply weights, sum them
- **Best for**: When representations have complementary strengths
- **Flexibility**: Easy to tune weights based on representation importance

### 2. Max Score
- **How**: Take the maximum score across all representations for each document
- **Best for**: When any strong match in any representation is sufficient
- **Conservative**: Tend to favor documents with at least one very strong match

### 3. Rank Fusion (RRF)
- **How**: Convert rankings to scores using 1/(k + rank) formula, then sum
- **Best for**: When rankings are more reliable than raw scores
- **Robust**: Less sensitive to score distribution differences between representations

### 4. Distribution-Based (Z-score)
- **How**: Normalize scores using Z-score normalization per representation, then combine
- **Best for**: When score distributions vary significantly between representations
- **Statistical**: Accounts for varying score ranges and variances

## Implementation Approach

This implementation follows a modular design:

1. **RepresentationConfig**: Defines how each representation should be indexed
2. **BaseRepresentationIndex**: Abstract interface for representation-specific indexes
3. **RepresentationIndexWrapper**: Adapts existing indexes (inverted, vector, hybrid) to the representation interface
4. **MultiRepresentationIndex**: Main class that manages multiple representations and handles fusion
5. **Helper Functions**: Default configurations and transformation functions (title extraction, summary generation, question generation)

## Advantages

- **Improved Recall**: Queries matching any representation can retrieve the relevant document
- **Flexibility**: Different representations can use different indexing strategies
- **Interpretability**: Can see which representations contributed to a match
- **Customizable**: Easy to add new representation types or fusion strategies

## When to Use

Multi-representation indexing is particularly beneficial when:
- Document collection is heterogeneous in content/style
- User queries vary widely in format and intent
- Precision is important but recall needs improvement
- Resources allow for maintaining multiple indexes
- Query logs show patterns that specific representations would catch

The technique adds storage and query-time computational overhead but often provides significant retrieval quality improvements that justify the cost.