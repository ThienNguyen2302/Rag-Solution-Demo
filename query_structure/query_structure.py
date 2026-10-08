"""
Query Structure Technique for RAG Systems

This module implements query structure techniques, which focus on analyzing,
restructuring, and enhancing user queries to improve retrieval performance
in Retrieval-Augmented Generation (RAG) systems.

Query structure techniques include:
- Query parsing and analysis
- Structural enhancement (adding constraints, metadata, formatting)
- Query templating and canonicalization
- Query decomposition and segmentation
- Intent and entity extraction for structured retrieval
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json


class QueryIntent(Enum):
    """Common query intents in RAG systems"""
    FACTUAL = "factual"          # Seeking specific facts
    EXPLANATORY = "explanatory"  # Seeking explanations or how-to
    COMPARATIVE = "comparative"  # Comparing entities or concepts
    PROCEDURAL = "procedural"    # Seeking step-by-step procedures
    TROUBLESHOOTING = "troubleshooting"  # Seeking solutions to problems
    EXPLORATORY = "exploratory"  # Broad exploration of a topic
    NAVIGATIONAL = "navigational"  # Seeking specific documents or resources


@dataclass
class QueryStructure:
    """Represents the structured form of a query"""
    original_query: str
    intent: QueryIntent
    entities: List[str]
    constraints: List[str]
    keywords: List[str]
    expanded_terms: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'original_query': self.original_query,
            'intent': self.intent.value,
            'entities': self.entities,
            'constraints': self.constraints,
            'keywords': self.keywords,
            'expanded_terms': self.expanded_terms,
            'confidence': self.confidence
        }

    def to_structured_prompt(self) -> str:
        """Generate a structured prompt version of the query"""
        parts = [f"Query: {self.original_query}"]

        if self.intent != QueryIntent.EXPLORATORY:  # Don't add if default
            parts.append(f"Intent: {self.intent.value}")

        if self.entities:
            parts.append(f"Entities: {', '.join(self.entities)}")

        if self.constraints:
            parts.append(f"Constraints: {', '.join(self.constraints)}")

        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")

        return " | ".join(parts)


class QueryStructurer:
    """
    Implements query structure analysis and enhancement for RAG systems.

    Analyzes user queries to extract intent, entities, constraints, and
    other structural elements that can improve retrieval and generation.
    """

    def __init__(self):
        # Intent classification patterns
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\b(what is|who is|when was|where is|define|name)\b',
                r'\b(list|identify|state)\b.*\b(of|for)\b',
                r'\bhow many\b',
                r'\bwhat\s+(are|is)\b'
            ],
            QueryIntent.EXPLANATORY: [
                r'\b(explain|describe|how does|why does|what causes)\b',
                r'\b(what is the process|how to|walk me through)\b',
                r'\btell me about\b',
                r'\b(what are the reasons|what leads to)\b'
            ],
            QueryIntent.COMPARATIVE: [
                r'\b(compare|difference between|versus|vs|better than)\b',
                r'\b(pros and cons|advantages and disadvantages)\b',
                r'\b(which is better|which one)\b',
                r'\b(differentiate|distinguish between)\b'
            ],
            QueryIntent.PROCEDURAL: [
                r'\b(how to|steps to|guide to|tutorial for)\b',
                r'\b(step by step|procedure for|method to)\b',
                r'\b(how do i|how can i)\b.*\b(to|for)\b',
                r'\b(instructions|directions|guidance)\b'
            ],
            QueryIntent.TROUBLESHOOTING: [
                r'\b(fix|resolve|solve|troubleshoot|debug)\b',
                r'\b(error|issue|problem|bug|failure)\b',
                r'\b(not working|doesn\'t work|won\'t start)\b',
                r'\b(how to fix|solution to|workaround for)\b'
            ],
            QueryIntent.NAVIGATIONAL: [
                r'\b(find|locate|get|download|access)\b.*\b(document|file|page|manual)\b',
                r'\b(where can i find|where is the)\b',
                r'\b(link to|url for|reference to)\b',
                r'\b(open|view|show me)\b.*\b(document|file|page)\b'
            ]
        }

        # Constraint indicators
        self.constraint_patterns = [
            r'\b(in|during|for|over|under|above|below|before|after)\b\s+\d+\s*\w*',
            r'\b(from|to|between)\s+\d{4}',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(recent|latest|current|newest|oldest)\b',
            r'\b(first|last|next|previous)\b',
            r'\b(only|just|exactly|specifically)\b',
            r'\b(limit|top|bottom|first|last)\s+\d+\b',
            r'\b(since|until|from|to)\b',
            r'\b(in|of|for)\s+(\w+\s+)?\w+\s+(edition|version|chapter|section)\b'
        ]

        # Entity patterns (simplified - in practice would use NER)
        self.entity_patterns = {
            'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'ORGANIZATION': r'\b[A-Z][a-zA-Z]*\s*(Inc|Corp|Ltd|LLC|Company|Co\.)\b',
            'TECHNOLOGY': r'\b(AI|ML|API|SDK|Framework|Language|Database|Cloud)\b',
            'PRODUCT': r'\b[vV]\d+\.?\d*\.?\d*\b|\b[A-Z]{2,}\d+\b'  # Version numbers, product codes
        }

    def analyze_query(self, query: str) -> QueryStructure:
        """
        Analyze a query to extract its structural components.

        Args:
            query: The user's original query string

        Returns:
            QueryStructure: The analyzed and structured query representation
        """
        query_lower = query.lower().strip()

        # Extract intent
        intent = self._classify_intent(query_lower)

        # Extract entities
        entities = self._extract_entities(query)

        # Extract constraints
        constraints = self._extract_constraints(query_lower)

        # Extract keywords
        keywords = self._extract_keywords(query_lower)

        # Generate expanded terms
        expanded_terms = self._generate_expanded_terms(query_lower, intent, keywords)

        # Calculate confidence (simplified)
        confidence = self._calculate_confidence(intent, entities, constraints, keywords)

        return QueryStructure(
            original_query=query,
            intent=intent,
            entities=entities,
            constraints=constraints,
            keywords=keywords,
            expanded_terms=expanded_terms,
            confidence=confidence
        )

    def _classify_intent(self, query_lower: str) -> QueryIntent:
        """Classify the intent of the query based on patterns"""
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent] = score

        # Return intent with highest score, default to EXPLORATORY
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        return QueryIntent.EXPLORATORY

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query (simplified implementation)"""
        entities = []

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)

        # Also extract quoted phrases as potential entities
        quoted_matches = re.findall(r'["\']([^"\']*)["\']', query)
        entities.extend(quoted_matches)

        # Remove duplicates and return
        return list(dict.fromkeys(entities))  # Preserves order while removing dupes

    def _extract_constraints(self, query_lower: str) -> List[str]:
        """Extract constraints from the query"""
        constraints = []

        for pattern in self.constraint_patterns:
            matches = re.findall(pattern, query_lower)
            # Handle tuple results from groups in regex
            for match in matches:
                if isinstance(match, tuple):
                    # Join tuple elements
                    constraint = ' '.join(filter(None, match))
                else:
                    constraint = match
                if constraint.strip():
                    constraints.append(constraint.strip())

        # Also look for explicit constraint indicators
        constraint_indicators = [
            (r'\b(with|without|including|excluding|containing)\b\s+([^,.]+)', 2),
            (r'\b(only|just|exactly|specifically)\b\s+([^,.]+)', 2),
            (r'\b(limit|restrict|confine)\s+to\b\s+([^,.]+)', 2)
        ]

        for pattern, group_idx in constraint_indicators:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if match.strip():
                    constraints.append(match.strip())

        return list(dict.fromkeys(constraints))  # Remove duplicates

    def _extract_keywords(self, query_lower: str) -> List[str]:
        """Extract important keywords from the query"""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'can', 'will', 'just', 'should', 'now'
        }

        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z]+\b', query_lower)

        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Also extract quoted phrases as keywords
        quoted_keywords = re.findall(r'["\']([^"\']*)["\']', query_lower)
        keywords.extend([k.lower() for k in quoted_keywords if len(k) > 2])

        return list(dict.fromkeys(keywords))  # Remove duplicates

    def _generate_expanded_terms(self, query_lower: str, intent: QueryIntent,
                               keywords: List[str]) -> List[str]:
        """Generate expanded terms for better retrieval"""
        expanded = []

        # Add intent-specific expansions
        intent_expansions = {
            QueryIntent.FACTUAL: ['definition', 'meaning', 'explanation'],
            QueryIntent.EXPLANATORY: ['process', 'mechanism', 'reason', 'cause'],
            QueryIntent.COMPARATIVE: ['difference', 'similarity', 'versus', 'comparison'],
            QueryIntent.PROCEDURAL: ['steps', 'guide', 'tutorial', 'method', 'procedure'],
            QueryIntent.TROUBLESHOOTING: ['solution', 'fix', 'resolve', 'troubleshoot'],
            QueryIntent.NAVIGATIONAL: ['document', 'file', 'manual', 'guide', 'reference']
        }

        if intent in intent_expansions:
            expanded.extend(intent_expansions[intent])

        # Add synonyms for key terms (simplified - in practice would use WordNet or embeddings)
        synonym_map = {
            'how': ['method', 'way', 'technique', 'approach'],
            'what': ['definition', 'explanation', 'description', 'information'],
            'why': ['reason', 'cause', 'explanation', 'rationale'],
            'best': ['top', 'leading', 'premier', 'optimal', 'recommended'],
            'fix': ['repair', 'resolve', 'correct', 'solve', 'remedy']
        }

        for keyword in keywords:
            if keyword in synonym_map:
                expanded.extend(synonym_map[keyword])

        return list(dict.fromkeys(expanded))  # Remove duplicates

    def _calculate_confidence(self, intent: QueryIntent, entities: List[str],
                            constraints: List[str], keywords: List[str]) -> float:
        """Calculate confidence score for the query structure analysis"""
        score = 0.5  # Base confidence

        # Boost for clear intent (non-exploratory)
        if intent != QueryIntent.EXPLORATORY:
            score += 0.2

        # Boost for entities found
        if entities:
            score += min(0.2, len(entities) * 0.05)

        # Boost for constraints found
        if constraints:
            score += min(0.15, len(constraints) * 0.05)

        # Boost for good keyword count
        if 3 <= len(keywords) <= 8:
            score += 0.1
        elif len(keywords) > 8:
            score += 0.05  # Diminishing returns for too many keywords

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

    def enhance_query_for_retrieval(self, query_structure: QueryStructure) -> List[str]:
        """
        Generate enhanced query variations for improved retrieval.

        Args:
            query_structure: The structured query analysis

        Returns:
            List[str]: List of enhanced query variations
        """
        enhanced_queries = [query_structure.original_query]  # Always include original

        # Add structured prompt version
        structured = query_structure.to_structured_prompt()
        if structured != f"Query: {query_structure.original_query}":
            enhanced_queries.append(structured)

        # Add keyword-focused query
        if query_structure.keywords:
            keyword_query = ' '.join(query_structure.keywords[:5])  # Top 5 keywords
            enhanced_queries.append(keyword_query)

        # Add intent-enhanced query
        if query_structure.intent != QueryIntent.EXPLORATORY:
            intent_prefixes = {
                QueryIntent.FACTUAL: "What is the definition of",
                QueryIntent.EXPLANATORY: "Explain how",
                QueryIntent.COMPARATIVE: "Compare and contrast",
                QueryIntent.PROCEDURAL: "Provide steps for",
                QueryIntent.TROUBLESHOOTING: "How to troubleshoot",
                QueryIntent.NAVIGATIONAL: "Find documentation for"
            }

            if query_structure.intent in intent_prefixes:
                prefix = intent_prefixes[query_structure.intent]
                core_terms = ' '.join(query_structure.keywords[:3]) if query_structure.keywords else query_structure.original_query
                enhanced_queries.append(f"{prefix} {core_terms}")

        # Add constraint-focused query
        if query_structure.constraints:
            constraint_phrase = ' '.join(query_structure.constraints[:3])
            core_query = query_structure.original_query
            enhanced_queries.append(f"{core_query} with constraints: {constraint_phrase}")

        return list(dict.fromkeys(enhanced_queries))  # Remove duplicates while preserving order

    def get_retrieval_weights(self, query_structure: QueryStructure) -> Dict[str, float]:
        """
        Suggest weights for different retrieval strategies based on query structure.

        Args:
            query_structure: The structured query analysis

        Returns:
            Dict[str, float]: Suggested weights for different retrieval approaches
        """
        weights = {
            'semantic': 0.4,      # Default semantic similarity weight
            'keyword': 0.3,       # Default keyword matching weight
            'structured': 0.2,    # Default structured/query format weight
            'entity': 0.1         # Default entity matching weight
        }

        # Adjust weights based on query characteristics
        if query_structure.intent == QueryIntent.FACTUAL:
            weights['entity'] += 0.15
            weights['keyword'] += 0.1
            weights['semantic'] -= 0.1
            weights['structured'] -= 0.15

        elif query_structure.intent == QueryIntent.EXPLANATORY:
            weights['semantic'] += 0.15
            weights['structured'] += 0.1
            weights['keyword'] -= 0.1
            weights['entity'] -= 0.15

        elif query_structure.intent == QueryIntent.PROCEDURAL:
            weights['structured'] += 0.2
            weights['semantic'] += 0.1
            weights['keyword'] -= 0.15
            weights['entity'] -= 0.15

        elif query_structure.intent == QueryIntent.TROUBLESHOOTING:
            weights['keyword'] += 0.2
            weights['structured'] += 0.1
            weights['semantic'] -= 0.1
            weights['entity'] -= 0.2

        elif query_structure.intent == QueryIntent.NAVIGATIONAL:
            weights['entity'] += 0.2
            weights['structured'] += 0.1
            weights['semantic'] -= 0.1
            weights['keyword'] -= 0.2

        # Boost for queries with many constraints
        if len(query_structure.constraints) > 2:
            weights['structured'] += 0.1
            weights['keyword'] += 0.05
            weights['semantic'] -= 0.05
            weights['entity'] -= 0.1

        # Boost for queries with specific entities
        if len(query_structure.entities) > 1:
            weights['entity'] += 0.15
            weights['keyword'] += 0.05
            weights['semantic'] -= 0.1
            weights['structured'] -= 0.1

        # Ensure all weights are non-negative and sum to 1.0
        total = sum(max(0, w) for w in weights.values())
        if total > 0:
            weights = {k: max(0, v) / total for k, v in weights.items()}

        return weights


# Example usage and testing
if __name__ == "__main__":
    structurer = QueryStructurer()

    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Latest news about AI developments in 2024",
        "How to fix a leaking faucet in the kitchen",
        "Compare Python vs JavaScript for web development",
        "Find the user manual for iPhone 15 Pro",
        "Explain how photosynthesis works in plants",
        "Count the number of users who logged in yesterday",
        "What are the symptoms of diabetes and how to treat it?",
        "Show me the configuration file for nginx web server",
        "Why does my computer keep freezing when I open Chrome?"
    ]

    print("Query Structure Analysis Examples:")
    print("=" * 60)

    for query in test_queries:
        print(f"\nOriginal Query: {query}")

        # Analyze query
        structure = structurer.analyze_query(query)

        print(f"  Intent: {structure.intent.value} (confidence: {structure.confidence:.2f})")
        print(f"  Entities: {structure.entities}")
        print(f"  Constraints: {structure.constraints}")
        print(f"  Keywords: {structure.keywords}")
        print(f"  Expanded Terms: {structure.expanded_terms[:5]}")  # Show first 5

        # Show structured prompt
        structured_prompt = structure.to_structured_prompt()
        if structured_prompt != f"Query: {query}":
            print(f"  Structured Prompt: {structured_prompt}")

        # Show enhancement suggestions
        enhanced = structurer.enhance_query_for_retrieval(structure)
        if len(enhanced) > 1:
            print(f"  Enhanced Queries ({len(enhanced)} total):")
            for i, eq in enumerate(enhanced[:3], 1):  # Show first 3
                print(f"    {i}. {eq}")

        # Show retrieval weights
        weights = structurer.get_retrieval_weights(structure)
        weight_str = ", ".join([f"{k}: {v:.2f}" for k, v in weights.items()])
        print(f"  Suggested Weights: {weight_str}")
        print("-" * 40)