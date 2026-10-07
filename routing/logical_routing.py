"""
Logical Routing Technique for RAG Systems

This module implements logical routing, which uses rule-based or logic-based
approaches to route queries to appropriate data sources or processing paths
in a Retrieval-Augmented Generation (RAG) system.

Logical routing typically involves:
- Keyword-based routing
- Metadata-based routing
- Rule-based decision trees
- Structured query analysis
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import re


class DataSource(Enum):
    """Available data sources for routing"""
    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"
    DATABASE = "database"
    API_ENDPOINT = "api_endpoint"
    DOCUMENT_STORE = "document_store"


class LogicalRouter:
    """
    Implements logical routing for RAG systems.

    Uses predefined rules and logic to determine the best data source
    or processing path for a given query.
    """

    def __init__(self):
        # Define routing rules
        self.routing_rules = {
            DataSource.KNOWLEDGE_BASE: [
                r'\b(what is|who is|when was|where is|define)\b',
                r'\b(explain|describe|tell me about)\b',
                r'\b(fact|information|details)\b'
            ],
            DataSource.WEB_SEARCH: [
                r'\b(latest|recent|news|today|current)\b',
                r'\b(stock|weather|sports|breaking)\b',
                r'\b(202[0-9]|trending)\b'
            ],
            DataSource.DATABASE: [
                r'\b(count|sum|average|max|min)\b.*\b(from|in)\b',
                r'\b(select|insert|update|delete)\b',
                r'\b(table|row|column|record)\b'
            ],
            DataSource.API_ENDPOINT: [
                r'\b(convert|calculate|translate)\b',
                r'\b(exchange rate|currency|weather)\b',
                r'\b(api|endpoint|service)\b'
            ],
            DataSource.DOCUMENT_STORE: [
                r'\b(document|file|pdf|report)\b',
                r'\b(search for|find in|look up)\b',
                r'\b(manual|guide|specification)\b'
            ]
        }

        # Priority order for conflicting matches
        self.priority_order = [
            DataSource.DATABASE,
            DataSource.API_ENDPOINT,
            DataSource.WEB_SEARCH,
            DataSource.KNOWLEDGE_BASE,
            DataSource.DOCUMENT_STORE
        ]

    def route_query(self, query: str) -> DataSource:
        """
        Route a query to the appropriate data source using logical rules.

        Args:
            query: The user's query string

        Returns:
            DataSource: The recommended data source for the query
        """
        query_lower = query.lower()
        matches = {}

        # Check each data source against routing rules
        for source, patterns in self.routing_rules.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1

            if score > 0:
                matches[source] = score

        # If no matches, default to knowledge base
        if not matches:
            return DataSource.KNOWLEDGE_BASE

        # Return the source with highest score
        # In case of tie, use priority order
        best_source = max(matches.items(),
                         key=lambda x: (x[1], self.priority_order.index(x[0])))

        return best_source[0]

    def add_routing_rule(self, source: DataSource, pattern: str):
        """
        Add a custom routing rule for a data source.

        Args:
            source: The data source to add the rule for
            pattern: Regex pattern to match against queries
        """
        if source not in self.routing_rules:
            self.routing_rules[source] = []
        self.routing_rules[source].append(pattern)

    def get_route_explanation(self, query: str) -> Dict[str, Any]:
        """
        Get detailed explanation of routing decision.

        Args:
            query: The user's query string

        Returns:
            Dict containing routing decision and explanation
        """
        query_lower = query.lower()
        source_scores = {}

        for source, patterns in self.routing_rules.items():
            matched_patterns = []
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matched_patterns.append(pattern)

            if matched_patterns:
                source_scores[source] = {
                    'score': len(matched_patterns),
                    'matched_patterns': matched_patterns
                }

        if not source_scores:
            selected_source = DataSource.KNOWLEDGE_BASE
            explanation = "No specific patterns matched, defaulting to knowledge base"
        else:
            # Select best source
            selected_source = max(source_scores.items(),
                                key=lambda x: (x[1]['score'],
                                             self.priority_order.index(x[0])))[0]
            explanation = f"Matched {source_scores[selected_source]['score']} patterns"

        return {
            'query': query,
            'selected_source': selected_source.value,
            'explanation': explanation,
            'all_scores': {source.value: data['score']
                          for source, data in source_scores.items()}
        }


# Example usage
if __name__ == "__main__":
    router = LogicalRouter()

    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Latest news about AI developments",
        "Count the number of users in the database",
        "Convert 100 USD to EUR",
        "Find the PDF manual for installation",
        "Explain how photosynthesis works"
    ]

    print("Logical Routing Examples:")
    print("=" * 50)

    for query in test_queries:
        result = router.get_route_explanation(query)
        print(f"\nQuery: {query}")
        print(f"Routed to: {result['selected_source']}")
        print(f"Explanation: {result['explanation']}")