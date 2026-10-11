"""
Inverted index implementation for keyword-based retrieval.
"""

import re
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from .base_index import BaseIndex
from .datastructures import DocumentChunk


class InvertedIndex(BaseIndex):
    """
    Inverted index implementation for keyword-based retrieval.

    Maps terms to the documents/chunks that contain them, enabling
    efficient keyword-based search with boolean and ranking capabilities.
    """

    def __init__(self):
        super().__init__("inverted")
        # Inverted index: term -> {chunk_id: frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        # Document frequency: term -> number of chunks containing term
        self.doc_freq: Dict[str, int] = {}
        # Field norms for length normalization
        self.field_norms: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - in practice would use more sophisticated tokenizer"""
        # Convert to lowercase and extract alphanumeric tokens
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Remove common stop words (simplified list)
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'can', 'will', 'just', 'should', 'now'
        }
        return [token for token in tokens if token not in stop_words and len(token) > 2]

    def _calculate_tf_idf(self, term_freq: int, doc_freq: int, total_chunks: int) -> float:
        """Calculate TF-IDF score"""
        if term_freq == 0 or doc_freq == 0:
            return 0.0
        tf = term_freq
        idf = math.log(total_chunks / doc_freq) if doc_freq > 0 else 0
        return tf * idf

    def add_document(self, document_id: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add document to inverted index"""
        if metadata is None:
            metadata = {}

        # Store document
        self.documents[document_id] = {
            'content': content,
            'metadata': metadata,
            'chunks': []
        }

        # For inverted index, we'll treat the whole document as one chunk for simplicity
        # In practice, you'd want to chunk first
        chunk_id = hashlib.md5(f"{document_id}:{content[:50]}".encode()).hexdigest()[:12]

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            start_index=0,
            end_index=len(content),
            metadata=metadata,
            tokens=self._tokenize(content)
        )

        self.chunks[chunk_id] = chunk
        self.documents[document_id]['chunks'].append(chunk_id)

        # Update inverted index
        term_freq: Dict[str, int] = {}
        for token in chunk.tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
            chunk.tokens.append(token)  # Store tokens in chunk

        # Update global structures
        for term, freq in term_freq.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
                self.doc_freq[term] = 0

            self.inverted_index[term][chunk_id] = freq
            self.doc_freq[term] += 1

        # Update stats
        self.stats.total_documents = len(self.documents)
        self.stats.total_chunks = len(self.chunks)
        self.stats.total_tokens = sum(len(chunk.tokens) for chunk in self.chunks.values())
        if self.stats.total_chunks > 0:
            self.stats.average_chunk_length = self.stats.total_tokens / self.stats.total_chunks
        self.stats.vocabulary_size = len(self.inverted_index)

        return [chunk_id]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using TF-IDF scoring"""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Calculate query term frequencies
        query_tf: Dict[str, int] = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1

        # Score documents using TF-IDF cosine similarity
        scores: Dict[str, float] = {}
        query_norm = 0.0

        # Calculate query vector norm
        for term, tf in query_tf.items():
            if term in self.doc_freq:
                idf = math.log(self.stats.total_chunks / self.doc_freq[term]) if self.doc_freq[term] > 0 else 0
                query_norm += (tf * idf) ** 2

        query_norm = math.sqrt(query_norm) if query_norm > 0 else 1.0

        # Calculate document scores
        for term, query_tf_val in query_tf.items():
            if term not in self.inverted_index:
                continue

            idf = math.log(self.stats.total_chunks / self.doc_freq[term]) if self.doc_freq[term] > 0 else 0
            query_weight = query_tf_val * idf

            for chunk_id, term_freq in self.inverted_index[term].items():
                # Get chunk document
                chunk = self.chunks.get(chunk_id)
                if not chunk:
                    continue

                doc_id = chunk.document_id

                # Calculate term frequency in document
                tf = term_freq

                # Calculate TF-IDF for document term
                if self.doc_freq[term] > 0:
                    idf_val = math.log(self.stats.total_chunks / self.doc_freq[term])
                    doc_weight = tf * idf_val
                else:
                    doc_weight = 0

                # Accumulate score
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += query_weight * doc_weight

        # Normalize scores by document length (simple approach)
        normalized_scores: Dict[str, float] = {}
        for doc_id, score in scores.items():
            doc = self.documents.get(doc_id)
            if doc and doc['chunks']:
                # Simple length normalization
                length_factor = 1.0 / math.sqrt(len(doc['chunks'])) if len(doc['chunks']) > 0 else 1.0
                normalized_scores[doc_id] = score * length_factor
            else:
                normalized_scores[doc_id] = score

        # Sort and return top-k
        sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def delete_document(self, document_id: str) -> bool:
        """Remove document from index"""
        if document_id not in self.documents:
            return False

        # Remove chunks and update inverted index
        chunk_ids = self.documents[document_id]['chunks']
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                # Remove term frequencies from inverted index
                for term in chunk.tokens:
                    if term in self.inverted_index and chunk_id in self.inverted_index[term]:
                        del self.inverted_index[term][chunk_id]
                        # Remove term if no longer in any chunk
                        if not self.inverted_index[term]:
                            del self.inverted_index[term]
                            self.doc_freq.pop(term, None)

                del self.chunks[chunk_id]

        # Remove document
        del self.documents[document_id]

        # Update stats
        self.stats.total_documents = len(self.documents)
        self.stats.total_chunks = len(self.chunks)
        self.stats.total_tokens = sum(len(chunk.tokens) for chunk in self.chunks.values())
        if self.stats.total_chunks > 0:
            self.stats.average_chunk_length = self.stats.total_tokens / self.stats.total_chunks
        self.stats.vocabulary_size = len(self.inverted_index)

        return True