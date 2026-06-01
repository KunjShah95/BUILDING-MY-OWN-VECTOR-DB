"""BM25 sparse retrieval index for hybrid search.

Implements Okapi BM25 for term-based retrieval, complementing dense vector search.
Stores document frequency statistics for incremental updates and can save/load to JSON.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


_WORD_RE = re.compile(r"\b[a-zA-Z0-9_]+\b")


class BM25Index:
    """Okapi BM25 sparse retrieval index.

    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter (default 1.5).
    b : float
        Length normalization parameter (default 0.75).

    Usage
    -----
    index = BM25Index()
    index.add_document("the cat sat on the mat", "doc1")
    index.add_document("the dog chased the cat", "doc2")
    results = index.search("cat", k=5)  # [(doc_id, score), ...]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # Per-document state
        self.doc_texts: Dict[str, str] = {}          # doc_id -> original text
        self.doc_lengths: Dict[str, int] = {}         # doc_id -> token count
        self.doc_freqs: Dict[str, int] = {}           # term -> # docs containing term

        # Aggregate
        self.total_docs: int = 0
        self.total_terms: int = 0
        self.avg_doc_length: float = 0.0

    # ---- Public API --------------------------------------------------------

    def add_document(self, text: str, doc_id: str) -> None:
        """Index a single document by its text content.

        If *doc_id* already exists the old entry is replaced.
        """
        self._remove_document(doc_id)  # remove stale entry if any
        tokens = self._tokenize(text)
        if not tokens:
            return

        # Update per-document data
        self.doc_texts[doc_id] = text
        self.doc_lengths[doc_id] = len(tokens)

        # Update collection-level stats
        self.total_docs += 1
        self.total_terms += len(tokens)
        self.avg_doc_length = self.total_terms / self.total_docs

        # Update document frequencies
        unique_terms = set(tokens)
        for term in unique_terms:
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def add_documents(self, texts: List[str], doc_ids: List[str]) -> None:
        """Batch-add documents."""
        for text, doc_id in zip(texts, doc_ids):
            self.add_document(text, doc_id)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Return up to *k* ``(doc_id, bm25_score)`` tuples sorted by score descending.

        Scores are raw BM25 (not normalised).  An empty list is returned when the
        index has no documents or the query contains no known terms.
        """
        if self.total_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Count query term frequency for multi-occurrence boosting
        query_tf = Counter(query_tokens)

        scores: Dict[str, float] = {}
        for doc_id, length in self.doc_lengths.items():
            score = 0.0
            for term, qtf in query_tf.items():
                df = self.doc_freqs.get(term, 0)
                if df == 0:
                    continue

                # BM25 components
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = self._term_frequency(term, self.doc_texts[doc_id])
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1.0 - self.b + self.b * length / self.avg_doc_length)
                )
                score += idf * tf_norm * qtf  # query term frequency multiplier

            if score > 0:
                scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:k]

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index.  Returns True if found."""
        return self._remove_document(doc_id)

    def clear(self) -> None:
        """Remove all documents from the index."""
        self.doc_texts.clear()
        self.doc_lengths.clear()
        self.doc_freqs.clear()
        self.total_docs = 0
        self.total_terms = 0
        self.avg_doc_length = 0.0

    def __len__(self) -> int:
        return self.total_docs

    @property
    def doc_ids(self) -> List[str]:
        return list(self.doc_texts.keys())

    # ---- Persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the index to a JSON file."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_texts": self.doc_texts,
            "doc_lengths": self.doc_lengths,
            "doc_freqs": self.doc_freqs,
            "total_docs": self.total_docs,
            "total_terms": self.total_terms,
            "avg_doc_length": self.avg_doc_length,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> BM25Index:
        """Deserialize an index from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(k1=data["k1"], b=data["b"])
        obj.doc_texts = data["doc_texts"]
        obj.doc_lengths = {k: int(v) for k, v in data["doc_lengths"].items()}
        obj.doc_freqs = data["doc_freqs"]
        obj.total_docs = int(data["total_docs"])
        obj.total_terms = int(data["total_terms"])
        obj.avg_doc_length = float(data["avg_doc_length"])
        return obj

    # ---- Internals ---------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Lower-case word tokenization."""
        return [t.lower() for t in _WORD_RE.findall(text)]

    def _term_frequency(self, term: str, text: str) -> int:
        """Count occurrences of *term* in *text* (lower-case match)."""
        return text.lower().count(term)

    def _remove_document(self, doc_id: str) -> bool:
        if doc_id not in self.doc_texts:
            return False

        old_text = self.doc_texts.pop(doc_id)
        old_len = self.doc_lengths.pop(doc_id)

        # Decrement doc frequencies
        old_tokens = set(self._tokenize(old_text))
        for term in old_tokens:
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) - 1
            if self.doc_freqs[term] <= 0:
                del self.doc_freqs[term]

        self.total_docs -= 1
        self.total_terms -= old_len
        self.avg_doc_length = self.total_terms / self.total_docs if self.total_docs > 0 else 0.0
        return True
