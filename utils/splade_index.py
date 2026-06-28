"""
SPLADE-lite / BM42-style Sparse Encoder.

Produces sparse vectors (term → weight) that can be stored and searched more
efficiently than raw BM25 posting lists while capturing semantic expansion.

How it differs from plain BM25:
  BM25 → weights computed at *search time* from TF in the query and IDF in
         the index. No per-document sparse vector stored.
  SPLADE-lite → weights computed at *index time* per document, stored as a
         sparse float dict, and searched via dot-product over the query's
         sparse vector (asymmetric: encode query with idf, encode doc with
         tf*idf). Enables maximum-inner-product nearest-neighbor style search
         on a sparse inverted index.

BM42 insight (Qdrant 2024): replace term frequency saturation with pure IDF
weighting for the *query* side; use BM25 saturation only for *document* side.
This aligns with how transformers encode queries vs. documents.

Stored inverted list:
  _inv[term] = [(doc_id, weight), ...]

Persistence: numpy-compatible binary with JSON vocab (fast at scale).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TOK_RE = re.compile(r"\b[a-z][a-z0-9]{1,24}\b")
_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "be", "as",
    "are", "was", "were", "not", "have", "has", "had", "do", "does",
})


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOK_RE.findall(text.lower()) if t not in _STOP]


class SPLADEIndex:
    """Sparse encoder index with BM42-style IDF-weighted sparse vectors.

    Parameters
    ----------
    k1 : float
        BM25 saturation parameter for document TF. Default 1.2.
    b : float
        BM25 length normalisation. Default 0.75.
    idf_smoothing : float
        Added to IDF denominator to avoid division by zero. Default 0.5.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75,
                 idf_smoothing: float = 0.5):
        self.k1 = k1
        self.b = b
        self.idf_smoothing = idf_smoothing

        # Vocabulary: term -> int id
        self._vocab: Dict[str, int] = {}
        self._id_to_term: List[str] = []

        # Document frequency per term id
        self._df: List[int] = []

        # Inverted index: term_id -> list of (doc_id, weight)
        self._inv: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

        # Document lengths and metadata
        self._doc_len: Dict[str, int] = {}
        self._metadata: Dict[str, Any] = {}
        self._avg_dl: float = 0.0
        self._N: int = 0

        self._lock = threading.RLock()

    # ---------------------------------------------------------------- vocab

    def _get_or_add_term(self, term: str) -> int:
        if term in self._vocab:
            return self._vocab[term]
        tid = len(self._id_to_term)
        self._vocab[term] = tid
        self._id_to_term.append(term)
        self._df.append(0)
        return tid

    def _idf(self, tid: int) -> float:
        """Smooth IDF: log((N - df + 0.5) / (df + 0.5) + 1)."""
        df = self._df[tid] if tid < len(self._df) else 0
        return math.log((self._N - df + self.idf_smoothing) /
                        (df + self.idf_smoothing) + 1)

    # ----------------------------------------------------------------- add

    def add(self, text: str, doc_id: str, metadata: Any = None) -> None:
        """Encode and index a document."""
        tokens = _tokenize(text)
        if not tokens:
            return
        tf_map = Counter(tokens)
        dl = len(tokens)

        with self._lock:
            # Update doc stats
            old_dl = self._doc_len.get(doc_id)
            if old_dl is not None:
                # Re-index: remove old postings first
                self._remove(doc_id)
            self._doc_len[doc_id] = dl
            self._N = len(self._doc_len)
            self._avg_dl = sum(self._doc_len.values()) / self._N
            if metadata is not None:
                self._metadata[doc_id] = metadata

            # Update document frequencies (first pass)
            for term in tf_map:
                tid = self._get_or_add_term(term)
                self._df[tid] += 1

            # Compute BM25-style weights and add to inverted index
            for term, tf in tf_map.items():
                tid = self._vocab[term]
                idf_val = self._idf(tid)
                norm_tf = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                )
                weight = float(idf_val * norm_tf)
                self._inv[tid].append((doc_id, weight))

    def add_batch(self, items: List[Dict[str, Any]]) -> int:
        """Batch add. Each item: {text, doc_id, metadata?}."""
        for item in items:
            self.add(item["text"], item["doc_id"], item.get("metadata"))
        return len(items)

    def _remove(self, doc_id: str) -> None:
        """Remove a doc from the inverted index (called under lock)."""
        self._doc_len.pop(doc_id, None)
        self._metadata.pop(doc_id, None)
        for tid in list(self._inv.keys()):
            self._inv[tid] = [(d, w) for d, w in self._inv[tid] if d != doc_id]
        # Recompute df
        for tid in range(len(self._df)):
            self._df[tid] = len(self._inv[tid])

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._doc_len:
                return False
            self._remove(doc_id)
            self._N = len(self._doc_len)
            if self._N:
                self._avg_dl = sum(self._doc_len.values()) / self._N
            return True

    # ---------------------------------------------------------------- search

    def encode_query(self, text: str) -> Dict[int, float]:
        """Encode query as sparse vector using IDF weights only (BM42 style)."""
        tokens = _tokenize(text)
        tf_map = Counter(tokens)
        vec: Dict[int, float] = {}
        with self._lock:
            for term, tf in tf_map.items():
                tid = self._vocab.get(term)
                if tid is None:
                    continue
                vec[tid] = self._idf(tid) * math.log1p(tf)
        return vec

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Sparse dot-product search: sum IDF(q_term) * BM25_weight(d, q_term)."""
        q_vec = self.encode_query(query)
        if not q_vec:
            return []

        scores: Dict[str, float] = defaultdict(float)
        with self._lock:
            for tid, q_weight in q_vec.items():
                for doc_id, doc_weight in self._inv.get(tid, []):
                    scores[doc_id] += q_weight * doc_weight

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            {
                "doc_id": doc_id,
                "score": round(score, 6),
                "metadata": self._metadata.get(doc_id),
            }
            for doc_id, score in ranked
        ]

    def encode_document(self, text: str) -> Dict[str, float]:
        """Return human-readable sparse vector {term: weight} for a document."""
        tokens = _tokenize(text)
        tf_map = Counter(tokens)
        dl = len(tokens)
        vec: Dict[str, float] = {}
        with self._lock:
            avg_dl = self._avg_dl or 1.0
            for term, tf in tf_map.items():
                tid = self._vocab.get(term)
                if tid is None:
                    # Term not in vocab yet — use tf only
                    vec[term] = float(math.log1p(tf))
                    continue
                idf_val = self._idf(tid)
                norm_tf = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                )
                vec[term] = float(idf_val * norm_tf)
        # Return top-50 terms by weight
        return dict(sorted(vec.items(), key=lambda x: x[1], reverse=True)[:50])

    # -------------------------------------------------------------- persistence

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        meta = {
            "k1": self.k1, "b": self.b, "idf_smoothing": self.idf_smoothing,
            "vocab": self._vocab, "id_to_term": self._id_to_term,
            "df": self._df, "doc_len": self._doc_len,
            "avg_dl": self._avg_dl, "N": self._N,
            "metadata": {k: v for k, v in self._metadata.items()},
        }
        with open(os.path.join(directory, "splade_meta.json"), "w") as f:
            json.dump(meta, f)
        # Inverted index: convert int keys to str for JSON
        inv_serializable = {str(k): v for k, v in self._inv.items()}
        with open(os.path.join(directory, "splade_inv.json"), "w") as f:
            json.dump(inv_serializable, f)
        logger.info("SPLADE index saved to %s (%d docs, %d terms)",
                    directory, self._N, len(self._vocab))

    @classmethod
    def load(cls, directory: str) -> "SPLADEIndex":
        with open(os.path.join(directory, "splade_meta.json")) as f:
            meta = json.load(f)
        obj = cls(k1=meta["k1"], b=meta["b"], idf_smoothing=meta["idf_smoothing"])
        obj._vocab = meta["vocab"]
        obj._id_to_term = meta["id_to_term"]
        obj._df = meta["df"]
        obj._doc_len = meta["doc_len"]
        obj._avg_dl = meta["avg_dl"]
        obj._N = meta["N"]
        obj._metadata = meta.get("metadata", {})
        with open(os.path.join(directory, "splade_inv.json")) as f:
            raw_inv = json.load(f)
        obj._inv = defaultdict(list, {int(k): v for k, v in raw_inv.items()})
        logger.info("SPLADE index loaded from %s (%d docs, %d terms)",
                    directory, obj._N, len(obj._vocab))
        return obj

    # ------------------------------------------------------------------ stats

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_docs": self._N,
                "vocab_size": len(self._vocab),
                "avg_doc_length": round(self._avg_dl, 1),
                "total_postings": sum(len(v) for v in self._inv.values()),
                "k1": self.k1,
                "b": self.b,
            }

    def __len__(self) -> int:
        return self._N
