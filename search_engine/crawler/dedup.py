"""Near-duplicate content detection via SimHash + Hamming distance.

Catches pages that are textually near-identical (mirrors, boilerplate-heavy
templates, reposts) even when their URLs differ. Pure stdlib (hashlib) — no deps.
"""

from __future__ import annotations

import hashlib
import re
from typing import List

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _hash64(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def simhash(text: str, bits: int = 64) -> int:
    """Compute the SimHash fingerprint of *text*."""
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return 0
    # Use word shingles (bigrams) for more robust similarity than single tokens.
    shingles = tokens if len(tokens) < 2 else [
        f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)
    ]
    weights = [0] * bits
    for sh in shingles:
        h = _hash64(sh)
        for i in range(bits):
            if (h >> i) & 1:
                weights[i] += 1
            else:
                weights[i] -= 1
    fingerprint = 0
    for i in range(bits):
        if weights[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class SimHashDedup:
    """Tracks seen fingerprints; flags near-duplicates within a Hamming threshold."""

    def __init__(self, bits: int = 64, threshold: int = 3):
        self.bits = bits
        self.threshold = threshold
        self._seen: List[int] = []

    def is_duplicate(self, text: str) -> bool:
        fp = simhash(text, self.bits)
        return any(hamming(fp, seen) <= self.threshold for seen in self._seen)

    def add(self, text: str) -> int:
        fp = simhash(text, self.bits)
        self._seen.append(fp)
        return fp

    def check_and_add(self, text: str) -> bool:
        """Return True if duplicate (and do NOT add); else add and return False."""
        fp = simhash(text, self.bits)
        if any(hamming(fp, seen) <= self.threshold for seen in self._seen):
            return True
        self._seen.append(fp)
        return False

    def __len__(self) -> int:
        return len(self._seen)
