"""URL frontier: a persistent, prioritized crawl queue backed by SQLite.

Dedups URLs (canonicalized, fragment-stripped), tracks crawl status, and pops
the highest-priority pending URL. SQLite makes it survive process restarts so a
crawl can be paused and resumed.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Optional, Tuple
from urllib.parse import urldefrag, urlparse, urlunparse, parse_qsl, urlencode


def canonicalize(url: str) -> str:
    """Normalize a URL for dedup: strip fragment, lowercase host, drop tracking params."""
    url, _frag = urldefrag(url.strip())
    parts = urlparse(url)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    # Drop common tracking params, keep the rest sorted for stability.
    drop = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "gclid", "fbclid", "ref", "ref_src"}
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if k not in drop))
    path = parts.path or "/"
    return urlunparse((scheme, netloc, path, parts.params, query, ""))


class Frontier:
    """Persistent prioritized URL queue."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS frontier (
                url       TEXT PRIMARY KEY,
                depth     INTEGER NOT NULL DEFAULT 0,
                priority  REAL    NOT NULL DEFAULT 0.0,
                status    TEXT    NOT NULL DEFAULT 'pending',
                added     REAL    NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON frontier(status)")
        self.conn.commit()

    def add(self, url: str, depth: int = 0, priority: float = 0.0) -> bool:
        """Enqueue a URL. Returns True if newly added, False if already seen."""
        curl = canonicalize(url)
        cur = self.conn.execute("SELECT 1 FROM frontier WHERE url = ?", (curl,))
        if cur.fetchone():
            return False
        self.conn.execute(
            "INSERT INTO frontier (url, depth, priority, status, added) "
            "VALUES (?, ?, ?, 'pending', ?)",
            (curl, depth, priority, time.time()),
        )
        self.conn.commit()
        return True

    def next(self) -> Optional[Tuple[str, int]]:
        """Pop the highest-priority pending URL, mark it in_progress."""
        cur = self.conn.execute(
            "SELECT url, depth FROM frontier WHERE status = 'pending' "
            "ORDER BY priority DESC, depth ASC, added ASC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        url, depth = row
        self.conn.execute("UPDATE frontier SET status = 'in_progress' WHERE url = ?", (url,))
        self.conn.commit()
        return url, depth

    def mark_done(self, url: str) -> None:
        self._set_status(url, "done")

    def mark_failed(self, url: str) -> None:
        self._set_status(url, "failed")

    def _set_status(self, url: str, status: str) -> None:
        self.conn.execute(
            "UPDATE frontier SET status = ? WHERE url = ?", (status, canonicalize(url))
        )
        self.conn.commit()

    def seen(self, url: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM frontier WHERE url = ?", (canonicalize(url),))
        return cur.fetchone() is not None

    def pending_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM frontier WHERE status = 'pending'")
        return int(cur.fetchone()[0])

    def count(self, status: Optional[str] = None) -> int:
        if status is None:
            cur = self.conn.execute("SELECT COUNT(*) FROM frontier")
        else:
            cur = self.conn.execute("SELECT COUNT(*) FROM frontier WHERE status = ?", (status,))
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self.conn.close()
