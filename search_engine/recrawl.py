"""Freshness tracking + recrawl scheduling.

Records when each URL was last crawled and how often it tends to change, then
emits the set of URLs due for a recrawl. SQLite-backed so freshness survives
restarts. An adaptive interval shrinks for pages that change and grows for
pages that don't (bounded), keeping the crawl budget on volatile content.
"""

from __future__ import annotations

import sqlite3
import time
from typing import List, Optional


class FreshnessTracker:
    def __init__(self, db_path: str = ":memory:",
                 base_interval: float = 86_400.0,      # 1 day
                 min_interval: float = 3_600.0,        # 1 hour
                 max_interval: float = 2_592_000.0):   # 30 days
        self.conn = sqlite3.connect(db_path)
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS freshness (
                url           TEXT PRIMARY KEY,
                last_crawled  REAL NOT NULL,
                content_hash  INTEGER,
                interval      REAL NOT NULL,
                changes       INTEGER NOT NULL DEFAULT 0,
                checks        INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self.conn.commit()

    def record(self, url: str, content_hash: Optional[int] = None,
               now: Optional[float] = None) -> None:
        """Record a (re)crawl. Adapts the interval based on whether content changed."""
        now = now if now is not None else time.time()
        cur = self.conn.execute(
            "SELECT content_hash, interval, changes, checks FROM freshness WHERE url = ?",
            (url,),
        )
        row = cur.fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO freshness (url, last_crawled, content_hash, interval, changes, checks) "
                "VALUES (?, ?, ?, ?, 0, 1)",
                (url, now, content_hash, self.base_interval),
            )
        else:
            old_hash, interval, changes, checks = row
            changed = content_hash is not None and old_hash is not None and content_hash != old_hash
            if changed:
                interval = max(self.min_interval, interval * 0.5)   # changed -> check sooner
                changes += 1
            else:
                interval = min(self.max_interval, interval * 1.5)   # stable -> back off
            self.conn.execute(
                "UPDATE freshness SET last_crawled = ?, content_hash = ?, interval = ?, "
                "changes = ?, checks = ? WHERE url = ?",
                (now, content_hash if content_hash is not None else old_hash,
                 interval, changes, checks + 1, url),
            )
        self.conn.commit()

    def due(self, now: Optional[float] = None, limit: int = 100) -> List[str]:
        """URLs whose (last_crawled + interval) is in the past — ready to recrawl."""
        now = now if now is not None else time.time()
        cur = self.conn.execute(
            "SELECT url FROM freshness WHERE (last_crawled + interval) <= ? "
            "ORDER BY (last_crawled + interval) ASC LIMIT ?",
            (now, limit),
        )
        return [r[0] for r in cur.fetchall()]

    def next_due_at(self, url: str) -> Optional[float]:
        cur = self.conn.execute(
            "SELECT last_crawled + interval FROM freshness WHERE url = ?", (url,)
        )
        row = cur.fetchone()
        return float(row[0]) if row else None

    def stats(self) -> dict:
        cur = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(changes),0), COALESCE(AVG(interval),0) FROM freshness"
        )
        total, changes, avg_interval = cur.fetchone()
        return {"tracked": int(total), "total_changes": int(changes),
                "avg_interval_sec": float(avg_interval)}

    def close(self) -> None:
        self.conn.close()
