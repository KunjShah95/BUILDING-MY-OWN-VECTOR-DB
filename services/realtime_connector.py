"""
Real-Time Data Connectors (Phase 7: Production Search Orchestration).

Built-in web crawlers and headless-browser ingestion for capturing live data
from websites, RSS feeds, and REST APIs and automatically embedding them
into the vector database.

Supported connectors:
  - WebCrawler: crawls websites, extracts text, chunks, and indexes
  - RSSFeedConnector: monitors RSS/Atom feeds for new content
  - APIConnector: polls any REST API and ingests JSON responses
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)


@dataclass
class ConnectorConfig:
    """Base configuration for a real-time connector."""

    name: str
    collection_id: str
    interval_seconds: float = 300.0  # 5 min default
    max_items_per_run: int = 50
    embed_fn: Optional[Callable[[str], List[float]]] = None
    ingest_fn: Optional[Callable[[str, List[float], Dict], Any]] = None


@dataclass
class WebCrawlerConfig(ConnectorConfig):
    """Configuration for web crawler connector."""

    start_urls: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)
    max_pages: int = 100
    respect_robots_txt: bool = True
    user_agent: str = "VectorDB-Crawler/1.0"


@dataclass
class RSSFeedConfig(ConnectorConfig):
    """Configuration for RSS/Atom feed connector."""

    feed_urls: List[str] = field(default_factory=list)


@dataclass
class APIConnectorConfig(ConnectorConfig):
    """Configuration for REST API connector."""

    api_url: str = ""
    api_key: Optional[str] = None
    response_jsonpath: str = ""  # e.g. "data.items" to extract items from response
    text_field: str = "text"
    id_field: str = "id"


class RealTimeDataConnector:
    """Base class for real-time data connectors."""

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._seen_items: Set[str] = set()
        self._state_path = f"connector_state_{config.name}.json"
        self._load_state()

    def _load_state(self):
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path) as f:
                    data = json.load(f)
                self._seen_items = set(data.get("seen_items", []))
            except Exception:
                pass

    def _save_state(self):
        with open(self._state_path, "w") as f:
            json.dump({"seen_items": list(self._seen_items)[-10000:]}, f)

    def _make_item_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    def _ingest_item(self, text: str, metadata: Dict[str, Any]):
        """Process and ingest a single item."""
        item_id = self._make_item_id(text)
        if item_id in self._seen_items:
            return

        if self.config.embed_fn and text.strip():
            try:
                vector = self.config.embed_fn(text)
                if self.config.ingest_fn:
                    self.config.ingest_fn(
                        self.config.collection_id,
                        vector,
                        {**metadata, "text": text[:1000]},
                    )
                self._seen_items.add(item_id)
            except Exception as exc:
                logger.error("Ingest failed for item: %s", exc)

    def _collect_items(self) -> List[Dict[str, Any]]:
        """Override in subclasses. Returns list of (text, metadata) items."""
        raise NotImplementedError

    def _run_once(self):
        """Single collection cycle."""
        try:
            items = self._collect_items()
            for item in items[: self.config.max_items_per_run]:
                self._ingest_item(item.get("text", ""), item.get("metadata", {}))
            self._save_state()
            logger.info(
                "%s collected %d items", self.config.name, len(items)
            )
        except Exception as exc:
            logger.error("%s run failed: %s", self.config.name, exc)

    def _loop(self):
        while self._running:
            self._run_once()
            time.sleep(self.config.interval_seconds)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"connector-{self.config.name}"
        )
        self._thread.start()
        logger.info("Connector '%s' started (interval=%ss)", self.config.name, self.config.interval_seconds)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Connector '%s' stopped", self.config.name)

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "running": self._running,
            "collection_id": self.config.collection_id,
            "interval_seconds": self.config.interval_seconds,
            "seen_items": len(self._seen_items),
        }


class WebCrawlerConnector(RealTimeDataConnector):
    """Crawl websites and index their content."""

    def __init__(self, config: WebCrawlerConfig):
        super().__init__(config)
        self.config = config
        self._visited: Set[str] = set()

    def _collect_items(self) -> List[Dict[str, Any]]:
        """Simple web crawler using requests + BeautifulSoup."""
        items = []
        if not self.config.start_urls:
            return items

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not installed. Install: pip install beautifulsoup4")
            return items

        queue = list(self.config.start_urls)
        while queue and len(self._visited) < self.config.max_pages:
            url = queue.pop(0)
            if url in self._visited:
                continue
            self._visited.add(url)

            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": self.config.user_agent},
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")

                # Extract text
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)

                if text:
                    # Chunk: split by paragraph, keep first 2000 chars
                    text = text[:5000]
                    items.append({
                        "text": text,
                        "metadata": {
                            "source": url,
                            "title": soup.title.string if soup.title else "",
                            "crawled_at": time.time(),
                            "connector": "web_crawler",
                        },
                    })

                # Add internal links to queue
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if href.startswith("http") and any(
                        domain in href for domain in self.config.allowed_domains
                    ):
                        if href not in self._visited:
                            queue.append(href)

            except Exception as exc:
                logger.debug("Failed to crawl %s: %s", url, exc)

        return items


class RSSFeedConnector(RealTimeDataConnector):
    """Monitor RSS/Atom feeds for new content."""

    def __init__(self, config: RSSFeedConfig):
        super().__init__(config)
        self.config = config

    def _collect_items(self) -> List[Dict[str, Any]]:
        items = []
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed. Install: pip install feedparser")
            return items

        for feed_url in self.config.feed_urls:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[: self.config.max_items_per_run]:
                    text = f"{entry.get('title', '')}\n{entry.get('summary', '')}\n{entry.get('content', [{}])[0].get('value', '') if entry.get('content') else ''}"
                    items.append({
                        "text": text,
                        "metadata": {
                            "source": feed_url,
                            "title": entry.get("title", ""),
                            "link": entry.get("link", ""),
                            "published": entry.get("published", ""),
                            "connector": "rss_feed",
                        },
                    })
            except Exception as exc:
                logger.error("Failed to parse feed %s: %s", feed_url, exc)

        return items


class APIConnector(RealTimeDataConnector):
    """Poll a REST API and ingest structured data."""

    def __init__(self, config: APIConnectorConfig):
        super().__init__(config)
        self.config = config

    def _collect_items(self) -> List[Dict[str, Any]]:
        items = []
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            resp = requests.get(self.config.api_url, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.warning("API returned %d for %s", resp.status_code, self.config.api_url)
                return items

            data = resp.json()

            # Navigate JSON path
            if self.config.response_jsonpath:
                for part in self.config.response_jsonpath.split("."):
                    if isinstance(data, dict):
                        data = data.get(part, [])
                    elif isinstance(data, list):
                        break

            if isinstance(data, list):
                for entry in data[: self.config.max_items_per_run]:
                    text = str(entry.get(self.config.text_field, json.dumps(entry)))
                    item_id = entry.get(self.config.id_field, self._make_item_id(text))
                    items.append({
                        "text": text,
                        "metadata": {
                            "source": self.config.api_url,
                            "item_id": item_id,
                            "connector": "api",
                        },
                    })

        except Exception as exc:
            logger.error("API poll failed for %s: %s", self.config.api_url, exc)

        return items
