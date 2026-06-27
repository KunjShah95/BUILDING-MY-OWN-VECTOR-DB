"""Crawl orchestrator: ties together frontier, robots, fetcher, parser, dedup.

Async BFS crawl with per-host politeness (crawl-delay), robots.txt compliance,
near-duplicate skipping, and an `on_page` callback for each accepted page. The
callback is where ingestion (embed + BM25) hooks in.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Set, Union
from urllib.parse import urlparse

from .dedup import SimHashDedup
from .fetcher import Fetcher
from .frontier import Frontier, canonicalize
from .parser import extract
from .robots import RobotsRules, robots_url_for


@dataclass
class CrawledPage:
    url: str
    title: str
    text: str
    depth: int
    links: List[str] = field(default_factory=list)


@dataclass
class CrawlConfig:
    max_pages: int = 50
    max_depth: int = 3
    same_domain_only: bool = True
    allowed_domains: Optional[Set[str]] = None
    blocked_domains: Optional[Set[str]] = None
    user_agent: str = "VectorDBBot/0.1 (+https://example.com/bot)"
    respect_robots: bool = True
    min_text_len: int = 200            # skip thin pages
    default_delay: float = 0.5         # politeness when robots gives none


# on_page may be sync or async; both supported.
PageCallback = Callable[[CrawledPage], Union[None, Awaitable[None]]]


class Crawler:
    def __init__(
        self,
        config: Optional[CrawlConfig] = None,
        frontier: Optional[Frontier] = None,
        fetcher: Optional[Fetcher] = None,
    ):
        self.config = config or CrawlConfig()
        self.frontier = frontier or Frontier()
        self.fetcher = fetcher  # injected (real or fake); created if None at crawl()
        self.robots = RobotsRules(self.config.user_agent)
        self.dedup = SimHashDedup()
        self._last_host_access: dict = {}
        self.pages_crawled = 0
        self.stats = {"fetched": 0, "skipped_robots": 0, "skipped_dup": 0,
                      "skipped_thin": 0, "errors": 0}

    # ---- domain gating -----------------------------------------------------

    def _domain_allowed(self, url: str, seed_hosts: Set[str]) -> bool:
        host = urlparse(url).netloc.lower()
        if self.config.blocked_domains and host in self.config.blocked_domains:
            return False
        if self.config.allowed_domains is not None:
            return host in self.config.allowed_domains
        if self.config.same_domain_only:
            return host in seed_hosts
        return True

    async def _politeness_wait(self, url: str) -> None:
        host = urlparse(url).netloc.lower()
        delay = (self.robots.crawl_delay(url, self.config.default_delay)
                 if self.config.respect_robots else self.config.default_delay)
        last = self._last_host_access.get(host, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_host_access[host] = time.monotonic()

    async def _ensure_robots(self, url: str) -> None:
        if not self.config.respect_robots or self.robots.has_host(url):
            return
        txt = await self.fetcher.fetch_text(robots_url_for(url))
        self.robots.load(url, txt)

    async def _emit(self, on_page: PageCallback, page: CrawledPage) -> None:
        result = on_page(page)
        if asyncio.iscoroutine(result):
            await result

    # ---- main loop ---------------------------------------------------------

    async def crawl(self, seeds: List[str], on_page: PageCallback) -> dict:
        seed_hosts = {urlparse(canonicalize(s)).netloc.lower() for s in seeds}
        for s in seeds:
            self.frontier.add(s, depth=0, priority=1.0)

        own_fetcher = self.fetcher is None
        if own_fetcher:
            self.fetcher = Fetcher(user_agent=self.config.user_agent)
            await self.fetcher.__aenter__()

        try:
            while self.pages_crawled < self.config.max_pages:
                nxt = self.frontier.next()
                if nxt is None:
                    break
                url, depth = nxt

                if depth > self.config.max_depth:
                    self.frontier.mark_done(url)
                    continue
                if not self._domain_allowed(url, seed_hosts):
                    self.frontier.mark_done(url)
                    continue

                if self.config.respect_robots:
                    await self._ensure_robots(url)
                    if not self.robots.allowed(url):
                        self.stats["skipped_robots"] += 1
                        self.frontier.mark_done(url)
                        continue

                await self._politeness_wait(url)
                res = await self.fetcher.fetch(url)
                self.stats["fetched"] += 1
                if not res.ok:
                    self.stats["errors"] += 1
                    self.frontier.mark_failed(url)
                    continue

                parsed = extract(res.text, url=url)
                text = parsed["text"]
                if len(text) < self.config.min_text_len:
                    self.stats["skipped_thin"] += 1
                    self.frontier.mark_done(url)
                    # still expand links from thin hub pages
                    self._enqueue_links(parsed["links"], depth, seed_hosts)
                    continue

                if self.dedup.check_and_add(text):
                    self.stats["skipped_dup"] += 1
                    self.frontier.mark_done(url)
                    continue

                page = CrawledPage(url=url, title=parsed["title"], text=text,
                                   depth=depth, links=parsed["links"])
                await self._emit(on_page, page)
                self.pages_crawled += 1
                self.frontier.mark_done(url)
                self._enqueue_links(parsed["links"], depth, seed_hosts)
        finally:
            if own_fetcher:
                await self.fetcher.__aexit__(None, None, None)
                self.fetcher = None

        return {"pages_crawled": self.pages_crawled, **self.stats}

    def _enqueue_links(self, links: List[str], depth: int, seed_hosts: Set[str]) -> None:
        if depth + 1 > self.config.max_depth:
            return
        for link in links:
            if self._domain_allowed(link, seed_hosts):
                self.frontier.add(link, depth=depth + 1, priority=0.0)
