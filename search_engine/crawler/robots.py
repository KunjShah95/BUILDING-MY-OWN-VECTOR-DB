"""robots.txt compliance.

I/O-free by design: the crawler fetches robots.txt (async) and feeds the text
here for parsing. We cache one parser per host and answer allow/deny + crawl-delay.
"""

from __future__ import annotations

import urllib.robotparser
from typing import Dict, Optional
from urllib.parse import urlparse


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def robots_url_for(url: str) -> str:
    parts = urlparse(url)
    scheme = parts.scheme or "https"
    return f"{scheme}://{parts.netloc}/robots.txt"


class RobotsRules:
    """Per-host robots.txt rules with allow/deny and crawl-delay lookup."""

    def __init__(self, user_agent: str = "VectorDBBot"):
        self.user_agent = user_agent
        self._parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def load(self, url: str, robots_txt: Optional[str]) -> None:
        """Register parsed robots.txt for the host of *url*.

        Pass robots_txt=None (e.g. fetch failed / 404) to allow everything.
        """
        host = host_of(url)
        rp = urllib.robotparser.RobotFileParser()
        if robots_txt:
            rp.parse(robots_txt.splitlines())
        else:
            rp.parse([])  # empty rules => allow all
        self._parsers[host] = rp

    def has_host(self, url: str) -> bool:
        return host_of(url) in self._parsers

    def allowed(self, url: str) -> bool:
        rp = self._parsers.get(host_of(url))
        if rp is None:
            return True  # unknown host: allow (caller should load first)
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def crawl_delay(self, url: str, default: float = 1.0) -> float:
        rp = self._parsers.get(host_of(url))
        if rp is None:
            return default
        try:
            d = rp.crawl_delay(self.user_agent)
            return float(d) if d else default
        except Exception:
            return default
