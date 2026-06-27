"""Web crawler: async fetch, robots compliance, URL frontier, dedup, extract."""

from .frontier import Frontier
from .robots import RobotsRules
from .dedup import SimHashDedup
from .parser import extract
from .fetcher import Fetcher, FetchResult
from .crawler import Crawler, CrawlConfig, CrawledPage

__all__ = [
    "Frontier",
    "RobotsRules",
    "SimHashDedup",
    "extract",
    "Fetcher",
    "FetchResult",
    "Crawler",
    "CrawlConfig",
    "CrawledPage",
]
