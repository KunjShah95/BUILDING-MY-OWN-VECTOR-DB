"""Sitemap.xml discovery and URL extraction.

Supports sitemap index files (which link to child sitemaps) and regular
sitemaps. Also reads Sitemap: directives from robots.txt as a fallback.
"""

from __future__ import annotations

from typing import List
from urllib.parse import urlparse
from xml.etree import ElementTree

_SM_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


def sitemap_url_for(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/sitemap.xml"


def _locs(xml_text: str) -> List[str]:
    """Extract all <loc> values from a sitemap or sitemapindex XML."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []
    return [el.text.strip() for el in root.iter(f"{{{_SM_NS}}}loc") if el.text]


def _is_sitemap_index(xml_text: str) -> bool:
    try:
        root = ElementTree.fromstring(xml_text)
        return "sitemapindex" in root.tag
    except ElementTree.ParseError:
        return False


async def discover_urls(fetcher, site_url: str, max_urls: int = 1000) -> List[str]:
    """Fetch sitemap(s) for *site_url* and return all discovered page URLs."""
    sitemap_url = sitemap_url_for(site_url)
    xml = await fetcher.fetch_text(sitemap_url)

    # Fallback: read Sitemap: line from robots.txt.
    if not xml:
        p = urlparse(site_url)
        robots_txt = await fetcher.fetch_text(f"{p.scheme}://{p.netloc}/robots.txt")
        if robots_txt:
            for line in robots_txt.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    xml = await fetcher.fetch_text(sitemap_url)
                    break

    if not xml:
        return []

    locs = _locs(xml)
    if not locs:
        return []

    # Sitemap index → fetch child sitemaps (limited).
    if _is_sitemap_index(xml):
        page_urls: List[str] = []
        for child_url in locs[:10]:
            child_xml = await fetcher.fetch_text(child_url)
            if child_xml:
                page_urls.extend(_locs(child_xml))
            if len(page_urls) >= max_urls:
                break
        return page_urls[:max_urls]

    return locs[:max_urls]
