"""RSS 2.0 / Atom 1.0 feed ingestion.

Parses feeds and returns article dicts suitable for WebIngestor.ingest_page().
Supports both formats with a single entry point.
"""

from __future__ import annotations

from typing import Dict, List
from xml.etree import ElementTree

_ATOM_NS = "http://www.w3.org/2005/Atom"
_CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"


def _text(el: ElementTree.Element, tag: str, ns: str = "") -> str:
    full = f"{{{ns}}}{tag}" if ns else tag
    child = el.find(full)
    return (child.text or "").strip() if child is not None else ""


def parse_feed(xml_text: str) -> List[Dict[str, str]]:
    """Parse RSS or Atom feed. Returns list of {url, title, text}."""
    articles: List[Dict[str, str]] = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return articles

    tag = root.tag

    # ── RSS 2.0 ──────────────────────────────────────────────────────────────
    if "rss" in tag.lower() or root.find("channel") is not None:
        channel = root.find("channel") or root
        for item in channel.findall("item"):
            url = _text(item, "link") or _text(item, "guid")
            title = _text(item, "title")
            content = _text(item, "encoded", _CONTENT_NS)
            desc = _text(item, "description")
            text = content or desc
            if url and (title or text):
                articles.append({"url": url, "title": title, "text": text})

    # ── Atom 1.0 ─────────────────────────────────────────────────────────────
    elif _ATOM_NS in tag:
        for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
            link_el = entry.find(f"{{{_ATOM_NS}}}link")
            url = link_el.get("href", "") if link_el is not None else ""
            title = _text(entry, "title", _ATOM_NS)
            summary = _text(entry, "summary", _ATOM_NS)
            content_el = entry.find(f"{{{_ATOM_NS}}}content")
            content = (content_el.text or "").strip() if content_el is not None else ""
            text = content or summary
            if url and (title or text):
                articles.append({"url": url, "title": title, "text": text})

    return articles


async def ingest_feed(url: str, fetcher, ingestor) -> int:
    """Fetch *url* as a feed, parse it, ingest all articles. Returns count."""
    xml = await fetcher.fetch_text(url)
    if not xml:
        return 0
    articles = parse_feed(xml)
    for art in articles:
        try:
            ingestor.ingest_page(art["url"], art["title"], art["text"],
                                 metadata={"source": "feed", "feed_url": url})
        except Exception:
            pass
    return len(articles)
