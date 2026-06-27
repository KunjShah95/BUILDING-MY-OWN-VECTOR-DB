"""HTML -> clean text + title + links.

Prefers `trafilatura` (best boilerplate removal) when installed; falls back to a
stdlib html.parser extractor so the crawler has zero hard third-party deps.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Dict, List
from urllib.parse import urljoin, urlparse


def _extract_links(html: str, base_url: str) -> List[str]:
    links: List[str] = []
    parser = _LinkParser()
    try:
        parser.feed(html)
    except Exception:
        pass
    for href in parser.hrefs:
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        absolute = urljoin(base_url, href)
        scheme = urlparse(absolute).scheme
        if scheme in ("http", "https"):
            links.append(absolute)
    # Dedup, preserve order.
    seen = set()
    out = []
    for link in links:
        if link not in seen:
            seen.add(link)
            out.append(link)
    return out


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.hrefs.append(value)


class _TextParser(HTMLParser):
    """Collect visible text and the <title>, skipping script/style/nav noise."""

    _SKIP = {"script", "style", "noscript", "head", "svg", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title_parts: List[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.parts.append(text)


def _fallback_extract(html: str) -> Dict[str, str]:
    parser = _TextParser()
    try:
        parser.feed(html)
    except Exception:
        pass
    title = " ".join(" ".join(parser.title_parts).split())
    text = "\n".join(parser.parts)
    return {"title": title, "text": text}


def extract(html: str, url: str = "") -> Dict[str, object]:
    """Return {title, text, links} from raw HTML.

    Uses trafilatura for main-content extraction when available; otherwise a
    stdlib fallback. Links are always extracted with the stdlib parser so the
    frontier can be expanded regardless of backend.
    """
    title = ""
    text = ""
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            html, include_comments=False, include_tables=False, url=url or None
        )
        if extracted:
            text = extracted
        meta = trafilatura.extract_metadata(html)
        if meta and getattr(meta, "title", None):
            title = meta.title
    except Exception:
        pass

    if not text or not title:
        fb = _fallback_extract(html)
        text = text or fb["text"]
        title = title or fb["title"]

    links = _extract_links(html, url) if url else []
    return {"title": title.strip(), "text": text.strip(), "links": links}
