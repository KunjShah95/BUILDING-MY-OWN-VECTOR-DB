"""Async HTTP fetcher built on httpx, with retry, timeout, and a polite UA."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional


@dataclass
class FetchResult:
    url: str
    status: int
    text: str
    content_type: str
    ok: bool
    error: Optional[str] = None


class Fetcher:
    """Thin async HTTP client. One instance per crawl; reuses a connection pool."""

    def __init__(
        self,
        user_agent: str = "VectorDBBot/0.1 (+https://example.com/bot)",
        timeout: float = 15.0,
        max_retries: int = 2,
        max_bytes: int = 5_000_000,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_bytes = max_bytes
        self._client = None

    async def __aenter__(self) -> "Fetcher":
        import httpx

        self._client = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> FetchResult:
        """GET *url* with retries. HTML/text only; oversized or binary -> not ok."""
        if self._client is None:
            raise RuntimeError("Fetcher must be used as an async context manager")

        last_err: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.get(url)
                ctype = resp.headers.get("content-type", "")
                if resp.status_code >= 400:
                    return FetchResult(url, resp.status_code, "", ctype, False,
                                       f"HTTP {resp.status_code}")
                if "html" not in ctype and "text" not in ctype and ctype:
                    return FetchResult(url, resp.status_code, "", ctype, False,
                                       f"unsupported content-type: {ctype}")
                content = resp.content[: self.max_bytes]
                text = content.decode(resp.encoding or "utf-8", errors="replace")
                return FetchResult(url, resp.status_code, text, ctype, True)
            except Exception as e:  # noqa: BLE001 - network errors are varied
                last_err = str(e)
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
        return FetchResult(url, 0, "", "", False, last_err)

    async def fetch_text(self, url: str) -> Optional[str]:
        """Best-effort fetch returning body text or None (used for robots.txt)."""
        res = await self.fetch(url)
        return res.text if res.ok else None
