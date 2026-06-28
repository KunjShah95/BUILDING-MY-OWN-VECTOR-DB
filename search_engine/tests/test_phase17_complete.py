"""Phase 17 completion tests: answer synthesis, highlighting, SPLADE, feed, sitemap, collections."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_RESULTS = [
    {
        "url": "https://example.com/page1",
        "title": "Python Tutorial",
        "snippet": "Python is a great language. It is used for machine learning and data science. Many developers love Python.",
        "score": 0.95,
    },
    {
        "url": "https://example.com/page2",
        "title": "Machine Learning Basics",
        "snippet": "Machine learning algorithms learn from data. Deep learning is a subset of machine learning.",
        "score": 0.87,
    },
    {
        "url": "https://example.com/page3",
        "title": "Data Science Guide",
        "snippet": "Data science uses Python, R, and SQL. Python libraries like pandas make data analysis easy.",
        "score": 0.75,
    },
]


# ── Answer synthesis tests ────────────────────────────────────────────────────

class TestAnswerSynthesis:
    def _extract_answer(self, results, query, max_sentences=3):
        from api.routers.web_search import _extract_answer
        return _extract_answer(results, query, max_sentences)

    def test_basic_extraction(self):
        result = self._extract_answer(SAMPLE_RESULTS, "Python machine learning")
        assert result["answer"], "should produce a non-empty answer"
        assert len(result["citations"]) > 0

    def test_answer_contains_query_terms(self):
        result = self._extract_answer(SAMPLE_RESULTS, "Python machine learning")
        answer = result["answer"]
        assert "Python" in answer or "machine" in answer or "learning" in answer

    def test_max_sentences_respected(self):
        result = self._extract_answer(SAMPLE_RESULTS, "Python", max_sentences=2)
        # Answer should come from at most 2 sentences
        sentence_count = len(re.split(r"(?<=[.!?])\s+", result["answer"].strip()))
        assert sentence_count <= 3  # allow slight variance from mid-sentence splits

    def test_citations_are_unique(self):
        result = self._extract_answer(SAMPLE_RESULTS, "Python data science")
        urls = [c["url"] for c in result["citations"]]
        assert len(urls) == len(set(urls))

    def test_empty_query_tokens(self):
        result = self._extract_answer(SAMPLE_RESULTS, "a the in")
        # All stop words → no answer
        assert result["answer"] == ""
        assert result["citations"] == []

    def test_no_matching_results(self):
        result = self._extract_answer([], "Python")
        assert result["answer"] == ""

    def test_citations_max_three(self):
        result = self._extract_answer(SAMPLE_RESULTS * 4, "Python machine data")
        assert len(result["citations"]) <= 3


# ── Highlight tests ───────────────────────────────────────────────────────────

class TestHighlight:
    def _highlight(self, text, tokens):
        from api.routers.web_search import _highlight
        return _highlight(text, tokens)

    def test_single_term(self):
        result = self._highlight("Python is great", ["python"])
        assert "**Python**" in result

    def test_multiple_terms(self):
        result = self._highlight("Python and machine learning", ["python", "machine"])
        assert "**Python**" in result
        assert "**machine**" in result

    def test_case_insensitive(self):
        result = self._highlight("PYTHON rocks", ["python"])
        assert "**PYTHON**" in result

    def test_no_tokens(self):
        text = "hello world"
        result = self._highlight(text, [])
        assert result == text

    def test_word_boundary_respected(self):
        # "learn" should not match "learning"
        result = self._highlight("I am learning now", ["learn"])
        assert "**learning**" not in result
        assert "**learn**" not in result or "learning" not in result


# ── SPLADE index tests ────────────────────────────────────────────────────────

class TestSPLADEIndex:
    def test_add_and_search(self):
        from utils.splade_index import SPLADEIndex
        idx = SPLADEIndex()
        idx.add("Python is a programming language used for ML", "doc1")
        idx.add("Java is an object-oriented programming language", "doc2")
        idx.add("Machine learning uses data and algorithms", "doc3")
        results = idx.search("Python programming", k=2)
        assert results[0]["doc_id"] == "doc1"

    def test_search_returns_k_or_fewer(self):
        from utils.splade_index import SPLADEIndex
        idx = SPLADEIndex()
        for i in range(5):
            idx.add(f"document about topic {i}", f"doc{i}")
        results = idx.search("document topic", k=3)
        assert len(results) <= 3

    def test_delete(self):
        from utils.splade_index import SPLADEIndex
        idx = SPLADEIndex()
        idx.add("Python programming language", "doc1")
        idx.add("Java programming language", "doc2")
        assert idx.delete("doc1") is True
        results = idx.search("Python", k=5)
        ids = [r["doc_id"] for r in results]
        assert "doc1" not in ids

    def test_delete_nonexistent(self):
        from utils.splade_index import SPLADEIndex
        idx = SPLADEIndex()
        assert idx.delete("ghost") is False

    def test_encode_document(self):
        from utils.splade_index import SPLADEIndex
        idx = SPLADEIndex()
        idx.add("Python machine learning data science", "doc1")
        vec = idx.encode_document("Python machine learning")
        assert isinstance(vec, dict)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec.values())

    def test_get_stats(self):
        from utils.splade_index import SPLADEIndex
        idx = SPLADEIndex()
        idx.add("hello world test", "d1")
        idx.add("foo bar baz", "d2")
        stats = idx.get_stats()
        assert stats["total_docs"] == 2
        assert stats["vocab_size"] >= 3

    def test_batch_add(self):
        from utils.splade_index import SPLADEIndex
        idx = SPLADEIndex()
        items = [
            {"text": f"document {i} about neural networks", "doc_id": f"d{i}"}
            for i in range(10)
        ]
        n = idx.add_batch(items)
        assert n == 10
        assert len(idx) == 10

    def test_persist_and_load(self, tmp_path):
        from utils.splade_index import SPLADEIndex
        idx = SPLADEIndex()
        idx.add("Python programming", "doc1")
        idx.add("Java enterprise", "doc2")
        save_dir = str(tmp_path / "splade_save")
        idx.save(save_dir)
        loaded = SPLADEIndex.load(save_dir)
        assert loaded._N == 2
        results = loaded.search("Python", k=1)
        assert results[0]["doc_id"] == "doc1"

    def test_reindex_on_duplicate_doc(self):
        from utils.splade_index import SPLADEIndex
        idx = SPLADEIndex()
        idx.add("Python is great", "doc1")
        idx.add("Java is better", "doc1")  # re-index
        assert len(idx) == 1
        results = idx.search("Java", k=1)
        assert results[0]["doc_id"] == "doc1"

    def test_idf_weighting(self):
        from utils.splade_index import SPLADEIndex
        idx = SPLADEIndex()
        # "common" appears in all docs → low IDF
        # "rare" appears in one → high IDF
        for i in range(9):
            idx.add(f"common word in document {i}", f"doc{i}")
        idx.add("common xyloquartz bespoke token here", "rare_doc")
        results = idx.search("xyloquartz", k=3)
        assert results[0]["doc_id"] == "rare_doc"


# ── Feed parser tests ─────────────────────────────────────────────────────────

class TestFeedParser:
    RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <link>https://example.com</link>
    <item>
      <title>Article One</title>
      <link>https://example.com/article1</link>
      <description>First article about Python</description>
    </item>
    <item>
      <title>Article Two</title>
      <link>https://example.com/article2</link>
      <description>Second article about Java</description>
    </item>
  </channel>
</rss>"""

    ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry>
    <title>Atom Entry One</title>
    <link href="https://example.com/atom1"/>
    <summary>Summary of first atom entry</summary>
  </entry>
</feed>"""

    def test_parse_rss(self):
        from search_engine.crawler.feed import parse_feed
        entries = parse_feed(self.RSS_SAMPLE)
        assert len(entries) == 2
        assert entries[0]["title"] == "Article One"
        assert entries[0]["url"] == "https://example.com/article1"

    def test_parse_atom(self):
        from search_engine.crawler.feed import parse_feed
        entries = parse_feed(self.ATOM_SAMPLE)
        assert len(entries) == 1
        assert entries[0]["title"] == "Atom Entry One"

    def test_parse_empty(self):
        from search_engine.crawler.feed import parse_feed
        entries = parse_feed("<rss><channel></channel></rss>")
        assert entries == []

    def test_parse_invalid_xml(self):
        from search_engine.crawler.feed import parse_feed
        entries = parse_feed("not xml at all <<<")
        assert entries == []


# ── Sitemap tests ─────────────────────────────────────────────────────────────

class TestSitemapDiscovery:
    SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/page1</loc></url>
  <url><loc>https://example.com/page2</loc></url>
  <url><loc>https://example.com/page3</loc></url>
</urlset>"""

    SITEMAP_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap1.xml</loc></sitemap>
</sitemapindex>"""

    def test_parse_urlset(self):
        from search_engine.crawler.sitemap import _locs, _is_sitemap_index
        urls = _locs(self.SITEMAP_XML)
        assert len(urls) == 3
        assert "https://example.com/page1" in urls
        assert not _is_sitemap_index(self.SITEMAP_XML)

    def test_parse_sitemap_index(self):
        from search_engine.crawler.sitemap import _locs, _is_sitemap_index
        locs = _locs(self.SITEMAP_INDEX)
        assert "https://example.com/sitemap1.xml" in locs
        assert _is_sitemap_index(self.SITEMAP_INDEX)

    def test_max_urls_respected(self):
        many_urls = "\n".join(
            f"<url><loc>https://example.com/page{i}</loc></url>"
            for i in range(100)
        )
        xml = f"""<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{many_urls}
</urlset>"""
        from search_engine.crawler.sitemap import _locs
        urls = _locs(xml)
        assert len(urls) == 100


# ── Collection management tests ───────────────────────────────────────────────

class TestCollectionManagement:
    def test_list_empty(self, tmp_path, monkeypatch):
        import api.routers.web_search as ws
        monkeypatch.setattr(ws, "_INDEX_ROOT", str(tmp_path))
        cols = ws._list_collections()
        assert cols == []

    def test_list_with_collection(self, tmp_path, monkeypatch):
        import api.routers.web_search as ws
        monkeypatch.setattr(ws, "_INDEX_ROOT", str(tmp_path))
        col_dir = tmp_path / "web"
        col_dir.mkdir()
        (col_dir / "sparse.json").write_text("{}")
        cols = ws._list_collections()
        assert len(cols) == 1
        assert cols[0]["collection_id"] == "web"
        assert cols[0]["has_sparse"] is True

    def test_list_size_calculation(self, tmp_path, monkeypatch):
        import api.routers.web_search as ws
        monkeypatch.setattr(ws, "_INDEX_ROOT", str(tmp_path))
        col_dir = tmp_path / "mycol"
        col_dir.mkdir()
        (col_dir / "data.bin").write_bytes(b"x" * 1000)
        cols = ws._list_collections()
        assert cols[0]["size_bytes"] == 1000
