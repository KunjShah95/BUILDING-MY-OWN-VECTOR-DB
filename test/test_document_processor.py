"""Tests for document processor."""
import pytest
import tempfile
from pathlib import Path


class TestDocumentProcessor:
    def test_extract_text_txt(self, tmp_path):
        from utils.document_processor import extract_text
        f = tmp_path / "test.txt"
        f.write_text("Hello world", encoding="utf-8")
        assert extract_text(str(f)) == "Hello world"

    def test_extract_text_md(self, tmp_path):
        from utils.document_processor import extract_text
        f = tmp_path / "test.md"
        f.write_text("# Heading\n\nParagraph text", encoding="utf-8")
        text = extract_text(str(f))
        assert "Heading" in text
        assert "Paragraph" in text

    def test_extract_text_unsupported(self, tmp_path):
        from utils.document_processor import extract_text
        f = tmp_path / "test.xyz"
        f.write_text("data")
        with pytest.raises(ValueError):
            extract_text(str(f))

    def test_extract_text_html(self, tmp_path):
        from utils.document_processor import extract_text
        f = tmp_path / "test.html"
        f.write_text("<html><body><p>Hello <b>world</b></p></html>", encoding="utf-8")
        text = extract_text(str(f))
        assert "Hello" in text
        assert "world" in text

    def test_extract_text_docx_not_installed(self):
        from utils.document_processor import extract_text_from_docx
        with pytest.raises(ImportError, match="python-docx"):
            extract_text_from_docx("/nonexistent/test.docx")

    def test_extract_text_html_rejects_nonexistent(self):
        from utils.document_processor import extract_text_from_html
        with pytest.raises(FileNotFoundError):
            extract_text_from_html("/nonexistent/test.html")

    def test_extract_text_markdown_rejects_nonexistent(self):
        from utils.document_processor import extract_text_from_markdown
        with pytest.raises(FileNotFoundError):
            extract_text_from_markdown("/nonexistent/test.md")

    def test_extract_images_from_pdf_function_exists(self):
        from utils.document_processor import extract_images_from_pdf
        assert callable(extract_images_from_pdf)
