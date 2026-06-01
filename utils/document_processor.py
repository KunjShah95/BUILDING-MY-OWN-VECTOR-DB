"""Multi-format document processor."""
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


def extract_text_from_docx(path: str) -> str:
    """Extract text from DOCX file."""
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx required: pip install python-docx")
    doc = docx.Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_html(path: str) -> str:
    """Extract text from HTML file."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 required: pip install beautifulsoup4 lxml")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n").strip()


def extract_text_from_markdown(path: str) -> str:
    """Extract text from Markdown file."""
    try:
        import markdown
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("markdown + beautifulsoup4 required: pip install markdown beautifulsoup4")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        html = markdown.markdown(f.read())
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n").strip()


def extract_text(path: str) -> str:
    """Auto-detect format and extract text."""
    ext = os.path.splitext(path)[1].lower()
    extractors = {
        ".pdf": lambda p: _import_fitz(p),
        ".docx": extract_text_from_docx,
        ".html": extract_text_from_html,
        ".htm": extract_text_from_html,
        ".md": extract_text_from_markdown,
        ".markdown": extract_text_from_markdown,
        ".txt": lambda p: open(p, "r", encoding="utf-8", errors="replace").read(),
    }
    extractor = extractors.get(ext)
    if not extractor:
        raise ValueError(f"Unsupported file type: {ext}")
    return extractor(path)


def _import_fitz(pdf_path: str) -> str:
    from utils.pdf_processor import extract_text_from_pdf
    return extract_text_from_pdf(pdf_path)


def extract_images_from_pdf(path: str) -> List[bytes]:
    """Extract embedded images from a PDF file."""
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF required: pip install PyMuPDF")

    images = []
    doc = fitz.open(path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            images.append(base_image["image"])
    doc.close()
    return images
