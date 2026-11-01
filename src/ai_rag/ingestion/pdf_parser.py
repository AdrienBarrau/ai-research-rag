"""Download and parse PDFs into LangChain Documents."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import fitz  # pymupdf
from langchain_core.documents import Document

from ai_rag.config import settings
from ai_rag.ingestion.arxiv_fetcher import Paper

log = logging.getLogger(__name__)


def download_pdf(paper: Paper) -> Path:
    """Download a PDF and cache it locally."""
    settings.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    path = settings.pdf_cache_dir / f"{paper.arxiv_id}.pdf"
    if path.exists():
        return path
    with httpx.stream("GET", paper.pdf_url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return path


def parse_pdf(paper: Paper) -> list[Document]:
    """Parse a PDF into one Document per page, with rich metadata."""
    pdf_path = paper.pdf_path or download_pdf(paper)
    doc = fitz.open(pdf_path)
    pages: list[Document] = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if not text.strip():
            continue
        pages.append(
            Document(
                page_content=text,
                metadata={
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": ", ".join(paper.authors[:5]),
                    "published": paper.published,
                    "categories": ",".join(paper.categories),
                    "pdf_url": paper.pdf_url,
                    "page": i + 1,
                    "source": paper.pdf_url,
                },
            )
        )
    log.info("Parsed %d pages from %s", len(pages), paper.arxiv_id)
    return pages