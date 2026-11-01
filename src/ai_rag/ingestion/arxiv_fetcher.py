"""Fetch papers from arXiv by query, category, or ID."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import arxiv

from ai_rag.config import settings

log = logging.getLogger(__name__)


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    categories: list[str]
    pdf_url: str
    pdf_path: Path | None = None


def fetch_papers(
    query: str | None = None,
    ids: list[str] | None = None,
    categories: list[str] | None = None,
    max_results: int = 20,
) -> list[Paper]:
    """Fetch papers from arXiv.

    Args:
        query: Search query (e.g. "diffusion models image generation").
        ids: Specific arXiv IDs (e.g. ["2312.00752"]).
        categories: arXiv categories (e.g. ["cs.CV", "cs.LG"]).
        max_results: Maximum number of papers to fetch.
    """
    settings.pdf_cache_dir.mkdir(parents=True, exist_ok=True)

    if ids:
        search = arxiv.Search(id_list=ids)
    else:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories) if categories else ""
        full_query = f"({query})" if query else ""
        if cat_filter:
            full_query = f"{full_query} AND ({cat_filter})" if full_query else cat_filter
        search = arxiv.Search(query=full_query.strip(), max_results=max_results)

    papers: list[Paper] = []
    client = arxiv.Client()
    for result in client.results(search):
        papers.append(
            Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                published=result.published.isoformat(),
                categories=result.categories,
                pdf_url=result.pdf_url,
            )
        )
    log.info("Fetched %d papers from arXiv", len(papers))
    return papers