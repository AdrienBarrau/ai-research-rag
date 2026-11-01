"""Fallback live : recherche arXiv + parsing + embedding à la volée."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import arxiv

from ai_rag.config import settings
from ai_rag.embeddings.embedder import get_embedder
from ai_rag.ingestion.pdf_parser import parse_pdf
from ai_rag.ingestion.arxiv_fetcher import Paper
from ai_rag.vectorstore.store import add_documents, get_vectorstore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)


def search_live(query: str, max_papers: int | None = None) -> list[Document]:
    """Recherche en direct sur arXiv, parse et embed les papiers trouvés.
    
    Les PDFs sont mis en cache dans data/pdfs/ pour ne pas être re-téléchargés.
    """
    max_papers = max_papers or settings.live_max_papers
    log.info(" Fallback live arXiv pour: %s", query)
    
    # Recherche sur les N derniers jours
    from_date = datetime.now(timezone.utc) - timedelta(days=settings.live_search_days)
    
    search = arxiv.Search(
        query=query,
        max_results=max_papers * 2,  # On en cherche plus au cas où certains échouent
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    
    papers: list[Paper] = []
    client = arxiv.Client()
    for result in client.results(search):
        if result.published.replace(tzinfo=timezone.utc) < from_date:
            continue
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
        if len(papers) >= max_papers:
            break
    
    if not papers:
        log.warning("Aucun papier trouvé sur arXiv pour: %s", query)
        return []
    
    log.info(" %d papiers trouvés, parsing en cours...", len(papers))
    
    # Parse les PDFs (avec cache automatique)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    all_chunks: list[Document] = []
    for paper in papers:
        try:
            pages = parse_pdf(paper)
            chunks = splitter.split_documents(pages)
            # Marque les chunks comme "live" pour les distinguer dans l'UI
            for c in chunks:
                c.metadata["source_type"] = "live"
            all_chunks.extend(chunks)
        except Exception as e:
            log.warning("Échec du parsing de %s: %s", paper.arxiv_id, e)
            continue
    
    if not all_chunks:
        return []
    
    # Indexe dans Chroma (sert de cache pour les prochaines requêtes)
    store = get_vectorstore(get_embedder())
    add_documents(store, all_chunks)
    log.info(" %d chunks live indexés dans le cache", len(all_chunks))
    
    return all_chunks