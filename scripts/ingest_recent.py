"""Ingestion mensuelle automatique des papiers récents sur arXiv."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import arxiv
from rich.console import Console
from rich.progress import Progress

from ai_rag.config import settings
from ai_rag.embeddings.embedder import get_embedder
from ai_rag.ingestion.arxiv_fetcher import Paper
from ai_rag.ingestion.pdf_parser import parse_pdf
from ai_rag.vectorstore.store import add_documents, get_vectorstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

console = Console()
log = logging.getLogger(__name__)

# Catégories IA principales
AI_CATEGORIES = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "stat.ML"]


def ingest_recent_papers(days_back: int = 30, max_results: int = 150) -> None:
    """Récupère et indexe les papiers récents des catégories IA."""
    console.print(f"[bold cyan] Ingestion des papiers des {days_back} derniers jours...[/]")
    
    # Date limite
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    # Construction de la requête arXiv
    cat_query = " OR ".join(f"cat:{cat}" for cat in AI_CATEGORIES)
    
    search = arxiv.Search(
        query=cat_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    client = arxiv.Client()
    papers_to_process = []
    
    with console.status("[cyan]Recherche des papiers récents sur arXiv...", spinner="dots"):
        for result in client.results(search):
            if result.published.replace(tzinfo=timezone.utc) < cutoff_date:
                break # On s'arrête dès qu'on dépasse la date limite
            
            papers_to_process.append(
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
            
            if len(papers_to_process) >= max_results:
                break

    console.print(f"[green] {len(papers_to_process)} nouveaux papiers trouvés.[/]")
    
    if not papers_to_process:
        console.print("[yellow]Aucun nouveau papier à ingérer.[/]")
        return

    # Parsing et Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    all_chunks = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Parsing et chunking...", total=len(papers_to_process))
        for paper in papers_to_process:
            try:
                pages = parse_pdf(paper)
                chunks = splitter.split_documents(pages)
                # Marquer comme ingestion auto pour le suivi
                for c in chunks:
                    c.metadata["ingestion_type"] = "automated_monthly"
                all_chunks.extend(chunks)
            except Exception as e:
                log.warning("Échec parsing %s: %s", paper.arxiv_id, e)
            progress.update(task, advance=1)

    console.print(f"[bold cyan] Total chunks générés: {len(all_chunks)}[/]")

    # Upsert dans Qdrant
    console.print("[cyan] Envoi vers Qdrant Cloud...[/]")
    store = get_vectorstore(get_embedder())
    
    # Grâce à nos UUIDv5 déterministes, Qdrant fera un "upsert" (mise à jour) 
    # et ne créera pas de doublons si un papier est déjà présent.
    add_documents(store, all_chunks)
    
    console.print(f"[bold green] Terminé ! {len(all_chunks)} chunks ajoutés/mis à jour dans Qdrant.[/]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_recent_papers(days_back=7, max_results=50)