"""Ingestion curée : papiers les plus pertinents par domaine IA via arXiv API."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import arxiv
from rich.console import Console
from rich.progress import Progress

from ai_rag.config import settings
from ai_rag.embeddings.embedder import get_embedder
from ai_rag.ingestion.arxiv_fetcher import Paper, fetch_papers
from ai_rag.ingestion.pdf_parser import parse_pdf
from ai_rag.vectorstore.store import add_documents, get_vectorstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

console = Console()
log = logging.getLogger(__name__)

# Grands thèmes IA avec queries arXiv optimisées pour ramener les papiers fondateurs
AI_DOMAINS = {
    "Transformers & Attention": "all:transformer AND all:attention",
    "Diffusion Models": "all:diffusion AND all:generation",
    "Large Language Models": "all:large language model OR all:LLM",
    "RLHF & Alignment": "all:RLHF OR all:reinforcement learning from human feedback",
    "Computer Vision": "cat:cs.CV AND (all:object detection OR all:segmentation)",
    "Self-Supervised Learning": "all:self-supervised AND all:contrastive",
    "Reinforcement Learning": "cat:cs.LG AND all:reinforcement learning",
    "Graph Neural Networks": "all:graph neural network OR all:GNN",
    "Vision Transformers": "all:vision transformer OR all:ViT",
    "Multimodal Learning": "all:multimodal AND (all:vision OR all:language)",
}


def ingest_curated(papers_per_domain: int = 100) -> None:
    """Ingère les papiers les plus pertinents de chaque domaine IA."""
    console.print(f"[bold cyan] Ingestion curée : {papers_per_domain} papiers par domaine[/]")
    console.print(f"[cyan] {len(AI_DOMAINS)} domaines IA[/]")
    console.print(f"[cyan] Objectif : ~{len(AI_DOMAINS) * papers_per_domain} papiers[/]\n")
    
    all_papers: list[Paper] = []
    
    for domain, query in AI_DOMAINS.items():
        console.print(f"\n[bold yellow] Domaine:[/] {domain}")
        console.print(f"   Query: {query}")
        
        try:
            # Recherche arXiv triée par pertinence (ramène les papiers les plus cités/fondateurs en premier)
            search = arxiv.Search(
                query=query,
                max_results=papers_per_domain,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )
            client = arxiv.Client()
            
            domain_papers = []
            with console.status(f"[cyan]Recherche arXiv...", spinner="dots"):
                for result in client.results(search):
                    domain_papers.append(
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
            
            console.print(f"   [green] {len(domain_papers)} papiers trouvés[/]")
            if domain_papers:
                console.print(f"      Top 1: {domain_papers[0].title[:60]}...")
            
            all_papers.extend(domain_papers)
            
            # Rate limit arXiv : 1 requête / 3 secondes
            time.sleep(3)
            
        except Exception as e:
            console.print(f"   [red] Erreur: {e}[/]")
            continue
    
    console.print(f"\n[bold cyan] Total: {len(all_papers)} papiers uniques à ingérer[/]")
    
    # Dédoublonnage par arxiv_id
    seen_ids = set()
    unique_papers = []
    for p in all_papers:
        if p.arxiv_id not in seen_ids:
            seen_ids.add(p.arxiv_id)
            unique_papers.append(p)
    
    console.print(f"[cyan]🧹 Après dédoublonnage: {len(unique_papers)} papiers[/]")
    
    # Parse + Chunk + Embed
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    all_chunks = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Parsing PDFs...", total=len(unique_papers))
        for paper in unique_papers:
            try:
                pages = parse_pdf(paper)
                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)
            except Exception as e:
                log.warning("Failed to parse %s: %s", paper.arxiv_id, e)
            progress.update(task, advance=1)
    
    console.print(f"\n[bold cyan] Total chunks: {len(all_chunks)}[/]")
    
    # Estimation de la taille
    estimated_size_mb = len(all_chunks) * 7 / 1024
    console.print(f"[cyan]💾 Taille estimée Qdrant: ~{estimated_size_mb:.0f} MB[/]")
    
    if estimated_size_mb > 900:
        console.print("[yellow] Attention : proche de la limite 1 GB ![/]")
    
    # Index dans Qdrant
    console.print("[cyan] Indexation dans Qdrant Cloud...[/]")
    store = get_vectorstore(get_embedder())
    add_documents(store, all_chunks)
    
    console.print(f"\n[bold green] Terminé ! {len(all_chunks)} chunks indexés dans Qdrant[/]")
    console.print(f"[green] Taille estimée: ~{estimated_size_mb:.0f} MB / 1024 MB[/]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # On commence avec 100 par domaine pour tester rapidement (1000 papiers total)
    # Tu pourras augmenter à 500 une fois que tu as validé que tout fonctionne
    ingest_curated(papers_per_domain=100)