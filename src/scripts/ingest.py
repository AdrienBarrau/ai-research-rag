"""CLI to ingest papers into the vector store."""

from __future__ import annotations

import argparse
import logging

from rich.logging import RichHandler

from ai_rag.config import settings
from ai_rag.embeddings.embedder import get_embedder
from ai_rag.ingestion.arxiv_fetcher import fetch_papers
from ai_rag.ingestion.pdf_parser import parse_pdf
from ai_rag.vectorstore.store import add_documents, get_vectorstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--categories", nargs="+")
    p.add_argument("--max", type=int, default=20)
    args = p.parse_args()

    papers = fetch_papers(
        query=args.query, ids=args.ids, categories=args.categories, max_results=args.max
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = []
    for p in papers:
        chunks.extend(splitter.split_documents(parse_pdf(p)))
    add_documents(get_vectorstore(get_embedder()), chunks)
    log.info("✅ %d papers → %d chunks", len(papers), len(chunks))


if __name__ == "__main__":
    main()