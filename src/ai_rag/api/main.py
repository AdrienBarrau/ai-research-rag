"""FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ai_rag.api.schemas import (
    IngestRequest, IngestResponse, QueryRequest, QueryResponse, SourceOut,
)
from ai_rag.ingestion.arxiv_fetcher import fetch_papers
from ai_rag.ingestion.pdf_parser import parse_pdf
from ai_rag.rag.pipeline import RAGPipeline
from ai_rag.vectorstore.store import add_documents, get_vectorstore
from ai_rag.embeddings.embedder import get_embedder
from ai_rag.config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAGPipeline()
    yield


app = FastAPI(
    title="AI Research RAG",
    version="0.1.0",
    description="RAG over arXiv papers for AI research Q&A",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    try:
        result = app.state.rag.query(req.question)
    except Exception as e:
        log.exception("Query failed")
        raise HTTPException(500, str(e))
    sources = [
        SourceOut(
            arxiv_id=d.metadata["arxiv_id"],
            title=d.metadata["title"],
            page=d.metadata["page"],
            pdf_url=d.metadata["pdf_url"],
            snippet=d.page_content[:400],
        )
        for d in result.sources
    ]
    return QueryResponse(answer=result.answer, sources=sources)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    papers = fetch_papers(
        query=req.query, ids=req.ids, categories=req.categories, max_results=req.max_results
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    all_chunks = []
    for p in papers:
        pages = parse_pdf(p)
        all_chunks.extend(splitter.split_documents(pages))
    store = get_vectorstore(get_embedder())
    add_documents(store, all_chunks)
    return IngestResponse(papers_fetched=len(papers), chunks_indexed=len(all_chunks))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}