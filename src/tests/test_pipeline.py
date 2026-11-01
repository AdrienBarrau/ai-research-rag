"""Smoke tests (require GROQ_API_KEY for e2e)."""

from __future__ import annotations

import os

import pytest
from langchain_core.documents import Document

from ai_rag.rag.retriever import format_context


def test_format_context_truncates():
    docs = [
        Document(
            page_content="x" * 5000,
            metadata={"title": "A", "arxiv_id": "1", "page": 1},
        ),
        Document(
            page_content="y" * 5000,
            metadata={"title": "B", "arxiv_id": "2", "page": 1},
        ),
        Document(
            page_content="z" * 5000,
            metadata={"title": "C", "arxiv_id": "3", "page": 1},
        ),
    ]
    ctx = format_context(docs, max_chars=8000)
    assert "[1]" in ctx
    assert "[2]" in ctx
    assert "[3]" not in ctx


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY required",
)
def test_pipeline_smoke(tmp_env):
    from ai_rag.embeddings.embedder import get_embedder
    from ai_rag.rag.pipeline import RAGPipeline
    from ai_rag.vectorstore.store import add_documents, get_vectorstore

    store = get_vectorstore(get_embedder())
    add_documents(
        store,
        [
            Document(
                page_content="DPO outperforms RLHF on AlpacaEval 2.0 with 10% win-rate gain.",
                metadata={
                    "arxiv_id": "2305.18290",
                    "title": "DPO Paper",
                    "page": 1,
                    "pdf_url": "x",
                    "authors": "Rafailov",
                    "published": "2023",
                    "categories": "cs.LG",
                    "source": "x",
                },
            )
        ],
    )
    rag = RAGPipeline()
    result = rag.query("Compare DPO et RLHF")
    assert result.answer
    assert result.sources