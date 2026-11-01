"""Retriever with optional reranking."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from ai_rag.config import settings


def get_retriever(store: VectorStore, k: int | None = None):
    k = k or settings.retriever_k
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def format_context(docs: list[Document], max_chars: int = 12_000) -> str:
    """Format retrieved docs into a context string with citations."""
    parts, total = [], 0
    for i, d in enumerate(docs, 1):
        header = f"[{i}] {d.metadata.get('title', '?')} ({d.metadata.get('arxiv_id', '?')}, p.{d.metadata.get('page', '?')})"
        block = f"{header}\n{d.page_content.strip()}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)