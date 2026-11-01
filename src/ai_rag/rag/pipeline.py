"""End-to-end RAG pipeline avec support hybride."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document

from ai_rag.embeddings.embedder import get_embedder
from ai_rag.rag.generator import build_prompt, get_llm
from ai_rag.rag.hybrid_retriever import retrieve_hybrid
from ai_rag.rag.retriever import format_context
from ai_rag.vectorstore.store import get_collection_count, get_vectorstore

log = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    sources: list[Document]
    context: str
    retrieval_source: str = "local"  # "local", "live", ou "hybrid"
    confidence_score: float = 0.0


class EmptyDatabaseError(Exception):
    pass


class RAGPipeline:
    def __init__(self) -> None:
        self.embedder = get_embedder()
        self.store = get_vectorstore(self.embedder)
        self.llm = get_llm()

    def query(self, question: str) -> RAGResult:
        count = get_collection_count(self.store)
        log.info("📚 Base locale: %d chunks", count)
        
        # Retrieval hybride (gère le fallback live automatiquement)
        retrieval_result = retrieve_hybrid(question, self.store)
        
        # Formatage du contexte
        context = format_context(retrieval_result.documents)
        
        # Génération
        messages = build_prompt(question, context)
        answer = self.llm.invoke(messages).content
        
        return RAGResult(
            answer=answer,
            sources=retrieval_result.documents,
            context=context,
            retrieval_source=retrieval_result.source,
            confidence_score=retrieval_result.max_similarity_score,
        )