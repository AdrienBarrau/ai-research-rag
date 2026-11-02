"""End-to-end RAG pipeline avec presets de qualité."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from ai_rag.embeddings.embedder import get_embedder
from ai_rag.rag.generator import build_prompt, get_llm
from ai_rag.rag.presets import get_preset, RetrievalPreset
from ai_rag.rag.retriever import format_context
from ai_rag.vectorstore.store import get_collection_count, get_vectorstore, search_with_preset

log = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    sources: list[Document]
    context: str
    retrieval_source: str = "local"
    confidence_score: float = 0.0
    preset_used: str = "balanced"


class EmptyDatabaseError(Exception):
    pass


class RAGPipeline:
    def __init__(self) -> None:
        self.embedder = get_embedder()
        self.store = get_vectorstore(self.embedder)

    def query(self, question: str, preset_name: str = "balanced") -> RAGResult:
        """Query with a specific quality preset."""
        preset = get_preset(preset_name)
        
        count = get_collection_count(self.store)
        log.info("📚 Base locale: %d chunks | Preset: %s", count, preset_name)
        
        # Retrieval avec les paramètres du preset
        docs = search_with_preset(
            self.store,
            question,
            k=preset.k,
            ef=preset.ef,
        )
        
        # Formatage du contexte
        context = format_context(docs)
        
        # Génération avec le modèle du preset
        llm = get_llm(model_override=preset.llm_model)
        messages = build_prompt(question, context)
        answer = llm.invoke(messages).content
        
        return RAGResult(
            answer=answer,
            sources=docs,
            context=context,
            retrieval_source="local",
            confidence_score=0.0,  # TODO: calculer le score
            preset_used=preset_name,
        )