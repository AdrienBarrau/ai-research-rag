"""Retriever hybride : local d'abord, fallback live si nécessaire."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from ai_rag.config import settings
from ai_rag.rag.live_arxiv import search_live

log = logging.getLogger(__name__)


@dataclass
class HybridRetrievalResult:
    documents: list[Document]
    source: str  # "local", "live", ou "hybrid"
    max_similarity_score: float


def retrieve_hybrid(query: str, store: VectorStore, k: int | None = None) -> HybridRetrievalResult:
    """Retrieval hybride avec fallback live.
    
    1. Cherche dans la base locale
    2. Si le meilleur score < seuil → fallback live arXiv
    3. Combine les résultats
    """
    k = k or settings.retriever_k
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # 1. Recherche locale
    local_docs: list[Document] = retriever.invoke(query)
    
    # Calcule le score de similarité du meilleur résultat
    # Chroma ne retourne pas les scores par défaut, on utilise similarity_search_with_score
    results_with_scores = store.similarity_search_with_score(query, k=k)
    max_score = 1.0 - results_with_scores[0][1] if results_with_scores else 0.0  # Chroma retourne des distances
    
    log.info("📊 Score local max: %.3f (seuil: %.3f)", max_score, settings.retrieval_threshold)
    
    # 2. Décision : fallback live ?
    if max_score >= settings.retrieval_threshold:
        log.info("✅ Retrieval local suffisant")
        return HybridRetrievalResult(
            documents=local_docs,
            source="local",
            max_similarity_score=max_score,
        )
    
    if not settings.live_fallback_enabled:
        log.info("⚠️ Retrieval local faible mais fallback désactivé")
        return HybridRetrievalResult(
            documents=local_docs,
            source="local",
            max_similarity_score=max_score,
        )
    
    # 3. Fallback live
    log.info("🔄 Fallback live arXiv activé...")
    live_docs = search_live(query)
    
    if not live_docs:
        log.info("❌ Fallback live n'a rien trouvé")
        return HybridRetrievalResult(
            documents=local_docs,
            source="local",
            max_similarity_score=max_score,
        )
    
    # Combine : live docs en priorité, puis local docs
    combined = live_docs[:k] + local_docs[:k//2]
    
    return HybridRetrievalResult(
        documents=combined,
        source="hybrid",
        max_similarity_score=max_score,
    )