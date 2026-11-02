"""Quality presets for RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalPreset:
    """Configuration pour un preset de qualité."""
    name: str
    k: int  # Nombre de chunks à récupérer
    ef: int  # Paramètre HNSW pour Qdrant
    use_reranker: bool
    llm_model: str
    description: str


# Presets disponibles
PRESETS = {
    "fast": RetrievalPreset(
        name="fast",
        k=3,
        ef=64,
        use_reranker=False,
        llm_model="qwen/qwen3.8-27b",  # Ultra rapide
        description="⚡ Rapide (~2-3s) - Idéal pour des questions simples",
    ),
    "balanced": RetrievalPreset(
        name="balanced",
        k=8,
        ef=128,
        use_reranker=False,
        llm_model="qwen/qwen3.8-27b",  # Ton modèle préféré
        description="⚖️ Équilibré (~5-7s) - Bon compromis vitesse/précision",
    ),
    "precise": RetrievalPreset(
        name="precise",
        k=15,
        ef=256,
        use_reranker=True,
        llm_model="qwen/qwen3.8-27b",  # Ton modèle préféré
        description="🎯 Précis (~10-15s) - Maximum de précision",
    ),
}


def get_preset(name: str) -> RetrievalPreset:
    """Get a preset by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]