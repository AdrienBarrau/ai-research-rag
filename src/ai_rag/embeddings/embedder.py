"""Local embedding model via sentence-transformers."""

from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from ai_rag.config import settings

log = logging.getLogger(__name__)


def get_embedder() -> HuggingFaceEmbeddings:
    """Return a local embedding model.

    Le modèle est téléchargé au premier appel (~2.2 GB pour bge-m3),
    puis mis en cache dans ~/.cache/huggingface ou HF_HOME.
    """
    settings.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    # bge-m3 nécessite normalize_embeddings=True pour de bonnes performances
    normalize = "bge" in settings.embedding_model.lower()

    log.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "device": "cpu",  # passe à "cuda" si GPU dispo
        },
        encode_kwargs={
            "normalize_embeddings": normalize,
            "batch_size": 32,
        },
        cache_folder=str(settings.hf_cache_dir),
    )