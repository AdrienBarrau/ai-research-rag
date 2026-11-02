"""Embedding model factory - utilise HF Inference API sur Streamlit Cloud."""

from __future__ import annotations

import logging
import os

from ai_rag.config import settings

log = logging.getLogger(__name__)


def get_embedder():
    """Return an embedding model.
    
    - Sur Streamlit Cloud : utilise HF Inference API (0 CPU)
    - En local : utilise sentence-transformers (GPU si disponible)
    """
    # Détecte si on est sur Streamlit Cloud
    is_streamlit_cloud = os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("DEPLOYMENT") == "cloud"
    
    if is_streamlit_cloud:
        log.info(" Streamlit Cloud detected - using HF Inference API")
        from ai_rag.embeddings.hf_inference import HuggingFaceInferenceEmbeddings
        return HuggingFaceInferenceEmbeddings(model=settings.embedding_model)
    else:
        log.info(" Local environment - using sentence-transformers")
        from langchain_huggingface import HuggingFaceEmbeddings
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        normalize = "bge" in settings.embedding_model.lower()
        
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": normalize,
                "batch_size": 64 if device == "cuda" else 32,
            },
            cache_folder=str(settings.hf_cache_dir),
        )