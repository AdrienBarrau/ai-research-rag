"""Hugging Face Inference API pour les embeddings (0 CPU local)."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)


class HuggingFaceInferenceEmbeddings(Embeddings):
    """Embeddings via HF Inference API (gratuit, pas de CPU local)."""
    
    def __init__(self, model: str = "BAAI/bge-m3", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if not self.api_key:
            log.warning(" HF_TOKEN not set. Using unauthenticated API (rate limited).")
        
        # Import lazy pour éviter les erreurs si pas installé
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_key)
        except ImportError:
            raise ImportError("Please install huggingface-hub: pip install huggingface-hub")
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            try:
                # HF Inference API pour les embeddings
                response = self.client.feature_extraction(text, model=self.model)
                # bge-m3 retourne un vecteur de shape (1, 1024), on prend [0]
                if hasattr(response, 'shape') and len(response.shape) > 1:
                    embeddings.append(response[0].tolist())
                else:
                    embeddings.append(response.tolist())
            except Exception as e:
                log.error(f"Embedding failed for text: {e}")
                # Fallback: vecteur vide (sera ignoré par Qdrant)
                embeddings.append([0.0] * 1024)
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]