"""Hugging Face Inference API pour les embeddings (0 CPU local)."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)


class HuggingFaceInferenceEmbeddings(Embeddings):
    """Embeddings via HF Inference API (gratuit, pas de CPU local)."""
    
    def __init__(self, model: str = "BAAI/bge-m3", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if not self.api_key:
            log.warning("⚠️ HF_TOKEN not set. Using unauthenticated API (rate limited).")
        
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_key)
        except ImportError:
            raise ImportError("Please install huggingface-hub: pip install huggingface-hub")
    
    def _embed_single(self, text: str) -> list[float]:
        """Embed a single text with robust type handling."""
        try:
            response = self.client.feature_extraction(text, model=self.model)
            
            # ✅ Gestion robuste du type de retour
            if isinstance(response, np.ndarray):
                vector = response.flatten().tolist()
            elif hasattr(response, 'tolist'):
                # PyTorch tensor ou numpy array
                vector = np.array(response).flatten().tolist()
            elif isinstance(response, list):
                # Liste imbriquée potentielle
                if response and isinstance(response[0], list):
                    vector = response[0]  # Prendre le premier embedding
                else:
                    vector = response
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
            
            return vector
            
        except Exception as e:
            log.error(f"Embedding failed: {e}")
            # Fallback: vecteur nul (sera ignoré)
            return [0.0] * 1024
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._embed_single(text) for text in texts]
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embed_single(text)