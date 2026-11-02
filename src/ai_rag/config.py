"""Centralized configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (Groq - gratuit)
    groq_api_key: str = ""
    llm_model: str = "qwen/qwen3.8-27b"

    # Embeddings (local, gratuit)
    embedding_model: str = "BAAI/bge-m3"

    # Qdrant Cloud (remplace Chroma)
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "ai_papers"

    # LangSmith (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "ai-research-rag"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    retriever_k: int = 8
    retrieval_threshold: float = 0.65

    # Live fallback
    live_fallback_enabled: bool = True
    live_max_papers: int = 3
    live_search_days: int = 30

    # Paths
    pdf_cache_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    hf_cache_dir: Path = PROJECT_ROOT / "data" / "hf_cache"


settings = Settings()