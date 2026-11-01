"""Qdrant Cloud vector store wrapper."""

from __future__ import annotations

import logging
import uuid

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ai_rag.config import settings

log = logging.getLogger(__name__)

# Namespace fixe pour générer des UUID déterministes à partir de chaînes
UUID_NAMESPACE = uuid.NAMESPACE_DNS


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant Cloud client."""
    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY must be set in .env. "
            "Create a free cluster at https://cloud.qdrant.io"
        )
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )


def get_vectorstore(embeddings: Embeddings) -> QdrantVectorStore:
    """Return a Qdrant vector store.
    
    Crée la collection si elle n'existe pas encore.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    
    # Vérifie si la collection existe, sinon la crée
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if not collection_exists:
        log.info("📦 Création de la collection Qdrant: %s", collection_name)
        vector_size = 1024 if "bge-m3" in settings.embedding_model else 384
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def _generate_chunk_id(doc: Document, index: int) -> str:
    """Generate a deterministic UUID for each chunk based on its content and metadata."""
    # On crée une chaîne unique pour ce chunk spécifique
    unique_string = (
        f"{doc.metadata.get('arxiv_id', 'unknown')}|"
        f"p{doc.metadata.get('page', 0)}|"
        f"idx{index}|"
        f"{doc.page_content[:100]}"  # On inclut le début du contenu pour éviter les collisions
    )
    # uuid5 génère toujours le même UUID pour la même chaîne d'entrée
    return str(uuid.uuid5(UUID_NAMESPACE, unique_string))


def add_documents(store: QdrantVectorStore, docs: list[Document]) -> list[str]:
    """Add documents with unique UUIDs."""
    ids = [_generate_chunk_id(doc, i) for i, doc in enumerate(docs)]
    store.add_documents(docs, ids=ids)
    log.info("✅ Upserted %d chunks into Qdrant collection '%s'", len(docs), settings.qdrant_collection_name)
    return ids


def get_collection_count(store: QdrantVectorStore) -> int:
    """Return the number of documents in the collection."""
    try:
        client = get_qdrant_client()
        collection_info = client.get_collection(settings.qdrant_collection_name)
        return collection_info.points_count
    except Exception as e:
        log.warning("Failed to get collection count: %s", e)
        return 0