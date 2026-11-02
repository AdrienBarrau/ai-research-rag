"""Qdrant Cloud vector store wrapper - compatible multi-versions."""

from __future__ import annotations

import logging
import uuid

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, SearchParams

from ai_rag.config import settings

log = logging.getLogger(__name__)

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


def _get_embeddings_attribute(store: QdrantVectorStore) -> Embeddings:
    """Get the embeddings object, compatible with all langchain-qdrant versions."""
    # Essaie "embedding" (singulier) puis "embeddings" (pluriel)
    if hasattr(store, 'embedding'):
        return store.embedding
    elif hasattr(store, 'embeddings'):
        return store.embeddings
    else:
        raise AttributeError(
            "QdrantVectorStore has neither 'embedding' nor 'embeddings' attribute. "
            "Please check your langchain-qdrant version."
        )


def get_vectorstore(embeddings: Embeddings) -> QdrantVectorStore:
    """Return a Qdrant vector store with optimized HNSW config."""
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if not collection_exists:
        log.info("📦 Création de la collection Qdrant: %s", collection_name)
        vector_size = 1024 if "bge-m3" in settings.embedding_model else 384
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=128,
            ),
        )
    
    # ✅ Détection dynamique du nom de paramètre
    import inspect
    sig = inspect.signature(QdrantVectorStore.__init__)
    params = sig.parameters
    
    # Si "embeddings" (pluriel) est dans les paramètres, utilise-le
    if 'embeddings' in params:
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )
    # Sinon utilise "embedding" (singulier)
    else:
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )


def search_with_preset(
    store: QdrantVectorStore,
    query: str,
    k: int,
    ef: int,
) -> list[Document]:
    """Search with custom HNSW parameters using the new query_points API."""
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    
    # ✅ Détection dynamique de l'attribut
    embedder = _get_embeddings_attribute(store)
    query_vector = embedder.embed_query(query)
    
    search_params = SearchParams(
        hnsw_ef=ef,
        exact=False,
    )
    
    # Utilisation de query_points (API moderne)
    query_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
        search_params=search_params,
    )
    
    # La structure de retour est query_result.points
    docs = []
    for point in query_result.points:
        payload = point.payload or {}
        doc = Document(
            page_content=payload.get("page_content", ""),
            metadata=payload.get("metadata", {}),
        )
        docs.append(doc)
    
    return docs


def _generate_chunk_id(doc: Document, index: int) -> str:
    """Generate a deterministic UUID for each chunk."""
    unique_string = (
        f"{doc.metadata.get('arxiv_id', 'unknown')}|"
        f"p{doc.metadata.get('page', 0)}|"
        f"idx{index}|"
        f"{doc.page_content[:100]}"
    )
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