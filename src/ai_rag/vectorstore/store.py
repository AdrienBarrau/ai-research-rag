"""Qdrant Cloud vector store wrapper avec support des paramètres HNSW."""

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


def get_vectorstore(embeddings: Embeddings) -> QdrantVectorStore:
    """Return a Qdrant vector store with optimized HNSW config."""
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if not collection_exists:
        log.info(" Création de la collection Qdrant: %s", collection_name)
        vector_size = 1024 if "bge-m3" in settings.embedding_model else 384
        
        # Configuration HNSW optimisée pour la vitesse
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                m=16,  # Nombre de connexions par noeud (16 est un bon défaut)
                ef_construct=128,  # Qualité de construction (128 est bon)
            ),
        )
    
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
    """Search with custom HNSW parameters for speed/quality tradeoff."""
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    
    #  Correction : embeddings (pluriel) et non embedding
    query_vector = store.embeddings.embed_query(query)
    
    # Search with custom ef parameter
    search_params = SearchParams(
        hnsw_ef=ef,
        exact=False,
    )
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k,
        search_params=search_params,
    )
    
    # Convert to LangChain Documents
    docs = []
    for result in results:
        doc = Document(
            page_content=result.payload.get("page_content", ""),
            metadata=result.payload.get("metadata", {}),
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
    log.info(" Upserted %d chunks into Qdrant collection '%s'", len(docs), settings.qdrant_collection_name)
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