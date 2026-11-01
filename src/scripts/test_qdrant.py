"""Test la connexion à Qdrant Cloud."""

from ai_rag.config import settings
from ai_rag.embeddings.embedder import get_embedder
from ai_rag.vectorstore.store import get_vectorstore, get_collection_count

print(" Test de connexion à Qdrant Cloud...")
print(f" URL: {settings.qdrant_url}")

try:
    embedder = get_embedder()
    store = get_vectorstore(embedder)
    count = get_collection_count(store)
    print(f" Connexion réussie ! Chunks dans la collection: {count}")
except Exception as e:
    print(f" Erreur: {e}")
    print(" Vérifie que QDRANT_URL et QDRANT_API_KEY sont bien dans ton .env")