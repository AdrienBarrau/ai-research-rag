"""Test complet de la connexion Qdrant + embeddings + search."""

import sys
import os

# Ajouter src au path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ai_rag.config import settings
from ai_rag.embeddings.embedder import get_embedder
from ai_rag.vectorstore.store import (
    get_qdrant_client,
    get_vectorstore,
    get_collection_count,
    search_with_preset,
)

print("=" * 60)
print("🧪 TEST COMPLET DU PIPELINE RAG")
print("=" * 60)

# Test 1 : Connexion Qdrant
print("\n[1/5] 🔌 Test connexion Qdrant...")
try:
    client = get_qdrant_client()
    collections = client.get_collections().collections
    print(f"✅ Connexion OK. Collections: {[c.name for c in collections]}")
except Exception as e:
    print(f"❌ Erreur Qdrant: {e}")
    sys.exit(1)

# Test 2 : Embeddings
print("\n[2/5] 🧠 Test embeddings...")
try:
    embedder = get_embedder()
    test_vector = embedder.embed_query("test query")
    print(f"✅ Embeddings OK. Dimension: {len(test_vector)}")
except Exception as e:
    print(f"❌ Erreur embeddings: {e}")
    sys.exit(1)

# Test 3 : Vector store
print("\n[3/5] 📦 Test vector store...")
try:
    store = get_vectorstore(embedder)
    count = get_collection_count(store)
    print(f"✅ Vector store OK. Chunks: {count}")
except Exception as e:
    print(f"❌ Erreur vector store: {e}")
    sys.exit(1)

# Test 4 : Search avec preset
print("\n[4/5] 🔍 Test search avec preset (ef=128, k=3)...")
try:
    docs = search_with_preset(store, "transformer attention mechanism", k=3, ef=128)
    print(f"✅ Search OK. {len(docs)} documents trouvés")
    if docs:
        print(f"   Premier doc: {docs[0].metadata.get('title', 'N/A')[:60]}")
except Exception as e:
    print(f"❌ Erreur search: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5 : Pipeline complet
print("\n[5/5] 🚀 Test pipeline RAG complet...")
try:
    from ai_rag.rag.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    result = pipeline.query("Qu'est-ce que l'attention dans les transformers ?", preset_name="fast")
    print(f"✅ Pipeline OK.")
    print(f"   Sources: {len(result.sources)}")
    print(f"   Réponse (100 premiers chars): {result.answer[:100]}...")
except Exception as e:
    print(f"❌ Erreur pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ TOUS LES TESTS PASSENT ! Le pipeline est opérationnel.")
print("=" * 60)