"""Streamlit UI - Version robuste et moderne."""

import sys
import os
import streamlit as st

# Astuce pour Streamlit Cloud : ajouter le dossier 'src' au chemin de recherche Python
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ai_rag.rag.pipeline import EmptyDatabaseError, RAGPipeline

st.set_page_config(page_title="AI Research RAG", page_icon="🧠", layout="wide")
st.title("🧠 AI Research RAG")
st.caption("Pose des questions pointues sur la recherche IA (arXiv)")

# Initialisation du pipeline avec gestion d'erreur large
@st.cache_resource
def get_pipeline():
    try:
        return RAGPipeline()
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation du pipeline : {e}")
        st.info("💡 Vérifie que tes clés API (GROQ_API_KEY, QDRANT_API_KEY) sont bien configurées dans les **Secrets** de l'application (menu ⋮ en haut à droite → Settings → Secrets).")
        return None

with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Afficher l'état de la base de données
    try:
        from ai_rag.vectorstore.store import get_collection_count, get_vectorstore
        from ai_rag.embeddings.embedder import get_embedder
        
        pipeline = get_pipeline()
        if pipeline and pipeline.store:
            count = get_collection_count(pipeline.store)
            st.metric("Chunks en base locale", count)
            if count == 0:
                st.warning("⚠️ Base vide ! Le fallback live arXiv s'activera.")
    except Exception as e:
        st.warning(f"Impossible de lire la base de données : {e}")
    
    if st.button("🔄 Vider le cache et recharger"):
        st.cache_resource.clear()
        st.rerun()

# Zone de chat (toujours affichée, même en cas d'erreur)
st.markdown("---")
question = st.chat_input("Pose ta question sur l'IA (ex: 'Quel est le SOTA sur ImageNet ?')")

if question:
    st.markdown(f"### 🧑‍💻 **Toi :** {question}")
    
    pipeline = get_pipeline()
    if pipeline is None:
        st.error("Impossible de répondre car le pipeline n'a pas pu être initialisé. Vérifie les logs et les secrets.")
    else:
        with st.spinner("🔍 Recherche dans la base + génération de la réponse..."):
            try:
                result = pipeline.query(question)
                
                # Indicateur de source
                if result.retrieval_source == "local":
                    st.success(f"✅ Réponse depuis la base locale (confiance: {result.confidence_score:.2f})")
                elif result.retrieval_source == "hybrid":
                    st.warning(f"🔄 Réponse combinée local + live arXiv (confiance locale: {result.confidence_score:.2f})")
                else:
                    st.info(f"ℹ️ Source: {result.retrieval_source}")

                st.markdown("### 🤖 **Réponse :**")
                st.markdown(result.answer)

                with st.expander(f"📚 {len(result.sources)} sources utilisées", expanded=False):
                    for i, d in enumerate(result.sources, 1):
                        source_type = d.metadata.get("source_type", "local")
                        badge = "🔴 LIVE" if source_type == "live" else "🟢 LOCAL"
                        st.markdown(
                            f"**[{i}] {badge} {d.metadata.get('title', 'Titre inconnu')}** "
                            f"(`{d.metadata.get('arxiv_id', '?')}`, p.{d.metadata.get('page', '?')})  \n"
                            f"[🔗 Voir le PDF]({d.metadata.get('pdf_url', '#')})  \n"
                            f"> *{d.page_content[:300]}...*"
                        )
            except EmptyDatabaseError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"❌ Une erreur inattendue est survenue : {e}")
                st.info("💡 Astuce : Clique sur les 3 petits points (⋮) en haut à droite → **Manage app** → **Logs** pour voir le détail de l'erreur.")