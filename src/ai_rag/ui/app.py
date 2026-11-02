"""Streamlit UI avec sélecteur de qualité."""

import sys
import os
import streamlit as st

# Astuce pour Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ai_rag.rag.pipeline import EmptyDatabaseError, RAGPipeline
from ai_rag.rag.presets import PRESETS

st.set_page_config(page_title="AI Research RAG", page_icon="", layout="wide")
st.title(" AI Research RAG")
st.caption("Pose des questions pointues sur la recherche IA (arXiv)")

@st.cache_resource
def get_pipeline():
    try:
        return RAGPipeline()
    except Exception as e:
        st.error(f" Erreur d'initialisation : {e}")
        st.info(" Vérifie les Secrets (⋮ → Settings → Secrets)")
        return None

with st.sidebar:
    st.header(" Paramètres")
    
    # Sélecteur de preset
    preset_options = {k: v.description for k, v in PRESETS.items()}
    selected_preset = st.selectbox(
        " Qualité de réponse",
        options=list(preset_options.keys()),
        format_func=lambda x: preset_options[x],
        index=1,  # "balanced" par défaut
    )
    
    # Afficher l'état de la base
    try:
        from ai_rag.vectorstore.store import get_collection_count, get_vectorstore
        from ai_rag.embeddings.embedder import get_embedder
        
        pipeline = get_pipeline()
        if pipeline and pipeline.store:
            count = get_collection_count(pipeline.store)
            st.metric("Chunks en base", count)
    except Exception as e:
        st.warning(f"Impossible de lire la base : {e}")
    
    if st.button(" Vider le cache"):
        st.cache_resource.clear()
        st.rerun()

st.markdown("---")
question = st.chat_input("Pose ta question sur l'IA...")

if question:
    st.markdown(f"### 🧑 **Toi :** {question}")
    
    pipeline = get_pipeline()
    if pipeline is None:
        st.error("Pipeline non initialisé. Vérifie les Secrets.")
    else:
        with st.spinner(f" Recherche (preset: {selected_preset})..."):
            try:
                result = pipeline.query(question, preset_name=selected_preset)
                
                st.success(f" Réponse générée (preset: {result.preset_used})")
                st.markdown("###  **Réponse :**")
                st.markdown(result.answer)

                with st.expander(f" {len(result.sources)} sources", expanded=False):
                    for i, d in enumerate(result.sources, 1):
                        st.markdown(
                            f"**[{i}] {d.metadata.get('title', '?')}** "
                            f"(`{d.metadata.get('arxiv_id', '?')}`)  \n"
                            f"[🔗 PDF]({d.metadata.get('pdf_url', '#')})  \n"
                            f"> *{d.page_content[:200]}...*"
                        )
            except EmptyDatabaseError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f" Erreur : {e}")
                st.info(" Vérifie les Logs (⋮ → Manage app → Logs)")