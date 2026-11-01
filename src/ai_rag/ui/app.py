"""Streamlit UI avec indicateur de source (local/live)."""

from __future__ import annotations

import streamlit as st

from ai_rag.rag.pipeline import EmptyDatabaseError, RAGPipeline


st.set_page_config(page_title="AI Research RAG", page_icon="🧠", layout="wide")
st.title("🧠 AI Research RAG")
st.caption("RAG hybride : base locale + fallback live arXiv")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


with st.sidebar:
    st.header("⚙️ Paramètres")
    k = st.slider("Nombre de sources", 1, 20, 8)
    
    try:
        from ai_rag.vectorstore.store import get_collection_count, get_vectorstore
        from ai_rag.embeddings.embedder import get_embedder
        count = get_collection_count(get_vectorstore(get_embedder()))
        st.metric("Chunks en base locale", count)
    except Exception:
        pass
    
    if st.button("🔄 Recharger le pipeline"):
        get_pipeline.clear()

question = st.text_input(
    "Ta question",
    placeholder="Quel est le SOTA sur ImageNet en top-1 accuracy ?",
)

if question:
    try:
        with st.spinner("Retrieval + génération…"):
            result = get_pipeline().query(question)

        # Indicateur de source
        if result.retrieval_source == "local":
            st.success(f"✅ Réponse depuis la base locale (confiance: {result.confidence_score:.2f})")
        elif result.retrieval_source == "hybrid":
            st.warning(f"🔄 Réponse combinée local + live arXiv (confiance locale: {result.confidence_score:.2f})")
        else:
            st.info(f"ℹ️ Source: {result.retrieval_source}")

        st.markdown("### 💬 Réponse")
        st.markdown(result.answer)

        with st.expander(f"📚 {len(result.sources)} sources utilisées"):
            for i, d in enumerate(result.sources, 1):
                source_type = d.metadata.get("source_type", "local")
                badge = "🔴 LIVE" if source_type == "live" else "🟢 LOCAL"
                st.markdown(
                    f"**[{i}] {badge} {d.metadata['title']}** "
                    f"(`{d.metadata['arxiv_id']}`, p.{d.metadata.get('page', '?')})  \n"
                    f"[PDF]({d.metadata['pdf_url']})  \n"
                    f"> {d.page_content[:300]}…"
                )
    except EmptyDatabaseError as e:
        st.error(str(e))