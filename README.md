*# AI Research RAG

Application RAG pour indexer les papiers de recherche en IA et répondre à des questions pointues (SOTA, approches techniques, etc.) avec des sources vérifiables.

**[ Essayer l'app en ligne](https://ai-research-rag-kfjbnxzq4m3igszngx4apx.streamlit.app/)**  


---

## Features

- **Base de connaissances** : Indexation des papiers les plus cités par domaine (Transformers, Diffusion, SSL, LLMs, etc.).
- **Retrieval Hybride** : Interrogation prioritaire de la base vectorielle (Qdrant). Si le score de similarité est trop faible, fallback automatique en temps réel sur l'API arXiv pour aller chercher des papiers récents.
- **Presets de qualité** : L'utilisateur choisit entre Fast, Balanced ou Precise. Ça ajuste dynamiquement les paramètres de recherche ANN (HNSW) et le modèle LLM pour contrôler le trade-off vitesse/précision.
- **100% Gratuit en prod** : 
  - Embeddings via Hugging Face Inference API (évite le throttling CPU sur Streamlit Cloud).
  - LLM via Groq API (Qwen-2.5-32b / Llama-3.1-8b-instant).
  - Vector Store sur Qdrant Cloud (Free tier 1GB).
- **UI Streamlit** propre avec affichage des sources et des liens PDF.

---

## Stack Technique

- **Langage** : Python 3.11+ (typage strict, `pydantic-settings`)
- **Orchestration** : LangChain
- **Vector Store** : Qdrant Cloud (recherche ANN avec paramètres HNSW dynamiques)
- **Embeddings** : `BAAI/bge-m3` (via Hugging Face Inference API)
- **LLM** : Groq API (`qwen/qwen2.5-32b`, `llama-3.1-8b-instant`)
- **UI** : Streamlit
- **Parsing** : PyMuPDF (`pymupdf`)
- **Déploiement** : Streamlit Community Cloud

---

## Quickstart (Local)

### 1. Installation

```bash
git clone https://github.com/AdrienBarrau/ai-research-rag.git
cd ai-research-rag

python -m venv .venv
# Windows PowerShell :
.\.venv\Scripts\Activate.ps1
# Linux/Mac :
source .venv/bin/activate

pip install -r requirements.txt*