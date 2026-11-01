#  AI Research RAG

Application RAG pour indexer les papiers de recherche IA (arXiv) et répondre à des questions pointues : SOTA sur un dataset, approches pour un type de données, etc.

## ✨ Features

- Ingestion automatique depuis arXiv (par query, catégorie, ou arXiv ID)
-  Parsing PDF scientifique avec préservation des sections/équations
-  Embeddings locaux `BAAI/bge-m3` (gratuit, multilingue)
-  RAG avec Groq Llama 3.1 70B (gratuit, 14 400 req/jour)
-  ChromaDB persistant
-  API REST (FastAPI) + UI (Streamlit)
-  Docker-ready, déployable en 1-click
-  **100% gratuit** (pas de clé OpenAI nécessaire)

##  Quickstart

### 1. Installation

```bash
# Cloner le repo
git clone <repo> && cd ai-research-rag

# Copier et éditer la config
cp .env.example .env
# Ajouter ta GROQ_API_KEY dans .env

# Créer un environnement virtuel
python -m venv .venv

# Activer le venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Installer le projet
pip install -e .
```

### 2. Ingérer des papiers

```bash
# Exemple : 20 papiers sur les diffusion models
python scripts/ingest.py --query "diffusion models image generation" --max 20

# Par catégories arXiv
python scripts/ingest.py --categories cs.CV cs.LG --max 30

# Par IDs spécifiques
python scripts/ingest.py --ids 2312.00752 2401.12345
```

**Premier run** : télécharge le modèle d'embeddings (~2.2 GB, une seule fois).

### 3. Poser des questions

**En ligne de commande** :

```bash
python scripts/query.py "Quel est le SOTA sur ImageNet en top-1 accuracy ?"
```

**Via l'UI Streamlit** :

```bash
streamlit run src/ai_rag/ui/app.py --server.port 8501
```

→ Ouvre http://localhost:8501

**Via l'API REST** :

```bash
uvicorn src.ai_rag.api.main:app --reload --port 8000
```

→ Ouvre http://localhost:8000/docs pour tester l'API

##  Exemples de questions

- *"Quel est le SOTA sur ImageNet en top-1 accuracy en 2025 ?"*
- *"Quelles approches existent pour la segmentation médicale avec peu de données annotées ?"*
- *"Compare les méthodes de RLHF vs DPO pour l'alignement des LLMs"*
- *"Quels sont les meilleurs modèles pour la génération vidéo en 2025 ?"*

##  Architecture

```
[arXiv] → [PDF Parser] → [Chunker] → [Embedder bge-m3] → [ChromaDB]
                                                       ↓
[User Query] → [Retriever] → [LLM Llama 3.1 70B] → [Answer + Citations]
```

### Structure du code

```
src/ai_rag/
├── config.py              # Configuration centralisée (pydantic-settings)
├── ingestion/
│   ├── arxiv_fetcher.py   # Fetch papers depuis arXiv
│   └── pdf_parser.py      # Parse PDFs en Documents LangChain
├── embeddings/
│   └── embedder.py        # Modèle d'embeddings local (bge-m3)
├── vectorstore/
│   └── store.py           # ChromaDB wrapper
├── rag/
│   ├── retriever.py       # Retrieval + formatage contexte
│   ├── generator.py       # LLM Groq + prompts
│   └── pipeline.py        # Pipeline RAG end-to-end
├── api/
│   ├── main.py            # FastAPI app
│   └── schemas.py         # Pydantic schemas
└── ui/
    └── app.py             # Streamlit UI
```

##  Déploiement

### API sur Railway / Fly.io / Render

```bash
# Railway
railway up

# Fly.io
fly launch
fly deploy

# Render : connecter le repo GitHub, détecte le Dockerfile
```

### UI Streamlit sur Streamlit Cloud

1. Push le repo sur GitHub
2. Va sur [share.streamlit.io](https://share.streamlit.io)
3. Connecte le repo, set `main file: src/ai_rag/ui/app.py`
4. Ajoute `GROQ_API_KEY` dans les Secrets

### Docker

```bash
# Build et run
docker compose up --build

# Ou directement
docker build -t ai-rag .
docker run -p 8000:8000 --env-file .env ai-rag
```

##  Stack

- **Python** 3.11+
- **LLM** : Groq (Llama 3.1 70B) — gratuit
- **Embeddings** : `BAAI/bge-m3` (local) — gratuit
- **Vector DB** : ChromaDB
- **Orchestration** : LangChain LCEL
- **API** : FastAPI
- **UI** : Streamlit
- **Parsing** : PyMuPDF
- **Container** : Docker

##  Tests

```bash
# Installer les deps dev
pip install -e ".[dev]"

# Lancer les tests
pytest tests -v

# Lint
ruff check src tests
mypy src

# Format
ruff format src tests
ruff check --fix src tests
```

##  Configuration

Édite `.env` :

```bash
# LLM (Groq - gratuit)
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.1-70b-versatile

# Embeddings (local, gratuit)
EMBEDDING_MODEL=BAAI/bge-m3

# Vector store
CHROMA_PERSIST_DIR=./data/chroma
COLLECTION_NAME=ai_papers

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_K=8
```

##  Tips

### Accélérer les embeddings (si GPU disponible)

Dans `src/ai_rag/embeddings/embedder.py`, change :

```python
model_kwargs={"device": "cpu"}  # → "cuda"
```

### Changer de modèle d'embeddings

Dans `.env` :

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Plus rapide, moins bon
EMBEDDING_MODEL=BAAI/bge-m3  # Meilleur (défaut)
```

### Traçabilité avec LangSmith

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_...
LANGCHAIN_PROJECT=ai-research-rag
```

##  Pour briller en entretien

1. **Traçabilité** : active LangSmith pour montrer les traces en live
2. **Évaluation** : ajoute un dataset de questions gold + métriques (faithfulness, relevancy)
3. **Reranking** : ajoute un reranker (Cohere, CrossEncoder) après retrieval
4. **Hybrid search** : combine similarity + full-text search dans Chroma
5. **Observability** : métriques Prometheus sur `/query` (latence, tokens, nb sources)

##  License

MIT