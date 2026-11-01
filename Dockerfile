FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour pymupdf + build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Installer les deps Python
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Pré-télécharger le modèle d'embeddings au build (évite le download au runtime)
ENV HF_HOME=/app/data/hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "src.ai_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]