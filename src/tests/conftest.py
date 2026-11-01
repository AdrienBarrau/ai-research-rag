from pathlib import Path
import pytest


@pytest.fixture
def tmp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("PDF_CACHE_DIR", str(tmp_path / "pdfs"))
    monkeypatch.setenv("HF_CACHE_DIR", str(tmp_path / "hf_cache"))
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # petit pour tests
    yield tmp_path