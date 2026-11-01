.PHONY: install install-dev lint format test ingest api ui docker-up docker-down

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest tests -v

ingest:
	@read -p "Query: " q; python scripts/ingest.py --query "$$q" --max 20

api:
	uvicorn src.ai_rag.api.main:app --reload --port 8000

ui:
	streamlit run src/ai_rag/ui/app.py --server.port 8501

docker-up:
	docker compose up --build

docker-down:
	docker compose down