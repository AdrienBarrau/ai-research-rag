"""CLI to query the RAG pipeline."""

from __future__ import annotations

import argparse
from rich import print as rprint
from rich.markdown import Markdown

from ai_rag.rag.pipeline import RAGPipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", type=str)
    args = p.parse_args()

    result = RAGPipeline().query(args.question)
    rprint(Markdown(result.answer))
    rprint("\n[bold]Sources:[/bold]")
    for i, d in enumerate(result.sources, 1):
        rprint(f"  [{i}] {d.metadata['title']} — {d.metadata['pdf_url']}")


if __name__ == "__main__":
    main()