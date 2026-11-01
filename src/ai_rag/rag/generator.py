"""LLM generator with structured output."""

from __future__ import annotations

from langchain_groq import ChatGroq

from ai_rag.config import settings


def get_llm(temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
    )


SYSTEM_PROMPT = """Tu es un expert en recherche IA. Réponds à la question en te basant UNIQUEMENT sur les extraits de papiers fournis.

Règles:
- Cite chaque affirmation avec [n] correspondant à la source.
- Si l'info n'est pas dans le contexte, dis-le clairement.
- Pour les SOTA, précise le papier, la métrique, la valeur et la date.
- Sois concis, technique, et structuré (listes, tableaux si pertinent).
- Termine par une liste "📚 Références" avec les arXiv IDs cités.
"""


def build_prompt(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"## Contexte (extraits de papiers)\n{context}\n\n## Question\n{question}",
        },
    ]