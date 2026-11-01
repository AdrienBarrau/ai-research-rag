"""Pydantic schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    k: int = Field(default=8, ge=1, le=50)


class SourceOut(BaseModel):
    arxiv_id: str
    title: str
    page: int
    pdf_url: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class IngestRequest(BaseModel):
    query: str | None = None
    ids: list[str] | None = None
    categories: list[str] | None = None
    max_results: int = Field(default=20, ge=1, le=200)


class IngestResponse(BaseModel):
    papers_fetched: int
    chunks_indexed: int