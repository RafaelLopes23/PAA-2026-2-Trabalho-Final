"""Schemas Pydantic da API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Pergunta do usuario em linguagem natural.")
    method: str = Field(
        default="hnsw",
        description="Metodo de busca: hnsw, sentence, word2vec ou cosine.",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedMovie(BaseModel):
    rank: int
    movie_id: int
    title: str
    score: float
    synopsis_preview: str


class QueryResponse(BaseModel):
    question: str
    method: str
    resolved_method: str
    top_k: int
    answer: str
    llm_backend: str
    results: list[RetrievedMovie]
    warnings: list[str]
    timings_ms: dict[str, float]
    available_methods: list[str]


class MethodDetails(BaseModel):
    description: str
    available: bool
    reason: str = ""
    resolved_method: str | None = None


class MethodsResponse(BaseModel):
    available_methods: list[str]
    methods: dict[str, MethodDetails]


class HealthResponse(BaseModel):
    status: str
    movies_count: int
    available_methods: list[str]
    methods: dict[str, MethodDetails]
