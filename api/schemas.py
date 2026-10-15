"""Pydantic request/response models for the API.

Kept in one place so the schema is the single source of truth for validation,
the OpenAPI docs, and the Streamlit client.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000,
                          examples=["What causes edge-ring defects on wafers?"])
    k: Optional[int] = Field(None, ge=1, le=100,
                             description="Candidates to retrieve before reranking")
    top_n: Optional[int] = Field(None, ge=1, le=20,
                                 description="Passages to keep after reranking")


class SourceOut(BaseModel):
    source: str
    page: Optional[int] = None
    score: float
    chunk_id: Optional[int] = None
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceOut]


class UploadFileResult(BaseModel):
    filename: str
    chunks_added: int
    status: Optional[str] = None   # "indexed" | "unchanged" | "no-text"


class UploadResponse(BaseModel):
    files: List[UploadFileResult]
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    indexed: bool
    chunks: int
    hybrid: bool
    llm: bool


class MetricOut(BaseModel):
    name: str
    value: float
    explanation: str


class EvalResponse(BaseModel):
    metrics: List[MetricOut]
    n_questions: int
    llm_judge: bool
