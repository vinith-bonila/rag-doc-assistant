"""Central configuration.

Everything tunable lives here so the pipeline, API and eval scripts all read
the same settings. Values can be overridden with environment variables (see
.env.example), which is what the Dockerfile relies on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # loads a local .env if present; no-op in production


@dataclass(frozen=True)
class Settings:
    # --- Models -----------------------------------------------------------
    # Small, fast, strong general-purpose embedder. 384-dim.
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Cross-encoder used to re-score the top candidates. This is the step most
    # tutorial RAG projects skip — it's what makes retrieval actually good.
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- Retrieval params -------------------------------------------------
    # How many chunks each retriever returns before fusion/reranking.
    retrieve_k: int = int(os.getenv("RETRIEVE_K", "20"))
    # How many chunks survive reranking and get sent to the LLM.
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    # Hybrid search: fuse dense (FAISS) + sparse (BM25). Set HYBRID=false to
    # fall back to dense-only retrieval (useful for ablation in the eval tab).
    hybrid: bool = os.getenv("HYBRID", "true").lower() in ("1", "true", "yes")
    # Reciprocal-rank-fusion constant. Larger => flatter weighting of rank.
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    # "semantic" (embedding-based breakpoints) or "recursive" (robust default).
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")
    # Extract tables from PDFs and serialise them to Markdown so numbers stay
    # attached to their row/column labels (rag.pdf p.6). Set PARSE_TABLES=false
    # to skip the (slower) table pass.
    parse_tables: bool = os.getenv("PARSE_TABLES", "true").lower() in ("1", "true", "yes")

    # --- Context construction (rag.pdf p.12) ------------------------------
    # Drop near-duplicate reranked chunks whose token-set Jaccard overlap is at
    # or above this before building the LLM context.
    dedup_jaccard: float = float(os.getenv("DEDUP_JACCARD", "0.88"))
    # Approximate token budget for the retrieved context sent to the model.
    context_token_budget: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "3000"))

    # --- Validation (rag.pdf p.14) ----------------------------------------
    # Verify after generation that every [n] citation maps to a real source.
    validate_citations: bool = os.getenv("VALIDATE_CITATIONS", "true").lower() in ("1", "true", "yes")

    # --- Limits -----------------------------------------------------------
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))

    # --- LLM (OpenAI-compatible) -----------------------------------------
    # Works with OpenAI, or any OpenAI-compatible endpoint. To use Groq's free
    # tier instead, set LLM_BASE_URL=https://api.groq.com/openai/v1 and
    # LLM_MODEL=openai/gpt-oss-20b with a Groq key. No key => retrieval-only.
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # --- Storage ----------------------------------------------------------
    index_dir: str = os.getenv("INDEX_DIR", "storage")

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_api_key)


settings = Settings()
