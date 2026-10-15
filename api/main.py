"""FastAPI service exposing the RAG pipeline.

Endpoints:
    GET  /health        liveness + index/LLM status
    POST /ask           grounded answer with citations (non-streaming)
    POST /ask/stream    same, streamed token-by-token over SSE
    POST /upload        add PDF/DOCX/TXT/MD files to the live index
    POST /eval          run the evaluation harness and return metrics

Run locally:
    uvicorn api.main:app --reload
    open http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from config import settings
from src.pipeline import RAGPipeline, Source
from src.uploads import UploadRejected, safe_filename

from .schemas import (
    AskRequest,
    AskResponse,
    EvalResponse,
    HealthResponse,
    MetricOut,
    SourceOut,
    UploadResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("docmind.api")

# One source of truth for the limit (config.MAX_UPLOAD_MB) so the API, the
# Streamlit uploader and the docs can't drift apart (docs/SECURITY_AUDIT.md DM-08).
MAX_UPLOAD_BYTES = settings.max_upload_bytes
MAX_UPLOAD_FILES = 10          # per request
READ_CHUNK = 1024 * 1024       # 1 MB

pipeline: Optional[RAGPipeline] = None
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading RAG pipeline (models + index)...")
    pipeline = RAGPipeline()
    logger.info("Pipeline ready: %d chunks, hybrid=%s, llm=%s",
                len(pipeline.store.chunks), settings.hybrid, settings.has_llm)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="DocMind RAG Document Assistant",
    description="Hybrid retrieval -> cross-encoder rerank -> grounded, cited, streaming answers.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
# The service has no authentication, so a wildcard origin would let any page a
# visitor has open drive it from their browser. Origins come from
# API_CORS_ORIGINS and default to localhost (docs/SECURITY_AUDIT.md DM-10).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded, slow down."})


def _require_pipeline() -> RAGPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")
    return pipeline


def _sources_payload(sources: List[Source]) -> List[dict]:
    return [
        {"source": s.source, "page": s.page, "score": round(s.score, 4),
         "chunk_id": s.chunk_id, "text": s.text}
        for s in sources
    ]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    n = len(pipeline.store.chunks) if pipeline else 0
    return HealthResponse(
        status="ok",
        indexed=pipeline is not None,
        chunks=n,
        hybrid=settings.hybrid,
        llm=settings.has_llm,
    )


@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
def ask(request: Request, req: AskRequest) -> AskResponse:
    pipe = _require_pipeline()
    # Log shape, not content: questions can quote private documents (DM-11).
    logger.info("ask: %d chars (k=%s, top_n=%s)", len(req.question), req.k, req.top_n)
    result = pipe.answer(req.question, k=req.k, top_n=req.top_n)
    return AskResponse(
        question=result.query,
        answer=result.answer,
        sources=[SourceOut(**s) for s in _sources_payload(result.sources)],
    )


@app.post("/ask/stream")
@limiter.limit("30/minute")
def ask_stream(request: Request, req: AskRequest) -> EventSourceResponse:
    """Server-Sent Events stream.

    Emits one `sources` event first (so the UI paints citations immediately),
    then a `token` event per delta, then a terminal `done` event.
    """
    pipe = _require_pipeline()
    logger.info("ask/stream: %d chars", len(req.question))
    sources, tokens = pipe.answer_stream(req.question, k=req.k, top_n=req.top_n)

    def event_stream():
        yield {"event": "sources", "data": json.dumps(_sources_payload(sources))}
        try:
            for delta in tokens:
                yield {"event": "token", "data": json.dumps({"delta": delta})}
        except Exception as exc:  # surface generation errors to the client
            logger.exception("streaming failed")
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(event_stream())


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload(request: Request, files: List[UploadFile] = File(...)) -> UploadResponse:
    pipe = _require_pipeline()
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files in one request (max {MAX_UPLOAD_FILES}).")

    payloads: List[tuple] = []
    for f in files:
        try:
            name = safe_filename(f.filename or "")
        except UploadRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        # Read in bounded blocks and abort as soon as the limit is passed, so an
        # oversized upload is never fully buffered in memory (DM-09).
        buf = bytearray()
        while True:
            block = await f.read(READ_CHUNK)
            if not block:
                break
            buf.extend(block)
            if len(buf) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{name!r} exceeds the "
                           f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
        payloads.append((name, bytes(buf)))

    logger.info("upload: %d file(s), %d bytes", len(payloads),
                sum(len(d) for _, d in payloads))
    try:
        # Full server-side validation (type, size, magic bytes) happens inside
        # add_documents, which every upload path shares.
        #
        # persist=False: this service has no authentication, so an upload must
        # never be written into the shared on-disk index that every future
        # visitor of the Streamlit app loads. Uploads live for this process
        # only; build a persistent index with `python -m src.ingest` instead
        # (docs/SECURITY_AUDIT.md DM-17).
        summary = pipe.add_documents(payloads, persist=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return UploadResponse(**summary)


@app.post("/eval", response_model=EvalResponse)
@limiter.limit("6/minute")
def eval_endpoint(request: Request) -> EvalResponse:
    from eval.evaluate import run_eval
    pipe = _require_pipeline()
    logger.info("eval: running harness")
    report = run_eval(pipe)
    return EvalResponse(
        metrics=[MetricOut(**m) for m in report["metrics"]],
        n_questions=report["n_questions"],
        llm_judge=report["llm_judge"],
    )
