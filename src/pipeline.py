"""End-to-end RAG pipeline: retrieve -> rerank -> generate (with citations).

Loading the models is expensive, so the pipeline is built once and reused (the
FastAPI app keeps one instance alive; the Streamlit app caches one via
``st.cache_resource``). Models are **lazy-loaded** — the embedder, reranker and
LLM client are only constructed on first use — so importing/booting the app is
cheap and memory-light (important on Streamlit Community Cloud's ~1 GB). The
single embedder instance is reused everywhere (indexing, retrieval, and the
embedding-based eval metrics) so a model is never loaded twice.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from config import settings

from .chunking import Chunk
from .embeddings import Embedder
from .ingest import build_index, chunk_document, parse_bytes
from .llm import Generator
from .reranker import Reranker
from .retrieval import Retriever, build_retriever
from .vectorstore import FaissStore

logger = logging.getLogger("docmind.pipeline")


@dataclass
class Source:
    text: str
    source: str
    score: float
    page: Optional[int] = None
    chunk_id: Optional[int] = None


@dataclass
class RAGResult:
    query: str
    answer: str
    sources: List[Source]


class RAGPipeline:
    def __init__(self, index_dir: Optional[str] = None,
                 bootstrap_docs: str = "data/sample_docs"):
        self.index_dir = index_dir or settings.index_dir
        # One pipeline is shared across Streamlit sessions (st.cache_resource),
        # i.e. across threads. This guards the index against a concurrent
        # upload (writer) racing a query (reader).
        self._lock = threading.RLock()
        # Lazy singletons — see the module docstring. None until first use.
        self._embedder: Optional[Embedder] = None
        self._reranker: Optional[Reranker] = None
        self._retriever: Optional[Retriever] = None
        self._generator: Optional[Generator] = None
        self._generator_loaded = False

        # Self-heal: if no persisted index exists (e.g. a fresh Streamlit Cloud
        # container), build one from the bundled sample docs, reusing our own
        # embedder so the model loads exactly once.
        if not self._index_exists() and bootstrap_docs:
            logger.info("No index at %r — bootstrapping from %r", self.index_dir, bootstrap_docs)
            build_index(bootstrap_docs, self.index_dir, embedder=self.embedder)
        self.store = FaissStore.load(self.index_dir)

    def _index_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.index_dir, "index.faiss"))
                and os.path.exists(os.path.join(self.index_dir, "chunks.json")))

    # --- lazy models ----------------------------------------------------------
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            logger.info("Loading embedder: %s", settings.embed_model)
            self._embedder = Embedder(settings.embed_model)
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            logger.info("Loading reranker: %s", settings.reranker_model)
            self._reranker = Reranker(settings.reranker_model)
        return self._reranker

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = build_retriever(
                self.store, self.embedder,
                hybrid=settings.hybrid, rrf_k=settings.rrf_k,
            )
        return self._retriever

    @property
    def generator(self) -> Optional[Generator]:
        if not self._generator_loaded:
            self._generator_loaded = True
            if settings.has_llm:
                logger.info("Configuring LLM generator: %s", settings.llm_model)
                self._generator = Generator(
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    model=settings.llm_model,
                )
        return self._generator

    def add_documents(self, files: List[Tuple[str, bytes]]) -> dict:
        """Parse, chunk, embed and index uploaded files at runtime, then persist.

        Returns a small summary (per-file chunk counts + new total). Rebuilds the
        BM25 side of the hybrid retriever, which must see the full chunk set.
        """
        # Validate/parse everything up front so a bad file fails before we mutate
        # the shared index.
        parsed = [(filename, parse_bytes(filename, data)) for filename, data in files]
        with self._lock:
            added: List[dict] = []
            for filename, segments in parsed:
                chunks = chunk_document(filename, segments, self.embedder,
                                        start_id=len(self.store.chunks))
                if not chunks:
                    added.append({"filename": filename, "chunks_added": 0})
                    continue
                vectors = self.embedder.encode([c.text for c in chunks])
                self.store.add(vectors, chunks)
                added.append({"filename": filename, "chunks_added": len(chunks)})

            self._retriever = None           # force BM25 rebuild over the full set on next use
            self.store.save(self.index_dir)  # persist so the new docs survive restart
            return {"files": added, "total_chunks": len(self.store.chunks)}

    def retrieve(self, query: str, k: Optional[int] = None
                 ) -> List[Tuple[Chunk, float]]:
        with self._lock:
            return self.retriever.retrieve(query, k or settings.retrieve_k)

    def _rerank(self, query: str, k: Optional[int], top_n: Optional[int]
                ) -> Tuple[List[Tuple[Chunk, float]], List[Source]]:
        """Retrieve -> rerank, returning both the reranked pairs (for the LLM)
        and the Source objects (for the response / UI citations)."""
        candidates = self.retrieve(query, k)
        reranked = self.reranker.rerank(query, candidates,
                                        top_n or settings.rerank_top_n)
        sources = [
            Source(text=c.text, source=c.source, score=s, page=c.page,
                   chunk_id=c.chunk_id)
            for c, s in reranked
        ]
        return reranked, sources

    @staticmethod
    def _no_llm_answer(reranked: List[Tuple[Chunk, float]]) -> str:
        # No LLM key configured: return the top passage as a stand-in so the
        # retrieval half of the system is still demonstrable end-to-end.
        return ("[No LLM configured — showing top retrieved passage]\n\n"
                + (reranked[0][0].text if reranked else "No relevant context found."))

    def answer(self, query: str, k: Optional[int] = None,
               top_n: Optional[int] = None) -> RAGResult:
        reranked, sources = self._rerank(query, k, top_n)
        if self.generator is not None:
            answer_text = self.generator.answer(query, reranked)
        else:
            answer_text = self._no_llm_answer(reranked)
        return RAGResult(query=query, answer=answer_text, sources=sources)

    def answer_stream(self, query: str, k: Optional[int] = None,
                      top_n: Optional[int] = None
                      ) -> Tuple[List[Source], Iterator[str]]:
        """Streaming variant: returns the citations immediately (so the UI can
        render the sources panel right away) plus a generator that yields the
        grounded answer token-by-token."""
        reranked, sources = self._rerank(query, k, top_n)
        if self.generator is not None:
            token_iter = self.generator.stream_answer(query, reranked)
        else:
            def _fallback() -> Iterator[str]:
                yield self._no_llm_answer(reranked)
            token_iter = _fallback()
        return sources, token_iter
