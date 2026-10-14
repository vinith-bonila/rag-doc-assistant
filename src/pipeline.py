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

import contextlib
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Set, Tuple

import numpy as np

from config import settings

from .chunking import Chunk
from .context import build_context
from .embeddings import Embedder
from .ingest import build_index, chunk_document, parse_bytes
from .llm import Generator
from .metadata import MetaFilter, content_hash, stamp_metadata
from .reranker import Reranker
from .retrieval import Retriever, build_retriever
from .trace import RAGTrace
from .validation import flag_invalid_citations, validate_citations
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
    validation: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)


def _build_filter(allowed_sources: Optional[Set[str]],
                  doc_types: Optional[Set[str]] = None) -> Optional[MetaFilter]:
    if not allowed_sources and not doc_types:
        return None
    return MetaFilter(sources=set(allowed_sources) if allowed_sources else None,
                      doc_types=set(doc_types) if doc_types else None)


def _span(trace: Optional[RAGTrace], name: str):
    return trace.span(name) if trace is not None else contextlib.nullcontext()


# The exact phrase the assistant prompt uses to refuse when the document lacks
# the answer. We hard-stop the stream after it so the model can't tack on a
# general-knowledge explanation the user didn't ask for.
_REFUSAL_MARK = "does not contain enough information"


def _stop_after_refusal(tokens: Iterator[str]) -> Iterator[str]:
    buf = ""
    tripped = False
    for delta in tokens:
        yield delta
        buf += delta
        if not tripped and _REFUSAL_MARK in buf.lower():
            tripped = True
        if tripped and "\n" in delta:      # end of the refusal line -> stop
            return


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

        Re-uploading a file (same name) **replaces** its previous chunks rather
        than appending duplicates: the old vectors are dropped and the index is
        rebuilt, so repeated uploads don't grow memory without bound. Returns a
        summary (per-file chunk counts + new total).
        """
        # Validate/parse everything up front so a bad file fails before we mutate
        # the shared index.
        parsed = [(filename, data, parse_bytes(filename, data)) for filename, data in files]
        with self._lock:
            # content-hash of each already-indexed source (rag.pdf p.5/19: skip
            # re-embedding an identical re-upload).
            existing_hash = {c.source: (c.meta or {}).get("content_hash")
                             for c in self.store.chunks}
            added: List[dict] = []
            new_chunks: List[Chunk] = []
            for filename, data, segments in parsed:
                if existing_hash.get(filename) == content_hash(data) \
                        and filename in existing_hash:
                    added.append({"filename": filename, "chunks_added": 0,
                                  "status": "unchanged"})
                    continue
                file_chunks = chunk_document(filename, segments, self.embedder)
                stamp_metadata(file_chunks, filename, data, settings.embed_model)
                new_chunks.extend(file_chunks)
                added.append({"filename": filename, "chunks_added": len(file_chunks),
                              "status": "indexed"})

            if not new_chunks:   # everything was unchanged -> nothing to rebuild
                return {"files": added, "total_chunks": len(self.store.chunks)}

            # Sources present in this batch replace any prior version of the same
            # file. Keep the rest and reuse their stored vectors (no re-embedding).
            replaced = {c.source for c in new_chunks}
            old_vectors = self.store.reconstruct_all()
            keep = [i for i, c in enumerate(self.store.chunks) if c.source not in replaced]
            kept_chunks = [self.store.chunks[i] for i in keep]
            kept_vectors = old_vectors[keep] if keep else np.zeros(
                (0, self.embedder.dim), dtype="float32")

            if new_chunks:
                new_vectors = self.embedder.encode([c.text for c in new_chunks])
                all_vectors = np.vstack([kept_vectors, new_vectors])
            else:
                all_vectors = kept_vectors
            all_chunks = kept_chunks + new_chunks

            # Reassign contiguous ids so they stay unique after removals.
            for new_id, chunk in enumerate(all_chunks):
                chunk.chunk_id = new_id

            self.store.rebuild(all_vectors, all_chunks)  # drops the old index/vectors
            self._retriever = None            # rebuild BM25 over the new chunk set on next use
            self.store.save(self.index_dir)   # persist so the new docs survive restart
            return {"files": added, "total_chunks": len(self.store.chunks)}

    def retrieve(self, query: str, k: Optional[int] = None,
                 allowed_sources: Optional[set] = None,
                 doc_types: Optional[set] = None,
                 filt: Optional[MetaFilter] = None
                 ) -> List[Tuple[Chunk, float]]:
        with self._lock:
            f = filt if filt is not None else _build_filter(allowed_sources, doc_types)
            return self.retriever.retrieve(query, k or settings.retrieve_k, filt=f)

    def _rerank(self, query: str, k: Optional[int], top_n: Optional[int],
                allowed_sources: Optional[set] = None,
                doc_types: Optional[set] = None,
                trace: Optional[RAGTrace] = None
                ) -> Tuple[List[Tuple[Chunk, float]], List[Source]]:
        """Retrieve -> rerank -> build context, returning the context pairs (for
        the LLM) and matching Source objects (for citations). Context construction
        deduplicates near-identical passages and trims to a token budget, so the
        [n] markers the model emits align 1:1 with the shown sources."""
        f = _build_filter(allowed_sources, doc_types)
        with _span(trace, "retrieve"):
            candidates = self.retrieve(query, k, filt=f)
        if trace is not None:
            trace.set("candidates_retrieved", len(candidates))
        with _span(trace, "rerank"):
            reranked = self.reranker.rerank(query, candidates,
                                            top_n or settings.rerank_top_n)
        with _span(trace, "context"):
            context_pairs, ctx_report = build_context(
                reranked, settings.context_token_budget, settings.dedup_jaccard)
        if trace is not None:
            trace.set("context", ctx_report)
        sources = [
            Source(text=c.text, source=c.source, score=s, page=c.page,
                   chunk_id=c.chunk_id)
            for c, s in context_pairs
        ]
        return context_pairs, sources

    @staticmethod
    def _no_llm_answer(context_pairs: List[Tuple[Chunk, float]]) -> str:
        # No LLM key configured: return the top passage as a stand-in so the
        # retrieval half of the system is still demonstrable end-to-end.
        return ("[No LLM configured — showing top retrieved passage]\n\n"
                + (context_pairs[0][0].text if context_pairs else "No relevant context found."))

    def answer(self, query: str, k: Optional[int] = None,
               top_n: Optional[int] = None,
               allowed_sources: Optional[set] = None,
               doc_types: Optional[set] = None) -> RAGResult:
        trace = RAGTrace()
        context_pairs, sources = self._rerank(query, k, top_n, allowed_sources,
                                              doc_types, trace)
        with _span(trace, "generate"):
            if self.generator is not None:
                answer_text = self.generator.answer(query, context_pairs)
            else:
                answer_text = self._no_llm_answer(context_pairs)
        report = validate_citations(answer_text, len(sources))
        if settings.validate_citations and not report.ok:
            answer_text = flag_invalid_citations(answer_text, len(sources))
        trace.set("citations", {"status": report.status, "n": report.n_citations,
                                "invalid": report.invalid_markers})
        trace.log(logger)
        return RAGResult(query=query, answer=answer_text, sources=sources,
                         validation={"status": report.status,
                                     "invalid_markers": report.invalid_markers,
                                     "n_citations": report.n_citations},
                         trace=trace.as_dict())

    def answer_stream(self, query: str, k: Optional[int] = None,
                      top_n: Optional[int] = None,
                      history: Optional[List[Tuple[str, str]]] = None,
                      allowed_sources: Optional[set] = None,
                      doc_types: Optional[set] = None,
                      trace: Optional[RAGTrace] = None
                      ) -> Tuple[List[Source], Iterator[str]]:
        """Streaming variant for the interactive assistant.

        Returns the citations immediately (so the UI can render the sources panel
        right away) plus a generator that streams the answer token-by-token. For
        follow-ups, ``history`` rewrites the search query and is also shown to the
        model. ``allowed_sources``/``doc_types`` scope retrieval. If a ``RAGTrace``
        is passed it is filled with per-stage timings/counts for observability.
        """
        search_query = query
        if history and self.generator is not None:
            with _span(trace, "condense"):
                search_query = self.generator.condense_query(history, query)
        context_pairs, sources = self._rerank(search_query, k, top_n,
                                              allowed_sources, doc_types, trace)
        if self.generator is not None:
            token_iter = _stop_after_refusal(
                self.generator.stream_answer(query, context_pairs, history=history))
        else:
            def _fallback() -> Iterator[str]:
                yield self._no_llm_answer(context_pairs)
            token_iter = _fallback()
        return sources, token_iter
