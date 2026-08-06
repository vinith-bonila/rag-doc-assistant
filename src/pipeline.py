"""End-to-end RAG pipeline: retrieve -> rerank -> generate (with citations).

Loading the models is expensive, so the pipeline is built once and reused
(the FastAPI app keeps a single instance alive across requests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from config import settings

from .chunking import Chunk
from .embeddings import Embedder
from .ingest import chunk_document, parse_bytes
from .llm import Generator
from .reranker import Reranker
from .retrieval import build_retriever
from .vectorstore import FaissStore


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
    def __init__(self, index_dir: Optional[str] = None):
        index_dir = index_dir or settings.index_dir
        self.index_dir = index_dir
        self.embedder = Embedder(settings.embed_model)
        self.store = FaissStore.load(index_dir)
        self._build_retriever()
        self.reranker = Reranker(settings.reranker_model)
        self.generator: Optional[Generator] = None
        if settings.has_llm:
            self.generator = Generator(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
            )

    def _build_retriever(self) -> None:
        self.retriever = build_retriever(
            self.store, self.embedder,
            hybrid=settings.hybrid, rrf_k=settings.rrf_k,
        )

    def add_documents(self, files: List[Tuple[str, bytes]]) -> dict:
        """Parse, chunk, embed and index uploaded files at runtime, then persist.

        Returns a small summary (per-file chunk counts + new total). Rebuilds the
        BM25 side of the hybrid retriever, which must see the full chunk set.
        """
        added: List[dict] = []
        for filename, data in files:
            segments = parse_bytes(filename, data)  # raises ValueError if unsupported
            chunks = chunk_document(filename, segments, self.embedder,
                                    start_id=len(self.store.chunks))
            if not chunks:
                added.append({"filename": filename, "chunks_added": 0})
                continue
            vectors = self.embedder.encode([c.text for c in chunks])
            self.store.add(vectors, chunks)
            added.append({"filename": filename, "chunks_added": len(chunks)})

        self._build_retriever()          # BM25 must be rebuilt over all chunks
        self.store.save(self.index_dir)  # persist so the new docs survive restart
        return {"files": added, "total_chunks": len(self.store.chunks)}

    def retrieve(self, query: str, k: Optional[int] = None
                 ) -> List[Tuple[Chunk, float]]:
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
