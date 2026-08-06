"""Grounded answer generation via an OpenAI-compatible chat API.

Key design choices that matter for a RAG project:
  * We number the retrieved passages and instruct the model to cite them inline
    as [1], [2], ... so every claim is traceable to a source.
  * We tell the model to answer ONLY from the context and to say it doesn't
    know when the context is insufficient. This is the first line of defence
    against hallucination.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

from openai import OpenAI

from .chunking import Chunk

SYSTEM_PROMPT = (
    "You are a precise assistant that answers strictly from the provided context. "
    "Cite the sources you use with inline markers like [1], [2] that refer to the "
    "numbered context passages. If the context does not contain the answer, say "
    "\"I don't have enough information in the provided documents to answer that.\" "
    "Do not use outside knowledge."
)


def _format_context(chunks: List[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {c.source})\n{c.text}")
    return "\n\n".join(blocks)


class Generator:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _messages(self, query: str, chunks: List[Tuple[Chunk, float]]) -> list:
        context = _format_context([c for c, _ in chunks])
        user = (
            f"Context passages:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above and cite sources as [n]."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def answer(self, query: str, chunks: List[Tuple[Chunk, float]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=self._messages(query, chunks),
        )
        return resp.choices[0].message.content.strip()

    def stream_answer(self, query: str,
                      chunks: List[Tuple[Chunk, float]]) -> Iterator[str]:
        """Yield answer text incrementally as the model produces it.

        Yields content deltas (token-ish substrings); the caller concatenates
        them for the full answer and can forward each to the UI as it arrives.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=self._messages(query, chunks),
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta
