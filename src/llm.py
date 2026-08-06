"""Answer generation via an OpenAI-compatible chat API.

Two modes share one retrieval/citation core:

* ``answer`` — strict **grounded** answer: only from the retrieved context, cited
  inline as [1], [2], refuses when the context is insufficient. Used by the API
  and by the evaluation harness (so faithfulness measures the RAG core honestly).

* ``stream_answer`` — the interactive **AI document assistant**: a grounded,
  cited answer first, then a clearly separated plain-language explanation and an
  optional example. It also takes conversation ``history`` so follow-ups
  ("why is it important?") stay in context.

* ``condense_query`` — rewrites a follow-up into a standalone search query using
  the conversation, so retrieval works on pronoun-y follow-ups.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

from openai import OpenAI

from .chunking import Chunk

# (user_text, assistant_text) pairs, oldest first.
Turn = Tuple[str, str]

# --- grounded mode (API + eval): strict, cited, refuses ----------------------
SYSTEM_PROMPT = (
    "You are a precise assistant that answers strictly from the provided context. "
    "Cite the sources you use with inline markers like [1], [2] that refer to the "
    "numbered context passages. If the context does not contain the answer, say "
    "\"I don't have enough information in the provided documents to answer that.\" "
    "Do not use outside knowledge."
)

# --- assistant mode (UI): grounded-first, then teaches -----------------------
ASSISTANT_SYSTEM_PROMPT = (
    "You are DocMind, an AI document assistant. The user's uploaded document(s) are "
    "the primary source of truth; you are given numbered context passages retrieved "
    "from them. The document can be anything — a report, research paper, resume, SOP, "
    "manual, legal/financial/medical document, source code, policy, or book — so stay "
    "domain-agnostic and handle any request (explain, summarize, teach, compare, "
    "translate, rewrite, find risks/action items/conclusions, make questions, etc.).\n\n"
    "Reply in GitHub markdown with these sections, in order:\n\n"
    "📄 **According to the document**\n"
    "A grounded answer using ONLY the retrieved passages, with inline citations "
    "like [1], [2]. If the passages do not contain enough information, write exactly: "
    "\"The uploaded document does not contain enough information to answer this "
    "question.\" and then STOP — do not add the sections below.\n\n"
    "💡 **AI explanation**\n"
    "Explain it in simple, clear language so the user genuinely understands — do not "
    "just repeat the document. You may use general knowledge here, but never "
    "contradict the document.\n\n"
    "🌍 **Example** (include ONLY if it genuinely helps)\n"
    "A short real-world example or analogy.\n\n"
    "Rules:\n"
    "- Keep 📄 (strictly from the document, cited) and 💡 (your own explanation) "
    "clearly separate; never blend outside knowledge into 📄.\n"
    "- Only ignore the document or answer purely from your own knowledge if the user "
    "explicitly asks (e.g. \"explain using your own knowledge\", \"ignore the document\").\n"
    "- Use the conversation so far to resolve follow-up references (\"it\", \"that\")."
)


def _format_context(chunks: List[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        loc = f", page {c.page}" if c.page is not None else ""
        blocks.append(f"[{i}] (source: {c.source}{loc})\n{c.text}")
    return "\n\n".join(blocks)


def _history_messages(history: List[Turn], max_turns: int = 3,
                      max_chars: int = 800) -> list:
    """Recent turns as chat messages, truncated to keep the prompt bounded."""
    msgs: list = []
    for user_text, assistant_text in (history or [])[-max_turns:]:
        msgs.append({"role": "user", "content": user_text[:max_chars]})
        msgs.append({"role": "assistant", "content": assistant_text[:max_chars]})
    return msgs


class Generator:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # --- grounded (API + eval) ------------------------------------------------
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

    # --- conversation memory --------------------------------------------------
    def condense_query(self, history: List[Turn], question: str) -> str:
        """Rewrite a follow-up into a standalone search query using the
        conversation. Returns the original question if there's no history or on
        any error (retrieval degrades gracefully rather than breaking)."""
        if not history:
            return question
        convo = "\n".join(
            f"User: {u}\nAssistant: {a[:300]}" for u, a in history[-3:]
        )
        prompt = (
            "Given the conversation and a follow-up question, rewrite the follow-up "
            "as a standalone search query that makes sense on its own (resolve "
            "pronouns like 'it'/'that'). Output ONLY the query, nothing else.\n\n"
            f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone query:"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = (resp.choices[0].message.content or "").strip().strip('"')
            return rewritten or question
        except Exception:  # noqa: BLE001 — never let query-rewrite break a query
            return question

    # --- assistant (UI), streamed with memory ---------------------------------
    def stream_answer(self, query: str, chunks: List[Tuple[Chunk, float]],
                      history: List[Turn] | None = None) -> Iterator[str]:
        """Stream the structured assistant answer token-by-token.

        Yields content deltas; the caller concatenates them for the full answer
        and forwards each to the UI as it arrives.
        """
        context = _format_context([c for c, _ in chunks])
        user = (
            f"Context passages (the uploaded document):\n{context}\n\n"
            f"User request: {query}"
        )
        messages = (
            [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]
            + _history_messages(history)
            + [{"role": "user", "content": user}]
        )
        stream = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=messages,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta
