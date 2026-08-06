# API Reference

The FastAPI service (`api/main.py`) exposes the same RAG pipeline the Streamlit
app uses, as REST + SSE. It's optional for the deployed demo (Streamlit calls the
pipeline in-process) but is the API surface for programmatic use and local dev.

Run it:

```bash
uvicorn api.main:app --port 8000
```

Interactive OpenAPI docs: <http://localhost:8000/docs>.

All request/response shapes are defined as Pydantic models in `api/schemas.py`.
Rate limits are per-client-IP (via slowapi).

---

## `GET /health`

Liveness + status. No body.

```json
{ "status": "ok", "indexed": true, "chunks": 12, "hybrid": true, "llm": true }
```

## `POST /ask`

Grounded answer with citations (non-streaming). Rate limit: 30/min.

**Request**

```json
{ "question": "What causes edge-ring defects on wafers?", "k": 20, "top_n": 5 }
```

| Field | Type | Notes |
|---|---|---|
| `question` | string | required, 3–2000 chars |
| `k` | int? | candidates to retrieve before reranking (1–100) |
| `top_n` | int? | passages to keep after reranking (1–20) |

**Response**

```json
{
  "question": "…",
  "answer": "Edge-ring defects are caused by … [1].",
  "sources": [
    { "source": "wafer_defects.md", "page": null, "score": 5.30, "chunk_id": 3, "text": "…" }
  ]
}
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes edge-ring defects on wafers?"}'
```

## `POST /ask/stream`

Same as `/ask` but streamed over **Server-Sent Events**. Rate limit: 30/min.
Events arrive in this order:

1. `event: sources` — `data:` is the JSON array of citations (so the UI paints
   the panel immediately).
2. `event: token` — `data: {"delta": "…"}`, one per streamed chunk of the answer.
3. `event: done` — `data: {}`.

(On an internal error, an `event: error` with `{"message": "..."}` is emitted
before `done`.)

## `POST /upload`

Add documents to the live index. `multipart/form-data` with one or more `files`.
Rate limit: 10/min. Allowed types: `.pdf`, `.docx`, `.txt`, `.md`. Max 15 MB/file.

**Response**

```json
{ "files": [ { "filename": "notes.pdf", "chunks_added": 8 } ], "total_chunks": 20 }
```

Errors: `400` unsupported type, `413` too large.

```bash
curl -X POST http://localhost:8000/upload -F "files=@notes.pdf"
```

## `POST /eval`

Run the evaluation harness over the bundled labelled set and return metrics.
Rate limit: 6/min (it calls the LLM per question).

**Response**

```json
{
  "metrics": [
    { "name": "recall@5", "value": 1.0, "explanation": "…" },
    { "name": "MRR", "value": 0.96, "explanation": "…" },
    { "name": "faithfulness", "value": 0.78, "explanation": "…" },
    { "name": "answer relevance", "value": 0.82, "explanation": "…" }
  ],
  "n_questions": 12,
  "llm_judge": true
}
```
