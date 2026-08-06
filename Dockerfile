FROM python:3.11-slim

ENV PORT=8501 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORKDIR /app

# Install deps first so Docker caches this layer when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedding + reranker models and build the FAISS index from
# the sample docs at build time, so the container answers immediately on boot.
RUN python -m src.ingest --docs data/sample_docs

RUN chmod +x run.sh

EXPOSE 8501

# LLM_API_KEY is provided at runtime (`docker run -e LLM_API_KEY=...`).
# Without it the app still does retrieval + reranking.
CMD ["./run.sh"]
