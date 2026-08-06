#!/usr/bin/env bash
# Launch the DocMind Streamlit app (which runs the RAG pipeline in-process).
# Used by the Docker image. $PORT is honored if the host sets it (falls back to
# Streamlit's default 8501 for a bare `docker run`).
set -euo pipefail

# The pipeline self-builds the index on first run, but pre-building here keeps
# container startup snappy.
if [ ! -f "storage/index.faiss" ]; then
  echo "No index found — building from data/sample_docs ..."
  python -m src.ingest --docs data/sample_docs
fi

exec streamlit run ui/streamlit_app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
