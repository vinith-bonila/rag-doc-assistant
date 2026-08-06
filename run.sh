#!/usr/bin/env bash
# Start the FastAPI backend and the Streamlit frontend in one container.
# Used by the Docker image / Hugging Face Space. The API listens on 8000
# (internal) and Streamlit serves the public UI on $PORT (7860 on Spaces).
set -euo pipefail

# Build the index from bundled sample docs if none is persisted yet.
if [ ! -f "storage/index.faiss" ]; then
  echo "No index found — building from data/sample_docs ..."
  python -m src.ingest --docs data/sample_docs
fi

# Backend (internal).
uvicorn api.main:app --host 127.0.0.1 --port 8000 --log-level warning &

# Frontend (public). API_URL points the UI at the local backend.
# $PORT is provided by the host (Cloud Run sets 8080; falls back to 7860 locally).
# CORS/XSRF are disabled because the app runs behind Cloud Run's TLS proxy, where
# Streamlit's default XSRF check can otherwise reject the websocket connection.
export API_URL="http://127.0.0.1:8000"
exec streamlit run ui/streamlit_app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
