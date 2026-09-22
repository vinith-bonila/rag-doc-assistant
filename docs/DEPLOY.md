# Deploying DocMind (free)

DocMind deploys **free** on **Streamlit Community Cloud** — no billing account, no
credit card. The Streamlit app runs the RAG pipeline in-process, so there's only
one service to host.

Two steps: push to **GitHub**, then point **Streamlit Community Cloud** at the repo.

---

## Part A — GitHub

1. Create an empty repo at <https://github.com/new> (e.g. `rag-doc-assistant`).
   Don't add a README/.gitignore — the repo already has them.
2. Create a Personal Access Token (classic) with `repo` scope at
   <https://github.com/settings/tokens> — you'll paste it as the password on push.
3. From the project folder:

   ```bash
   git remote add origin https://github.com/<your-username>/rag-doc-assistant.git
   git branch -M main
   git push -u origin main
   ```

   (Username = your GitHub handle; password = the token from step 2.)

---

## Part B — Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub. Authorize access to
   your repo if prompted.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository**: `<your-username>/rag-doc-assistant`
   - **Branch**: `main`
   - **Main file path**: `ui/streamlit_app.py`
   - **Python version** (Advanced settings): choose **3.11** or **3.12** — the
     most reliable prebuilt wheels for `torch` / `faiss-cpu` / `sentence-transformers`.
     (`requirements.txt` already forces the CPU-only PyTorch build, so the install
     stays small enough for the free tier.)
4. Open **Advanced settings → Secrets** and paste (TOML format):

   ```toml
   LLM_API_KEY = "your-groq-key"
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "openai/gpt-oss-20b"
   ```

   The app reads these into its config at startup (see the secrets bridge in
   `ui/streamlit_app.py`). Without a key it still runs retrieval + reranking.
5. Click **Deploy**. First boot installs dependencies, downloads the MiniLM +
   cross-encoder models, and builds the sample index (a couple of minutes). When
   it's live, copy the `https://<something>.streamlit.app` URL into the README's
   top line and commit.

### Memory notes (Streamlit Cloud has ~1 GB RAM)

Already handled in code, but good to understand:

- **One cached pipeline.** `ui/streamlit_app.py` builds the pipeline via
  `st.cache_resource`, so models load once and are reused across reruns/users
  (Streamlit reruns the whole script on every interaction).
- **Lazy loading.** The embedder, reranker, and LLM client are constructed on
  first use, so boot stays light; the single embedder instance is reused for
  indexing, retrieval, and the embedding-based eval metrics — no model is ever
  loaded twice.
- **Small models.** MiniLM-class embedder + cross-encoder (~90 MB each) are chosen
  for the RAM/quality tradeoff.

If you ever hit the memory limit, the cheapest wins are: keep `HYBRID=true` but
lower `RETRIEVE_K`, or disable the reranker via a smaller shortlist.

### Notes

- **Uploads are ephemeral.** Streamlit Cloud's filesystem resets on restart/redeploy,
  so user-uploaded docs don't persist across restarts. The bundled sample corpus
  always ships in the repo. Durable uploads would need an external vector store
  (out of scope for v1).
- **Updating the app.** Push to `main`; Streamlit Cloud auto-redeploys.

---

## Alternative: Docker (self-host)

The repo also ships a Dockerfile that runs the same Streamlit app:

```bash
docker build -t docmind .
docker run -p 8501:8501 \
  -e LLM_API_KEY=... -e LLM_BASE_URL=https://api.groq.com/openai/v1 -e LLM_MODEL=openai/gpt-oss-20b \
  docmind
```

Useful for any container host or local runs; Streamlit Community Cloud remains the
free, zero-ops path.
