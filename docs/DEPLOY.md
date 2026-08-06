# Deploying DocMind

The app ships as a single Docker container that runs the FastAPI backend and the
Streamlit UI together. The public entrypoint is the Streamlit UI on `$PORT`; it
calls the FastAPI backend on `127.0.0.1:8000` inside the container.

We deploy in two steps: push the code to **GitHub**, then deploy the container to
**Google Cloud Run** from **Cloud Shell** (browser-based — no local Docker or
gcloud install needed).

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

## Part B — Google Cloud Run

### One-time prerequisites

- A Google account.
- A Google Cloud **project** with **billing enabled**. Cloud Run's free monthly
  allowance is generous (well beyond a demo), but Google requires a billing
  account on file to enable the service. You won't be charged within the free
  tier. Create/select a project at <https://console.cloud.google.com>.

### Deploy from Cloud Shell

1. Open **Cloud Shell** (the `>_` icon, top-right of the Cloud Console). It's a
   free browser terminal with `gcloud` and Docker preinstalled.

2. Clone your repo and enter it:

   ```bash
   git clone https://github.com/<your-username>/rag-doc-assistant.git
   cd rag-doc-assistant
   ```

3. Set your project and enable the needed APIs (first time only):

   ```bash
   gcloud config set project <YOUR_PROJECT_ID>
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com
   ```

4. Deploy. Cloud Build builds the Dockerfile and Cloud Run serves it. Replace the
   LLM values with yours (Groq shown):

   ```bash
   gcloud run deploy rag-assistant \
     --source . \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 1 \
     --timeout 3600 \
     --session-affinity \
     --max-instances 1 \
     --set-env-vars LLM_API_KEY=YOUR_GROQ_KEY,LLM_BASE_URL=https://api.groq.com/openai/v1,LLM_MODEL=llama-3.1-8b-instant
   ```

   The first build takes several minutes (installs deps, downloads the MiniLM +
   cross-encoder models, builds the sample index into the image). When it
   finishes, gcloud prints a **Service URL** like
   `https://rag-assistant-xxxxxxxx-uc.a.run.app` — that's your live link.

5. Open the URL. Paste it into the README's top line and commit:

   ```bash
   git commit -am "Add live Cloud Run URL to README" && git push
   ```

### Why these flags

| Flag | Why |
|---|---|
| `--memory 2Gi` | The backend loads PyTorch + two transformer models (~600–900 MB resident). 2 GiB gives headroom; 512 MB would OOM. |
| `--session-affinity` | Streamlit uses a websocket; affinity keeps a client pinned to one instance so the socket stays stable. |
| `--max-instances 1` | Uploads mutate the in-container FAISS index; a single instance keeps that state consistent for the demo. |
| `--timeout 3600` | Allows long-lived websocket / streaming connections. |
| `--allow-unauthenticated` | Makes it a public demo URL. |

### Better: keep the key in Secret Manager (optional)

Instead of `--set-env-vars` for the key, store it as a secret:

```bash
echo -n "YOUR_GROQ_KEY" | gcloud secrets create llm-api-key --data-file=-
gcloud run services update rag-assistant --region us-central1 \
  --update-secrets LLM_API_KEY=llm-api-key:latest \
  --set-env-vars LLM_BASE_URL=https://api.groq.com/openai/v1,LLM_MODEL=llama-3.1-8b-instant
```

---

## Notes

- **Uploads are ephemeral.** The container filesystem resets on new revisions /
  scale-to-zero, so user-uploaded docs don't persist across restarts. The bundled
  sample corpus always ships in the image. Durable uploads would need a mounted
  volume or an external vector store (out of scope for v1).
- **Cold starts.** With `--max-instances 1` and scale-to-zero, the first request
  after idle spins the container up (a few seconds, plus model load). Set
  `--min-instances 1` to avoid cold starts (leaves one instance warm; uses more
  of the free allowance).
- **First query is slow-ish** on 1 vCPU (CPU inference for embed + rerank).
  Bump `--cpu 2` if you want snappier responses.
