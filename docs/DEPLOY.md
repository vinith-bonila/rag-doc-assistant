# Deploying DocMind to Hugging Face Spaces

The app ships as a single Docker container that runs the FastAPI backend and the
Streamlit UI together, so it fits one **Docker Space**. Free tier is enough.

## 1. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. **Owner / Space name**: e.g. `your-name/docmind`.
3. **SDK**: choose **Docker** → **Blank**.
4. **Hardware**: CPU basic (free) is fine.
5. Create the Space. It gives you a git repo URL like
   `https://huggingface.co/spaces/your-name/docmind`.

## 2. Add the Space metadata

Hugging Face reads a YAML front-matter block at the top of the Space's
`README.md`. Prepend this block (the content of
[`space_frontmatter.md`](space_frontmatter.md)) to the README **on the Space**
(you can keep the GitHub README clean and only add this on the HF side):

```yaml
---
title: DocMind RAG Assistant
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

`app_port: 7860` must match the `EXPOSE`/`PORT` in the Dockerfile.

## 3. Set the LLM key as a secret

In the Space: **Settings → Variables and secrets → New secret**:

| Name | Value |
|---|---|
| `LLM_API_KEY` | your Groq or OpenAI key |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` (Groq) or leave default for OpenAI |
| `LLM_MODEL` | `llama-3.1-8b-instant` (Groq) or `gpt-4o-mini` (OpenAI) |

Never commit the key — `.env` is gitignored. Without a key the Space still runs
retrieval + reranking (answers show the top passage).

## 4. Push the code

```bash
git init
git remote add space https://huggingface.co/spaces/your-name/docmind
git add .
git commit -m "Deploy DocMind"
git push space main
```

The Space builds the image (installs deps, downloads the MiniLM + cross-encoder
models, builds the sample index) and boots. First build takes a few minutes.
When it's live, the public URL is your definition-of-done link — put it at the
top of the README.

## Notes

- **Model download at build time.** The Dockerfile runs `src.ingest` during
  build, which pulls the embedding + reranker models into the image, so the
  container answers immediately on boot instead of downloading on first request.
- **Uploads are ephemeral on Spaces.** The container filesystem resets on
  rebuild/restart, so user-uploaded docs don't persist across restarts. The
  bundled sample corpus always ships in the image. For durable uploads, attach a
  persistent volume or an external vector store (out of scope for v1).
- **Alternative hosts.** The same container runs on Render/Railway — set the
  service port to `7860` (or read `$PORT`) and add the same secrets.
