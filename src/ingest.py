"""Parse documents -> chunk -> embed -> persist a FAISS index.

Supports .txt, .md, .pdf and .docx. Run once before serving:

    python -m src.ingest --docs data/sample_docs

The parsing layer returns *segments* — (page, text) pairs — so that PDF page
numbers survive all the way to the citations shown in the UI. Formats without a
reliable page model (txt/md/docx) yield a single page=None segment.

`parse_bytes` / `chunk_document` are also imported by the API's upload endpoint
so that indexing an uploaded file and indexing a folder share one code path.
"""
from __future__ import annotations

import argparse
import glob
import io
import logging
import os
import re
import unicodedata
from typing import List, Optional, Tuple

from config import settings

from .chunking import Chunk, recursive_chunk, semantic_chunk
from .embeddings import Embedder
from .vectorstore import FaissStore

logger = logging.getLogger(__name__)

# A parsed unit of a document: (1-indexed page or None, text).
Segment = Tuple[Optional[int], str]

SUPPORTED_EXTS = (".txt", ".md", ".pdf", ".docx")

# Codepoints are written with chr() so this source stays pure ASCII (no fragile
# literal glyphs). U+019F / U+0275 are the barred-O that broken PDF fonts emit
# for a "ti" ligature; mapping back to "ti" is safe (they never occur in real
# English / technical text). e.g. "opera<U+019F>ons" -> "operations".
_LIGATURE_FIXES = {
    chr(0x019F): "ti",   # LATIN CAPITAL LETTER O WITH MIDDLE TILDE
    chr(0x0275): "ti",   # LATIN SMALL LETTER BARRED O
}
# Private-use area (U+E000-U+F8FF): font-specific glyphs (bullets, icons).
_PUA_RE = re.compile("[" + chr(0xE000) + "-" + chr(0xF8FF) + "]")
_PUA_BULLET = chr(0xF0B7)      # the most common private-use bullet glyph
_BULLET = chr(0x2022)          # a real bullet
_REPLACEMENT = chr(0xFFFD)     # replacement char from failed decodes


def clean_pdf_text(text: str) -> str:
    """Repair common PDF text-extraction artifacts.

    1. NFKC expands real ligatures (U+FB00-FB06: ff, fi, fl, ffi, ...) to letters.
    2. Known broken ligature mappings are corrected (see ``_LIGATURE_FIXES``).
    3. Private-use glyphs become a real bullet or a space, and the replacement
       char (U+FFFD) is stripped.
    """
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _LIGATURE_FIXES.items():
        text = text.replace(bad, good)
    text = text.replace(_PUA_BULLET, _BULLET + " ")   # bullet glyph -> "* "
    text = _PUA_RE.sub(" ", text)                     # any other private-use glyphs
    return text.replace(_REPLACEMENT, " ")


# --- parsing ------------------------------------------------------------------
def _parse_pdf(data: bytes) -> List[Segment]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    segments: List[Segment] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean_pdf_text(page.extract_text() or "").strip()
        if text:
            segments.append((page_no, text))
    return segments


def _parse_docx(data: bytes) -> List[Segment]:
    from docx import Document
    doc = Document(io.BytesIO(data))
    # DOCX has no reliable page boundaries without rendering, so we treat the
    # whole document as one page=None segment built from its paragraphs.
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(None, text)] if text.strip() else []


def parse_bytes(filename: str, data: bytes) -> List[Segment]:
    """Parse raw file bytes into (page, text) segments based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _parse_pdf(data)
    if ext == ".docx":
        return _parse_docx(data)
    if ext in (".txt", ".md"):
        text = data.decode("utf-8", errors="ignore").strip()
        return [(None, text)] if text else []
    raise ValueError(f"Unsupported file type: {ext!r} (supported: {SUPPORTED_EXTS})")


def parse_file(path: str) -> List[Segment]:
    with open(path, "rb") as f:
        return parse_bytes(os.path.basename(path), f.read())


# --- chunking -----------------------------------------------------------------
def chunk_document(source: str, segments: List[Segment], embedder: Embedder,
                   start_id: int = 0) -> List[Chunk]:
    """Chunk one document's segments, preserving page numbers, with globally
    unique chunk ids continuing from ``start_id``."""
    chunks: List[Chunk] = []
    next_id = start_id
    for page, text in segments:
        if not text.strip():
            continue
        if settings.chunk_strategy == "semantic":
            seg_chunks = semantic_chunk(text, source, embedder,
                                        max_chars=settings.chunk_size,
                                        start_id=next_id, page=page)
        else:
            seg_chunks = recursive_chunk(text, source, settings.chunk_size,
                                         settings.chunk_overlap,
                                         start_id=next_id, page=page)
        next_id += len(seg_chunks)
        chunks.extend(seg_chunks)
    return chunks


# --- folder ingestion ---------------------------------------------------------
def load_documents(docs_dir: str) -> List[Tuple[str, List[Segment]]]:
    paths: List[str] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(docs_dir, f"**/*{ext}"), recursive=True))
    docs: List[Tuple[str, List[Segment]]] = []
    for p in sorted(set(paths)):
        segments = parse_file(p)
        if segments:
            docs.append((os.path.basename(p), segments))
    return docs


def build_index(docs_dir: str, out_dir: str,
                embedder: Optional[Embedder] = None) -> None:
    # Reuse the caller's embedder when given (e.g. the pipeline's) so the model
    # isn't loaded twice; otherwise construct one for standalone CLI use.
    embedder = embedder or Embedder(settings.embed_model)
    docs = load_documents(docs_dir)
    if not docs:
        raise SystemExit(f"No supported documents found in {docs_dir!r}")

    all_chunks: List[Chunk] = []
    for source, segments in docs:
        all_chunks.extend(chunk_document(source, segments, embedder,
                                         start_id=len(all_chunks)))

    logger.info("Loaded %d document(s) -> %d chunks [%s]",
                len(docs), len(all_chunks), settings.chunk_strategy)
    print(f"Loaded {len(docs)} document(s) -> {len(all_chunks)} chunks "
          f"[{settings.chunk_strategy}]")

    vectors = embedder.encode([c.text for c in all_chunks])
    store = FaissStore(dim=embedder.dim)
    store.add(vectors, all_chunks)
    store.save(out_dir)
    print(f"Index saved to {out_dir!r} ({store.index.ntotal} vectors)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/sample_docs")
    ap.add_argument("--out", default=settings.index_dir)
    args = ap.parse_args()
    build_index(args.docs, args.out)
