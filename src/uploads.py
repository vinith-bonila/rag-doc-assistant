"""Server-side upload validation — one enforcement point for every entry path.

Both the Streamlit uploader and the FastAPI ``/upload`` endpoint funnel through
``RAGPipeline.add_documents``, which calls ``validate_upload`` below. The
Streamlit file-picker's ``type=[...]`` list and ``maxUploadSize`` in
.streamlit/config.toml are **client/server hints only** — they are not a
security boundary, so the same rules are re-checked here (docs/SECURITY_AUDIT.md
DM-08).

Checks, in order:
  1. a safe, non-traversing filename;
  2. an extension on the allow-list;
  3. a per-file byte limit;
  4. a magic-byte sanity check for the formats that have one, so a renamed
     binary (``malware.exe`` -> ``report.pdf``) is rejected before any parser
     touches it rather than being handed to pypdf/python-docx.

Raises ``UploadRejected`` (a ``ValueError``) with a message that is safe to show
a user: it names the file and the rule, never a filesystem path or secret.
"""
from __future__ import annotations

import os
from typing import Tuple

from .ingest import SUPPORTED_EXTS


class UploadRejected(ValueError):
    """An upload failed validation. Subclasses ValueError so existing callers
    that catch ValueError (the API's 400 handler, tests) keep working."""


# Leading bytes that identify the binary formats we accept. Text formats
# (.txt/.md) have no magic number, so they are checked for decodability instead.
_MAGIC = {
    ".pdf": (b"%PDF-",),
    ".docx": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),   # zip container
}


def safe_filename(filename: str) -> str:
    """Strip any directory component and reject names that cannot be used as a
    plain document label.

    Uploaded names are only ever used as a *display/citation label* (never as a
    path), but normalising here means a crafted ``../../etc/passwd`` can never
    become one if a future change does write to disk.
    """
    if not filename or not filename.strip():
        raise UploadRejected("A file was uploaded without a name.")
    # Cut on both separators: a Windows client can send "..\..\x.pdf".
    base = os.path.basename(filename.replace("\\", "/")).strip()
    base = base.lstrip(".") if base in (".", "..") else base
    if not base or base in (".", ".."):
        raise UploadRejected(f"{filename!r} is not a usable file name.")
    if "\x00" in base:
        raise UploadRejected(f"{filename!r} contains an invalid character.")
    return base


def validate_upload(filename: str, data: bytes, max_bytes: int) -> Tuple[str, bytes]:
    """Return the sanitised ``(filename, data)`` or raise ``UploadRejected``."""
    name = safe_filename(filename)
    ext = os.path.splitext(name)[1].lower()

    if ext not in SUPPORTED_EXTS:
        allowed = ", ".join(SUPPORTED_EXTS)
        raise UploadRejected(
            f"{name!r} is not a supported file type. Supported: {allowed}.")

    if not data:
        raise UploadRejected(f"{name!r} is empty.")

    if len(data) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        actual_mb = len(data) / (1024 * 1024)
        raise UploadRejected(
            f"{name!r} is {actual_mb:.1f} MB, over the {limit_mb:.0f} MB limit.")

    magics = _MAGIC.get(ext)
    if magics and not data.startswith(magics):
        raise UploadRejected(
            f"{name!r} does not look like a real {ext.lstrip('.').upper()} file "
            "(its contents don't match its extension).")

    if ext in (".txt", ".md"):
        # Reject binaries renamed to .txt/.md: a NUL byte in the first block is
        # a reliable signal, and decoding is lossy-safe downstream anyway.
        if b"\x00" in data[:8192]:
            raise UploadRejected(
                f"{name!r} looks like a binary file, not plain text.")

    return name, data
