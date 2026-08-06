"""Shared pytest fixtures.

The pipeline fixture is session-scoped because it loads the embedding and
cross-encoder models once (first run downloads them, then they're cached).
"""
from __future__ import annotations

import pytest

from src.pipeline import RAGPipeline


@pytest.fixture(scope="session")
def pipeline() -> RAGPipeline:
    return RAGPipeline()
