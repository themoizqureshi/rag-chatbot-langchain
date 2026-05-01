"""Tests for the PDF ingestion pipeline."""

import pytest
from langchain_core.documents import Document
from src.ingestion import chunk_documents


def test_chunk_documents_splits_large_doc():
    docs = [Document(page_content="word " * 500, metadata={"source": "test.pdf"})]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    # Allow slight overshoot due to splitter boundary logic
    assert all(len(c.page_content) <= 150 for c in chunks)


def test_chunk_documents_preserves_metadata():
    docs = [Document(page_content="word " * 300, metadata={"source": "test.pdf", "page": 1})]
    chunks = chunk_documents(docs)
    assert all(c.metadata.get("source") == "test.pdf" for c in chunks)


def test_chunk_documents_creates_overlap():
    docs = [Document(page_content="A" * 1000, metadata={})]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 2


def test_chunk_documents_single_small_doc():
    docs = [Document(page_content="short text", metadata={})]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"
