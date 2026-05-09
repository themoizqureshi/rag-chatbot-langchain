"""
Stage 6a — Integration tests: full pipeline, no mocks.

Uses FastAPI's TestClient with a real (tmp) ChromaDB.
Uploads a fixture PDF, asks a question, asserts answer quality.

Requires:
    OPENROUTER_API_KEY or GOOGLE_API_KEY in environment
    (the test makes real LLM calls — skip with: pytest -m "not integration")

Run:
    pytest tests/test_integration.py -v -m integration
"""

import os
import textwrap
import pytest
from fastapi.testclient import TestClient

# Mark all tests in this file as integration so they can be skipped in fast CI
pytestmark = pytest.mark.integration

FIXTURE_TEXT = textwrap.dedent("""\
    Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval step
    with a language model generation step to produce grounded answers.

    In a RAG pipeline, the system first retrieves relevant documents from a vector store
    using semantic similarity search. The retrieved documents are then injected into the
    prompt as context, and the LLM generates an answer based solely on that context.

    RAG reduces hallucination by anchoring the model's output in retrieved evidence.
    The main components are: a document ingestion pipeline, an embedding model, a vector
    database (e.g. ChromaDB or Pinecone), a retriever, and an LLM.

    Evaluation of RAG pipelines commonly uses RAGAS, which measures faithfulness,
    answer relevancy, context recall, and context precision.
""")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Build TestClient against a temporary ChromaDB directory."""
    tmp_dir = tmp_path_factory.mktemp("chroma")
    os.environ["CHROMA_DB_PATH"] = str(tmp_dir)

    from src.api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def session_id(client, tmp_path_factory):
    """Ingest the fixture text as a PDF-like file and return the session_id."""
    tmp_dir = tmp_path_factory.mktemp("fixtures")
    txt_path = tmp_dir / "rag_overview.txt"
    txt_path.write_text(FIXTURE_TEXT)

    # The /ingest endpoint expects a file upload
    with open(txt_path, "rb") as f:
        resp = client.post("/ingest", files={"file": ("rag_overview.txt", f, "text/plain")})

    assert resp.status_code == 200, f"Ingest failed: {resp.text}"
    data = resp.json()
    assert "session_id" in data
    assert data["chunks_indexed"] > 0
    return data["session_id"]


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Ingest ────────────────────────────────────────────────────────────────────

def test_ingest_returns_session_id(session_id):
    assert isinstance(session_id, str) and len(session_id) > 0


def test_ingest_unknown_session_returns_404(client):
    resp = client.post("/chat", json={"question": "hello", "session_id": "nonexistent-session"})
    assert resp.status_code == 404


# ── Chat ──────────────────────────────────────────────────────────────────────

def test_chat_returns_grounded_answer(client, session_id):
    resp = client.post("/chat", json={
        "question": "What is RAG?",
        "session_id": session_id,
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert "answer" in data
    assert len(data["answer"]) > 20, "Answer is suspiciously short"
    # Answer should reference retrieval or generation
    answer_lower = data["answer"].lower()
    assert any(kw in answer_lower for kw in ["retrieval", "generation", "rag", "context"])


def test_chat_returns_sources(client, session_id):
    resp = client.post("/chat", json={
        "question": "What components make up a RAG pipeline?",
        "session_id": session_id,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "sources" in data
    assert len(data["sources"]) > 0


def test_chat_returns_usage_info(client, session_id):
    resp = client.post("/chat", json={
        "question": "What is faithfulness in RAGAS?",
        "session_id": session_id,
    })
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["cost_usd"] >= 0.0


def test_chat_refuses_out_of_scope_question(client, session_id):
    """The guardrail should block questions unrelated to the ingested document."""
    resp = client.post("/chat", json={
        "question": "Write me a Python script to scrape Twitter.",
        "session_id": session_id,
    })
    # Either 400 (guardrail rejection) or 200 with a refusal message
    if resp.status_code == 200:
        data = resp.json()
        guardrail = data.get("guardrail", {})
        # If status 200, guardrail should have flagged it
        assert not guardrail.get("input_passed", True) or "don't have" in data["answer"].lower()
    else:
        assert resp.status_code == 400


def test_chat_stream_returns_sse(client, session_id):
    """Streaming endpoint should return Server-Sent Events."""
    with client.stream("POST", "/chat/stream", json={
        "question": "What is RAG?",
        "session_id": session_id,
    }) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = b"".join(resp.iter_raw())
        assert b"data:" in body
