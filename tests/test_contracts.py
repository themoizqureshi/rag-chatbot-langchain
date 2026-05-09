"""
Stage 6d — Contract tests: API schema stability.

Asserts that ChatResponse, IngestResponse, and /health response shapes
match the expected schema. Catches silent schema breaks before they reach
consumers (eval pipeline, multi-agent tool, portfolio UI).

No LLM calls — the schema shapes are checked from Pydantic model definitions
directly, not from live API responses. Fast enough for pre-commit.

Run:
    pytest tests/test_contracts.py -v
"""

import pytest
from pydantic import BaseModel


# ── Import the Pydantic models from the API module ────────────────────────────

from src.api import ChatRequest, ChatResponse, IngestResponse, SourceItem, UsageInfo, GuardrailInfo


# ── ChatRequest contract ──────────────────────────────────────────────────────

class TestChatRequestContract:
    def test_has_question_field(self):
        fields = ChatRequest.model_fields
        assert "question" in fields

    def test_has_session_id_field(self):
        assert "session_id" in ChatRequest.model_fields

    def test_question_is_required(self):
        with pytest.raises(Exception):
            ChatRequest(session_id="abc")  # missing question

    def test_session_id_is_required(self):
        with pytest.raises(Exception):
            ChatRequest(question="hello")  # missing session_id

    def test_valid_construction(self):
        req = ChatRequest(question="What is RAG?", session_id="sess-123")
        assert req.question == "What is RAG?"
        assert req.session_id == "sess-123"


# ── SourceItem contract ───────────────────────────────────────────────────────

class TestSourceItemContract:
    def test_has_page_field(self):
        assert "page" in SourceItem.model_fields

    def test_has_snippet_field(self):
        assert "snippet" in SourceItem.model_fields

    def test_valid_construction(self):
        s = SourceItem(page=3, snippet="Some text from page 3")
        assert s.page == 3
        assert s.snippet == "Some text from page 3"


# ── UsageInfo contract ────────────────────────────────────────────────────────

class TestUsageInfoContract:
    def test_required_fields(self):
        fields = UsageInfo.model_fields
        assert "input_tokens" in fields
        assert "output_tokens" in fields
        assert "cost_usd" in fields

    def test_valid_construction(self):
        u = UsageInfo(input_tokens=100, output_tokens=50, cost_usd=0.000042)
        assert u.input_tokens == 100
        assert u.output_tokens == 50
        assert u.cost_usd == pytest.approx(0.000042)


# ── GuardrailInfo contract ────────────────────────────────────────────────────

class TestGuardrailInfoContract:
    def test_required_fields(self):
        fields = GuardrailInfo.model_fields
        assert "input_passed" in fields
        assert "output_passed" in fields

    def test_warning_is_optional(self):
        g = GuardrailInfo(input_passed=True, output_passed=True)
        assert g.warning is None

    def test_valid_with_warning(self):
        g = GuardrailInfo(input_passed=False, output_passed=True, warning="Off-topic query")
        assert g.warning == "Off-topic query"


# ── ChatResponse contract ─────────────────────────────────────────────────────

class TestChatResponseContract:
    def test_has_all_required_fields(self):
        fields = ChatResponse.model_fields
        assert "answer" in fields
        assert "sources" in fields
        assert "usage" in fields
        assert "guardrail" in fields

    def test_sources_is_list_of_source_items(self):
        # Check type annotation contains SourceItem
        annotation = str(ChatResponse.model_fields["sources"].annotation)
        assert "SourceItem" in annotation

    def test_valid_construction(self):
        resp = ChatResponse(
            answer="RAG is a technique...",
            sources=[SourceItem(page=1, snippet="RAG combines retrieval...")],
            usage=UsageInfo(input_tokens=200, output_tokens=80, cost_usd=0.0001),
            guardrail=GuardrailInfo(input_passed=True, output_passed=True),
        )
        assert resp.answer == "RAG is a technique..."
        assert len(resp.sources) == 1
        assert resp.usage.input_tokens == 200

    def test_serializes_to_dict(self):
        resp = ChatResponse(
            answer="Test answer",
            sources=[],
            usage=UsageInfo(input_tokens=10, output_tokens=5, cost_usd=0.0),
            guardrail=GuardrailInfo(input_passed=True, output_passed=True),
        )
        d = resp.model_dump()
        assert set(d.keys()) == {"answer", "sources", "usage", "guardrail"}
        assert isinstance(d["sources"], list)


# ── IngestResponse contract ───────────────────────────────────────────────────

class TestIngestResponseContract:
    def test_has_all_required_fields(self):
        fields = IngestResponse.model_fields
        assert "message" in fields
        assert "chunks_indexed" in fields
        assert "session_id" in fields
        assert "retriever_mode" in fields

    def test_valid_construction(self):
        r = IngestResponse(
            message="Indexed successfully",
            chunks_indexed=47,
            session_id="sess-abc-123",
            retriever_mode="hybrid",
        )
        assert r.chunks_indexed == 47
        assert r.retriever_mode == "hybrid"
