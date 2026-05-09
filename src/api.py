"""
FastAPI REST API for the RAG chatbot.

Endpoints:
  GET  /health       — Liveness check
  POST /ingest       — Upload and index a PDF into ChromaDB (returns session_id)
  POST /chat         — Ask a question, returns full answer + sources + cost
  POST /chat/stream  — Same but streams answer tokens via Server-Sent Events

Session management: each /ingest call creates a new session_id.
The client sends that session_id with every /chat call.
State is in-process — single worker only (fine for Cloud Run / Render free tier).
"""

import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# session_id → { "chain", "retriever", "answer_chain" }
_sessions: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG Chatbot API ready")
    yield
    _sessions.clear()
    logger.info("Shutting down")


app = FastAPI(
    title="RAG Chatbot API",
    description="LangChain RAG pipeline — upload a PDF and ask questions",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str


class SourceItem(BaseModel):
    page: int
    snippet: str


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float


class GuardrailInfo(BaseModel):
    input_passed: bool
    output_passed: bool
    warning: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    usage: UsageInfo
    guardrail: GuardrailInfo


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int
    session_id: str
    retriever_mode: str


# ── Helpers ──────────────────────────────────────────────────────────────────

def _require_session(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Call /ingest first.")
    return session


def _build_sources(docs) -> list[SourceItem]:
    return [
        SourceItem(page=int(doc.metadata.get("page", 0)), snippet=doc.page_content[:250])
        for doc in docs
    ]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "sessions_active": len(_sessions)}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF — embeds and stores in ChromaDB, returns a session_id."""
    if not (file.filename or "").endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    from .ingestion import ingest_pdf
    from .chain import build_rag_chain

    session_id = str(uuid.uuid4())
    persist_dir = f"/tmp/chroma_{session_id}"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        vectorstore, chunks = ingest_pdf(tmp_path, persist_directory=persist_dir)
        chain, retriever, answer_chain = build_rag_chain(vectorstore, chunks=chunks)
        chunk_count = vectorstore._collection.count()
    finally:
        os.unlink(tmp_path)

    _sessions[session_id] = {
        "chain": chain,
        "retriever": retriever,
        "answer_chain": answer_chain,
    }
    mode = "hybrid (BM25 + dense)"
    logger.info("Session %s: %d chunks from %s — %s", session_id, chunk_count, file.filename, mode)
    return IngestResponse(
        message=f"Indexed {file.filename}",
        chunks_indexed=chunk_count,
        session_id=session_id,
        retriever_mode=mode,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Ask a question. Returns full answer, sources, token usage, and guardrail results.

    Returns 400 if the input guardrail rejects the query (injection / off-topic).
    Returns 200 with guardrail.output_passed=False if the answer fails grounding check.
    """
    session = _require_session(req.session_id)
    retriever = session["retriever"]
    answer_chain = session["answer_chain"]

    from .cost_tracker import CostTracker
    from .chain import format_docs
    from .guardrails import check_input, check_output

    # ── Input guardrail ──────────────────────────────────────────────────────
    input_check = check_input(req.question)
    if not input_check:
        raise HTTPException(
            status_code=400,
            detail={"error": input_check.reason, "guardrail": input_check.guardrail},
        )

    # ── Retrieve + generate ──────────────────────────────────────────────────
    tracker = CostTracker()
    docs = retriever.invoke(req.question)
    context = format_docs(docs)
    answer = answer_chain.invoke(
        {"context": context, "question": req.question},
        config={"callbacks": [tracker]},
    )

    # ── Output guardrail ─────────────────────────────────────────────────────
    output_check = check_output(answer, context)

    return ChatResponse(
        answer=answer,
        sources=_build_sources(docs),
        usage=UsageInfo(
            input_tokens=tracker.summary.input_tokens,
            output_tokens=tracker.summary.output_tokens,
            cost_usd=round(tracker.summary.cost_usd, 6),
        ),
        guardrail=GuardrailInfo(
            input_passed=True,
            output_passed=output_check.passed,
            warning=output_check.reason if not output_check.passed else None,
        ),
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Stream answer tokens via Server-Sent Events.

    Input guardrail fires BEFORE streaming starts — returns 400 on rejection.
    Output guardrail result is included in the final done frame AFTER streaming completes.

    Token chunks:  data: <text>\\n\\n
    Final frame:   event: done\\ndata: {"sources":[...], "usage":{...}, "guardrail":{...}}\\n\\n
    """
    session = _require_session(req.session_id)
    retriever = session["retriever"]
    answer_chain = session["answer_chain"]

    from .cost_tracker import CostTracker
    from .chain import format_docs
    from .guardrails import check_input, check_output

    # Input guardrail fires before we open the stream
    input_check = check_input(req.question)
    if not input_check:
        raise HTTPException(
            status_code=400,
            detail={"error": input_check.reason, "guardrail": input_check.guardrail},
        )

    async def event_generator():
        tracker = CostTracker()
        docs = retriever.invoke(req.question)
        context = format_docs(docs)

        # Stream answer, accumulate full text for grounding check
        full_answer = ""
        async for chunk in answer_chain.astream(
            {"context": context, "question": req.question},
            config={"callbacks": [tracker]},
        ):
            full_answer += chunk
            yield f"data: {chunk}\n\n"

        # Output guardrail runs on the complete answer after streaming
        output_check = check_output(full_answer, context)

        final = {
            "sources": [
                {"page": int(d.metadata.get("page", 0)), "snippet": d.page_content[:250]}
                for d in docs
            ],
            "usage": {
                "input_tokens": tracker.summary.input_tokens,
                "output_tokens": tracker.summary.output_tokens,
                "cost_usd": round(tracker.summary.cost_usd, 6),
            },
            "guardrail": {
                "input_passed": True,
                "output_passed": output_check.passed,
                "warning": output_check.reason if not output_check.passed else None,
            },
        }
        yield f"event: done\ndata: {json.dumps(final)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
