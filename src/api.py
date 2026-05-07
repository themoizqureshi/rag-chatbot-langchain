"""
FastAPI REST API for the RAG chatbot.

Endpoints:
  GET  /health  — Liveness check
  POST /ingest  — Upload and index a PDF into ChromaDB (returns session_id)
  POST /chat    — Ask a question against the indexed document

Session management: each /ingest call creates a new session_id.
The client sends that session_id with every /chat call.
State is in-process — single worker only (fine for Railway/Render free tier).
"""

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# session_id → { "chain": chain, "retriever": retriever }
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
    version="1.0.0",
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


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int
    session_id: str


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
        vectorstore = ingest_pdf(tmp_path, persist_directory=persist_dir)
        chain, retriever = build_rag_chain(vectorstore)
        chunks = vectorstore._collection.count()
    finally:
        os.unlink(tmp_path)

    _sessions[session_id] = {"chain": chain, "retriever": retriever}
    logger.info(f"Session {session_id}: indexed {chunks} chunks from {file.filename}")
    return IngestResponse(message=f"Indexed {file.filename}", chunks_indexed=chunks, session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Ask a question. Requires /ingest to have been called first."""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Call /ingest first.")

    chain = session["chain"]
    retriever = session["retriever"]

    docs = retriever.invoke(req.question)
    answer = chain.invoke(req.question)

    sources = [
        SourceItem(
            page=int(doc.metadata.get("page", 0)),
            snippet=doc.page_content[:250],
        )
        for doc in docs
    ]

    return ChatResponse(answer=answer, sources=sources)
