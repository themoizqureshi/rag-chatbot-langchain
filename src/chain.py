"""
LangChain RAG chain using LCEL: retriever + prompt + Gemini LLM.

LangSmith auto-traces when LANGCHAIN_TRACING_V2=true in .env.

build_rag_chain returns three objects:
  chain        — full pipeline (retriever → prompt → LLM). Use for non-streaming /chat.
  retriever    — hybrid (BM25 + dense) or dense-only. Invoke separately before streaming.
  answer_chain — just (prompt → LLM → parser). Use for streaming: pass pre-fetched
                 context so you can show sources alongside the streamed answer.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .utils import get_llm
from .retriever import get_retriever, get_hybrid_retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
- Only answer from the given context. Never make up information.
- If the context doesn't contain the answer, say "I don't have enough information in the document to answer this."
- Always cite which part of the document your answer comes from.
- Be concise and clear.

Context:
{context}
"""


def format_docs(docs: List[Document]) -> str:
    """Format retrieved docs into a single context string with source metadata."""
    formatted = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source {i + 1} - Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def _build_prompt_and_llm():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    llm = get_llm(temperature=0)
    return prompt, llm


def build_rag_chain(
    vectorstore: Chroma,
    chunks: Optional[List[Document]] = None,
    k: int = 4,
):
    """
    Build the RAG chain and retriever.

    Args:
        vectorstore: ChromaDB instance (always required).
        chunks:      Raw document chunks from ingestion. When provided, enables
                     hybrid search (BM25 + dense). Without them, falls back to
                     dense-only similarity search.
        k:           Number of documents to retrieve.

    Returns:
        (chain, retriever, answer_chain)

        chain        — full pipeline, takes a question string, for /chat endpoint.
        retriever    — invoke separately to get docs (needed for /chat/stream).
        answer_chain — takes {"context": str, "question": str}, for streaming.
    """
    retriever = (
        get_hybrid_retriever(vectorstore, chunks, k=k)
        if chunks
        else get_retriever(vectorstore, k=k)
    )

    prompt, llm = _build_prompt_and_llm()
    parser = StrOutputParser()

    # answer_chain: takes a dict with pre-fetched context — used for streaming
    # so the caller can run the retriever first and capture docs for source display.
    answer_chain = prompt | llm | parser

    # Full chain: retriever is embedded, for non-streaming use.
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | answer_chain
    )

    mode = "hybrid (BM25 + dense)" if chunks else "dense-only"
    logger.info("RAG chain built — retriever: %s, k=%d", mode, k)
    return chain, retriever, answer_chain


def ask(chain, question: str) -> Dict[str, Any]:
    """Ask a question and return the answer dict. Used by /chat endpoint."""
    logger.info("Question: %s", question[:80])
    answer = chain.invoke(question)
    logger.info("Answer generated (%d chars)", len(answer))
    return {"answer": answer}
