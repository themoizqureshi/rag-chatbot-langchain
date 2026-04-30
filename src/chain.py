"""
LangChain RAG chain using LCEL: retriever + prompt + Gemini LLM.

LangSmith auto-traces when LANGCHAIN_TRACING_V2=true in .env.
"""

import logging
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

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


def format_docs(docs) -> str:
    """Format retrieved docs into a single context string with source metadata."""
    formatted = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source {i + 1} - Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore: Chroma, k: int = 4):
    """
    Build a RAG chain using LCEL.

    Returns:
        (chain, retriever) — chain takes a question string, returns answer string
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain built successfully")
    return chain, retriever


def ask(chain, question: str) -> Dict[str, Any]:
    """Ask a question and return the answer dict."""
    logger.info(f"Question: {question}")
    answer = chain.invoke(question)
    logger.info(f"Answer generated ({len(answer)} chars)")
    return {"answer": answer}
