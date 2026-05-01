"""Retriever helpers — thin wrappers around ChromaDB similarity search."""

import logging
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def get_retriever(vectorstore: Chroma, k: int = 4):
    """Return a LangChain retriever with similarity search."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_documents(vectorstore: Chroma, query: str, k: int = 4) -> List[Document]:
    """Retrieve the top-k most similar documents for a query."""
    retriever = get_retriever(vectorstore, k)
    docs = retriever.get_relevant_documents(query)
    logger.info(f"Retrieved {len(docs)} documents for: {query[:60]}")
    return docs
