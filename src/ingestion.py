"""
PDF ingestion pipeline: load → chunk → embed → store in ChromaDB.

Design decisions:
- chunk_size=1000, overlap=200: good balance for technical docs
- RecursiveCharacterTextSplitter: respects sentence boundaries
- BAAI/bge-small-en-v1.5: 384-dim HuggingFace embeddings, runs locally, no API key needed
"""

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and return list of LangChain Document objects."""
    logger.info(f"Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return BAAI/bge-small-en-v1.5 — 384-dim embeddings, runs locally, no API key."""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def create_vectorstore(
    chunks: List[Document],
    persist_directory: str = "./chroma_db",
) -> Chroma:
    """Embed chunks and store in ChromaDB. Persists to disk to avoid re-embedding."""
    logger.info("Creating embeddings and storing in ChromaDB...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    logger.info(f"Stored {len(chunks)} chunks in ChromaDB at {persist_directory}")
    return vectorstore


def load_existing_vectorstore(persist_directory: str = "./chroma_db") -> Chroma:
    """Load an existing ChromaDB from disk (avoids re-embedding on restart)."""
    embeddings = get_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def ingest_pdf(file_path: str, persist_directory: str = "./chroma_db") -> Chroma:
    """Full ingestion pipeline: PDF → chunks → ChromaDB. Main entry point."""
    docs = load_pdf(file_path)
    chunks = chunk_documents(docs)
    vectorstore = create_vectorstore(chunks, persist_directory)
    return vectorstore
