"""Tests for the retriever module."""

from unittest.mock import MagicMock
from langchain.schema import Document
from src.retriever import get_retriever, retrieve_documents


def test_get_retriever_configures_similarity_search():
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever

    result = get_retriever(mock_vs, k=4)

    mock_vs.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    assert result == mock_retriever


def test_get_retriever_respects_k_parameter():
    mock_vs = MagicMock()
    get_retriever(mock_vs, k=8)
    mock_vs.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 8},
    )


def test_retrieve_documents_returns_docs():
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    expected_docs = [
        Document(page_content="chunk 1", metadata={"page": 1}),
        Document(page_content="chunk 2", metadata={"page": 2}),
    ]
    mock_retriever.get_relevant_documents.return_value = expected_docs
    mock_vs.as_retriever.return_value = mock_retriever

    docs = retrieve_documents(mock_vs, "what is RAG?")

    assert docs == expected_docs
    mock_retriever.get_relevant_documents.assert_called_once_with("what is RAG?")
