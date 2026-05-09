"""Retriever helpers — dense similarity and hybrid (BM25 + dense) search."""

import logging
from typing import Any, List

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant — higher value reduces impact of outlier ranks


def _rrf_merge(
    bm25_docs: List[Document],
    dense_docs: List[Document],
    k: int,
    bm25_weight: float,
) -> List[Document]:
    """
    Reciprocal Rank Fusion of two ranked document lists.

    Score per doc = Σ weight_i / (RRF_K + rank_i + 1)
    Docs appearing in both lists receive contributions from both retrievers,
    naturally boosting results that keyword AND semantic search agree on.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + bm25_weight / (_RRF_K + rank + 1)
        doc_map[key] = doc

    dense_weight = 1.0 - bm25_weight
    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + dense_weight / (_RRF_K + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:k]]


class HybridRetriever(BaseRetriever):
    """
    BM25 (keyword) + ChromaDB (semantic) retriever fused via Reciprocal Rank Fusion.

    Avoids importing langchain.retrievers.EnsembleRetriever which has a broken
    import chain against langchain_core >= 1.3 (langchain_core.memory removed).
    Inherits BaseRetriever from langchain_core so it works as a LangChain Runnable.
    """

    bm25_retriever: Any
    dense_retriever: Any
    k: int = 4
    bm25_weight: float = 0.5

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        bm25_docs = self.bm25_retriever.invoke(query)
        dense_docs = self.dense_retriever.invoke(query)
        merged = _rrf_merge(bm25_docs, dense_docs, k=self.k, bm25_weight=self.bm25_weight)
        logger.debug("Hybrid RRF: %d BM25 + %d dense → %d merged", len(bm25_docs), len(dense_docs), len(merged))
        return merged


def get_retriever(vectorstore: Chroma, k: int = 4):
    """Dense similarity retriever (ChromaDB). Fallback when chunks are unavailable."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def get_hybrid_retriever(
    vectorstore: Chroma,
    chunks: List[Document],
    k: int = 4,
    bm25_weight: float = 0.5,
) -> HybridRetriever:
    """
    Hybrid retriever: BM25 (keyword) + ChromaDB (semantic) fused via RRF.

    BM25 catches exact-match queries that embedding similarity misses (e.g.
    product codes, names, acronyms). Dense retrieval catches paraphrase and
    semantic queries. Equal 50/50 weighting works well for general documents;
    tune bm25_weight toward 1.0 for keyword-heavy corpora (legal, medical).
    """
    bm25 = BM25Retriever.from_documents(chunks, k=k)
    dense = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    logger.info(
        "Hybrid retriever built (BM25 %.0f%% / dense %.0f%%, k=%d, RRF_K=%d)",
        bm25_weight * 100, (1 - bm25_weight) * 100, k, _RRF_K,
    )
    return HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        k=k,
        bm25_weight=bm25_weight,
    )


def retrieve_documents(vectorstore: Chroma, query: str, k: int = 4) -> List[Document]:
    """Retrieve top-k documents using dense similarity. Kept for backward compatibility."""
    retriever = get_retriever(vectorstore, k)
    docs = retriever.get_relevant_documents(query)
    logger.info("Retrieved %d documents for: %s", len(docs), query[:60])
    return docs
