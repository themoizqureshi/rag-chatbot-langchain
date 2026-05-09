"""
Stage 6b — Eval-as-pytest: RAGAS scores as assertions.

Runs RAGAS on 5 sample Q&A pairs and asserts score >= threshold.
If a prompt change causes an eval regression, pytest fails.

Skip in fast CI with: pytest -m "not eval"
Run in quality CI with: pytest tests/test_quality.py -v -m eval

Requires: OPENROUTER_API_KEY or GOOGLE_API_KEY
"""

import os
import textwrap

import pytest
from datasets import Dataset

pytestmark = pytest.mark.eval

THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.65,
    "context_recall": 0.60,
    "context_precision": 0.60,
}

FIXTURE_TEXT = textwrap.dedent("""\
    Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval step
    with a language model generation step to produce grounded answers.

    In a RAG pipeline, the system first retrieves relevant documents from a vector store
    using semantic similarity search. The retrieved documents are then injected into the
    prompt as context, and the LLM generates an answer based solely on that context.

    RAG reduces hallucination by anchoring the model's output in retrieved evidence.
    The main evaluation metrics are faithfulness, answer relevancy, context recall, and
    context precision. These are measured by RAGAS using an LLM judge.

    Chunk size affects retrieval quality. A chunk size of 1000 characters with 200 overlap
    typically gives better context_precision than 500-character chunks.
""")

QA_PAIRS = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "ground_truth": "RAG combines a retrieval step that fetches relevant documents with a language model generation step to produce grounded answers.",
    },
    {
        "question": "What does RAG reduce?",
        "ground_truth": "RAG reduces hallucination by anchoring the model's output in retrieved evidence.",
    },
    {
        "question": "What metrics does RAGAS measure?",
        "ground_truth": "RAGAS measures faithfulness, answer relevancy, context recall, and context precision.",
    },
    {
        "question": "What chunk size is recommended for better context precision?",
        "ground_truth": "A chunk size of 1000 characters with 200 overlap typically gives better context_precision than 500-character chunks.",
    },
    {
        "question": "How does a RAG pipeline work at query time?",
        "ground_truth": "The system retrieves relevant documents via semantic similarity search, injects them into the prompt as context, and the LLM generates an answer based on that context.",
    },
]


@pytest.fixture(scope="module")
def ragas_scores(tmp_path_factory):
    """Build chain from fixture text, run RAGAS, return mean scores."""
    import tempfile
    from langchain_core.documents import Document

    tmp_dir = tmp_path_factory.mktemp("eval_chroma")
    chroma_path = str(tmp_dir)

    # Inline ingestion so this test has no dependency on the API server
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=FIXTURE_TEXT, metadata={"source": "fixture"})]
    chunks = splitter.split_documents(docs)
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    def _get_llm():
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="google/gemini-2.0-flash-001",
                openai_api_key=key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY the context below. If unsure, say so.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt | _get_llm() | StrOutputParser()
    )

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for pair in QA_PAIRS:
        q = pair["question"]
        answer = chain.invoke(q)
        docs = retriever.invoke(q)
        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append([d.page_content for d in docs])
        data["ground_truth"].append(pair["ground_truth"])

    dataset = Dataset.from_dict(data)

    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=_get_llm(),
        embeddings=embeddings,
    )
    df = result.to_pandas()
    return {m: float(df[m].mean()) for m in THRESHOLDS}


def test_faithfulness(ragas_scores):
    score = ragas_scores["faithfulness"]
    threshold = THRESHOLDS["faithfulness"]
    assert score >= threshold, f"Faithfulness {score:.3f} < threshold {threshold} — prompt may be hallucinating"


def test_answer_relevancy(ragas_scores):
    score = ragas_scores["answer_relevancy"]
    threshold = THRESHOLDS["answer_relevancy"]
    assert score >= threshold, f"Answer relevancy {score:.3f} < threshold {threshold}"


def test_context_recall(ragas_scores):
    score = ragas_scores["context_recall"]
    threshold = THRESHOLDS["context_recall"]
    assert score >= threshold, f"Context recall {score:.3f} < threshold {threshold}"


def test_context_precision(ragas_scores):
    score = ragas_scores["context_precision"]
    threshold = THRESHOLDS["context_precision"]
    assert score >= threshold, f"Context precision {score:.3f} < threshold {threshold}"
