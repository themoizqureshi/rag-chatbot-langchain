# Architecture

## RAG Pipeline

```mermaid
graph TD
    A[User uploads PDF] --> B[PyPDFLoader]
    B --> C["RecursiveCharacterTextSplitter\nchunk_size=1000, overlap=200"]
    C --> D["Google text-embedding-004\n(free, 768 dims)"]
    D --> E[("ChromaDB\nLocal Vector Store")]

    F[User Question] --> G["Query Embedding\ntext-embedding-004"]
    G --> H["Similarity Search\ntop-k=4 chunks"]
    E --> H
    H --> I[Context Assembly]
    I --> J["ChatPromptTemplate\nSystem + Human messages"]
    J --> K["Gemini 2.0 Flash\ntemp=0"]
    K --> L[Answer with Citations]

    M[LangSmith] -.->|traces| J
    M -.->|traces| K
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/ingestion.py` | PDF loading, chunking, embedding, ChromaDB storage |
| `src/retriever.py` | Similarity search wrappers |
| `src/chain.py` | LCEL RAG chain (retriever → prompt → LLM → parser) |
| `src/utils.py` | Logging setup, directory helpers, API key validation |
| `app.py` | Streamlit UI, session state, PDF upload flow |

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| LLM | Gemini 2.0 Flash | Free tier, 1M token context window, fast |
| Embeddings | text-embedding-004 | Same API key as LLM, 768 dims, strong retrieval quality |
| Vector DB | ChromaDB (local) | Zero setup, persists to disk between sessions |
| Chunking | 1000 chars / 200 overlap | Balances context preservation and retrieval precision |
| Retrieved chunks | k=4 | Good default; increase for multi-part questions |
| Chain style | LCEL | Composable, async-ready, auto-traced by LangSmith |
