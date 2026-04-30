# RAG Chatbot — LangChain + ChromaDB + Gemini

> Upload any PDF → ask questions → get answers grounded in your document.  
> Built with LangChain 0.3, ChromaDB, and Google Gemini 2.0 Flash — **100% free to run**.

![Python](https://img.shields.io/badge/python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

## Architecture

```
User uploads PDF → chunk (1000 chars/200 overlap) → embed (text-embedding-004)
                → store in ChromaDB

User question → embed → similarity search (top-k=4) → inject as context
             → Gemini 2.0 Flash → answer with citations
```

See [docs/architecture.md](docs/architecture.md) for the full Mermaid diagram.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot-langchain
cd rag-chatbot-langchain

# Set up environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY (free at https://aistudio.google.com/apikey)

# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Gemini 2.0 Flash | Free, fast, 1M token context window |
| Embeddings | Google text-embedding-004 | Free with same API key, 768 dims |
| Vector DB | ChromaDB | Local, zero setup, persists to disk |
| Framework | LangChain 0.3 (LCEL) | Industry standard, auto-traced |
| UI | Streamlit | Fastest way to ship AI demos |
| Tracing | LangSmith | Free dev tier, visualize every chain step |

## API Keys Required

| Service | Free? | Where to get |
|---------|-------|--------------|
| Google AI Studio | Yes | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| LangSmith | Yes (dev tier) | [smith.langchain.com](https://smith.langchain.com) |

## Running Tests

```bash
pytest tests/ -v
pytest tests/test_ingestion.py -v   # single file
```

## Project Structure

```
rag-chatbot-langchain/
├── src/
│   ├── ingestion.py    # PDF → chunks → ChromaDB
│   ├── retriever.py    # Similarity search helpers
│   ├── chain.py        # LCEL RAG chain (Gemini 2.0 Flash)
│   └── utils.py        # Logging, env validation
├── app.py              # Streamlit UI
├── tests/              # pytest unit tests
└── docs/
    └── architecture.md # Mermaid diagram + design decisions
```

## Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | — |
| Answer Relevancy | — |
| Context Recall | — |

*Run [Project 2 — RAG Evaluation Pipeline](https://github.com/YOUR_USERNAME/rag-evaluation-pipeline) to populate these scores.*

## Lessons Learned

- *Fill in after building: what surprised you, what you'd do differently, what you'd improve.*

---

*Part of the [AI Engineer Portfolio](https://github.com/YOUR_USERNAME) — 5 projects targeting ₹40–50 LPA roles.*
