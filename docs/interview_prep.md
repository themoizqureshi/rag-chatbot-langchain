# Interview Prep — RAG Chatbot

> Use this file to prepare for technical interviews. Every answer references specific code in this project.
> 
> **Strategy:** Don't memorize answers. Understand the *why* behind each decision so you can adapt to follow-up questions.

---

## Concept Questions

### Q: Walk me through how RAG works.

**Answer (2-minute version):**

> "RAG has two phases. The first is indexing — happens once when you load the document. I load the PDF with PyPDFLoader, split it into 1000-character chunks with 200-char overlap using RecursiveCharacterTextSplitter, embed each chunk into a 768-dimensional vector using Google's text-embedding-004 model, and store those vectors in ChromaDB on disk.
>
> The second phase is querying — happens every time the user asks a question. I embed the user's question using the same model (critical that it matches), run a cosine similarity search against all stored chunk vectors, retrieve the top-4 most similar chunks, inject those as context into a prompt, and send it to Gemini 2.0 Flash at temperature 0. The LLM generates an answer grounded only in the retrieved context."

**Code reference:** `src/ingestion.py` for Phase 1, `src/chain.py` for Phase 2.

---

### Q: Why chunk the document? Why not embed the whole thing?

> "Two reasons. First, precision: a 100-page PDF embedded as a single vector tells you the document is vaguely related to your question, but not *which part* of it answers it. You'd return the entire document as context, which overflows the LLM's context window. Second, cost: embedding 500 small chunks once and searching fast is far cheaper than feeding the full document into the LLM on every query."

---

### Q: How did you choose chunk size of 1000 characters with 200 overlap?

> "1000 characters is a widely accepted starting point for technical documents — it's roughly 200-250 words, enough to contain a complete idea without being so large that retrieval becomes imprecise. Overlap of 200 characters (20%) preserves context at chunk boundaries — without it, a sentence starting near the end of chunk 1 and finishing at the start of chunk 2 would be split and neither chunk would have the full context.
>
> I'd tune this based on document type: legal documents often do better with 1500+ characters (dense, formal sentences), Q&A-style FAQs do better with 500 (short, self-contained answers). To measure the impact, I'd use RAGAS context_recall — low recall usually means chunks are either too small or too large."

---

### Q: What embedding model did you use and why?

> "Google's text-embedding-004 — 768 dimensions, free with the Gemini API, strong performance on MTEB retrieval benchmarks. The key constraint is that you must use the same model for indexing AND querying. If you embed documents with model A and queries with model B, the vector spaces are different and similarity search produces garbage.
>
> For comparison, OpenAI's text-embedding-3-small outputs 1536 dimensions and costs $0.02/1M tokens. HuggingFace's all-MiniLM-L6-v2 is 384 dimensions and runs fully locally. I chose Google here because it shares the same free API key as the LLM, reducing setup friction."

---

### Q: What is cosine similarity and why use it for vector search?

> "Cosine similarity measures the angle between two vectors — it's 1.0 if they point in exactly the same direction, 0 if they're perpendicular, -1 if opposite. For embeddings, we care about *direction* not *magnitude* — a long document and a short document about the same topic should have similar vectors. Cosine similarity handles this correctly, while Euclidean distance (straight-line distance) is biased by vector magnitude.
>
> In ChromaDB, cosine similarity is computed for every stored vector against your query vector, and the top-k are returned. This is an approximate nearest-neighbor search — it's O(n) for brute force, or ChromaDB uses HNSW indexing for O(log n) at scale."

---

### Q: Why use `temperature=0` for the LLM?

> "Temperature controls the randomness of LLM outputs. At 0, the model always picks the highest-probability token — outputs are deterministic. For factual Q&A grounded in retrieved context, you don't want creativity or variation. The answer should be the same every time you ask the same question about the same document. I'd raise temperature only for tasks where variation is desirable — creative writing, brainstorming, generating multiple alternatives."

---

### Q: What is LCEL and why did you use it?

> "LCEL is LangChain Expression Language — it uses the `|` operator to compose pipeline steps. So `retriever | format_docs | prompt | llm | parser` reads left-to-right like Unix pipes. The benefits over older LangChain chains: it's async-ready by default (each step can be awaited), it supports streaming (tokens stream to the UI as they're generated), it integrates with LangSmith tracing automatically, and it's more composable — you can swap any step without rewriting the whole chain."

---

### Q: How does LangSmith tracing work here?

> "When `LANGCHAIN_TRACING_V2=true` is set in the environment, LangChain automatically sends a trace to LangSmith for every chain invocation — no code changes needed. Each trace shows: the input question, which chunks were retrieved and their similarity scores, the exact prompt sent to Gemini, the LLM's response, latency at each step, and token counts. It's indispensable for debugging retrieval failures — if the answer is wrong, you can check whether the right chunks were retrieved or if the problem is in the prompt/LLM."

---

## Design Decision Questions

### Q: ChromaDB vs Pinecone — when would you use each?

> "ChromaDB is an open-source vector store that runs locally with zero infrastructure setup. It's perfect for development, prototyping, and small datasets (< 1M vectors). Data persists to a local directory. Zero cost, zero ops.
>
> Pinecone is a managed cloud service — it handles replication, scaling, and monitoring for you. You'd use it when: your dataset is too large for a single machine, you need multi-user or multi-tenant access (Pinecone has namespaces), you want high availability SLAs, or your team doesn't want to manage infrastructure.
>
> In my portfolio, I use ChromaDB in Projects 1 and 2 (local dev), and Pinecone in Project 3 specifically to demonstrate I understand the production trade-off."

---

### Q: Why Gemini 2.0 Flash instead of GPT-4o or Claude?

> "Gemini 2.0 Flash has a 1 million token context window — meaning I could technically inject the entire PDF as context without RAG, though I still use RAG for precision and cost. It's on Google's free tier with 15 requests per minute, which is more than enough for development and demos. For production, I'd evaluate on RAGAS metrics — if GPT-4o or Claude scored meaningfully higher on faithfulness and relevancy, the cost increase might be justified. The framework is model-agnostic: swapping Gemini for another LLM is a one-line change in `src/chain.py`."

---

### Q: How would you handle a user asking a question that the document doesn't answer?

> "Two layers of protection. First, the system prompt explicitly tells Gemini: 'If the context doesn't contain the answer, say I don't have enough information.' Second, at the retrieval layer, even 'irrelevant' chunks get returned if there's nothing more relevant — the similarity score tells you how confident the retrieval was. A production improvement would be to threshold the similarity score: if the best match is below 0.5, return a 'no relevant content found' message before even calling the LLM."

---

## Architecture / Scale Questions

### Q: How would you scale this to handle 1000-page documents?

> "A few changes. First, switch ChromaDB to Pinecone or Weaviate for a scalable managed vector store. Second, add a document chunking queue — for large PDFs, ingestion takes minutes, so you'd run it as a background job with status polling rather than blocking the UI. Third, add a chunk count limit per query — with 1000 pages you might have 10,000 chunks, so retrieval stays fast but you may need to tune k higher. Fourth, add metadata filtering — if the document has sections, let users filter retrieval to specific chapters. The core LCEL chain doesn't change."

---

### Q: How would you improve retrieval quality?

> "Several techniques, in order of complexity:
> 1. **Tune chunk size**: Run RAGAS evaluation (Project 2) and compare context_recall across chunk sizes.
> 2. **Hybrid search**: Combine dense vector search (semantic) with BM25 keyword search. Good for documents with specific terms like product names or medical codes that embeddings handle poorly.
> 3. **Re-ranking**: Use a cross-encoder model to re-rank the top-20 retrieved chunks, then pass only the top-4 to the LLM. Cross-encoders are slower but more accurate than bi-encoder embeddings.
> 4. **Query expansion**: Use the LLM to generate 3 paraphrases of the question, retrieve for each, and merge results. Helps with ambiguous questions.
> 5. **Contextual chunk headers**: Prepend each chunk with a summary of the document section it came from (Anthropic's 'contextual retrieval' paper). Improves recall by ~49% in some benchmarks."

---

### Q: What would you monitor in production?

> "Four things: (1) **Faithfulness score** via RAGAS — if it drops, the LLM is hallucinating. (2) **Latency per step** — LangSmith breaks this down; if retrieval slows, the vector index needs optimization. (3) **Token usage** — Gemini's free tier has rate limits; track to plan capacity. (4) **User feedback signals** — thumbs up/down on answers, which questions get follow-ups (indicates the first answer was insufficient). I'd alert if faithfulness drops below 0.70 in a 24h rolling window."

---

## Questions About Your Code Specifically

### Q: Why is there overlap logic in `RecursiveCharacterTextSplitter`? Show me in the code.

**Code:** `src/ingestion.py:34-42`

> "The `chunk_overlap=200` parameter tells the splitter to include the last 200 characters of chunk N at the start of chunk N+1. If a sentence starts on page 3 and finishes on page 4 and those pages happen to be on different chunks, both chunks will contain enough of that sentence to make sense. Without overlap, the boundary chunks would have half-sentences that provide no useful context to the LLM."

---

### Q: Walk me through what happens when a new PDF is uploaded in your Streamlit app.

**Code:** `app.py:40-57`

> "When a file is uploaded, Streamlit gives me a `UploadedFile` object. I can't pass it directly to `PyPDFLoader` because it expects a file path, so I write it to a `tempfile.NamedTemporaryFile` with a `.pdf` suffix. I call `ingest_pdf(tmp_path)` which runs the full load → chunk → embed → store pipeline. Then I call `build_rag_chain(vectorstore)` which sets up the LCEL chain. Both the chain is stored in `st.session_state.chain` — Streamlit reruns on every interaction, so without session state the chain would be lost. Finally I delete the temp file with `os.unlink()` to avoid filling up disk."

---

### Q: Your tests don't make any API calls — how do you test the chain logic without calling Gemini?

**Code:** `tests/test_retriever.py`

> "The ingestion and retriever tests use `unittest.mock.MagicMock` to replace the vectorstore and retriever with mock objects. I assert on what was called and with what arguments, rather than what the LLM returned. This means tests run in milliseconds, cost nothing, and pass in CI without any API keys. The trade-off is that these are unit tests — they don't catch integration failures like an actual embedding model returning unexpected dimensions. That's what RAGAS evaluation in Project 2 catches."

---

## Connecting to Your Production Experience

Always link back to your real work:

> "I built this as a learning project, but at Speridian I've worked with Google Discovery Engine, which is Google's managed RAG-as-a-service product built on similar principles — document ingestion, chunking, embedding, and retrieval, but fully managed. Building this from scratch gave me a much deeper understanding of what's happening under the hood when Discovery Engine processes a query, and what knobs I can tune when retrieval quality is poor."

> "The evaluation piece in Project 2 is something I particularly care about because in production I was responsible for achieving ~99% field-level accuracy on mortgage document extraction. You can't get to 99% without a rigorous eval loop — you need to measure, know what's failing, and iterate. RAGAS is the equivalent framework for RAG systems."
