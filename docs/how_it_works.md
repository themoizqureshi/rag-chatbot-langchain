# How It Works — RAG Chatbot

## The Core Problem RAG Solves

LLMs like Gemini are trained on public internet data up to a cutoff date. They know nothing about:
- Your private PDFs, contracts, manuals
- Internal company documents
- Anything uploaded after their training cutoff

**RAG (Retrieval-Augmented Generation)** solves this by injecting relevant text from your documents directly into the LLM's prompt at query time. The LLM doesn't need to "learn" your document — it just reads the relevant parts before answering.

---

## The Two Phases

```
PHASE 1 — INDEXING (runs once when you upload a PDF)
─────────────────────────────────────────────────────
PDF file → Load pages → Split into chunks → Embed each chunk → Store in ChromaDB

PHASE 2 — QUERYING (runs every time you ask a question)
─────────────────────────────────────────────────────────
Your question → Embed question → Find similar chunks → Build prompt → LLM → Answer
```

---

## Phase 1: Indexing Pipeline

### Step 1 — Load the PDF
**Code:** `src/ingestion.py` → `load_pdf()`

```python
loader = PyPDFLoader(file_path)
documents = loader.load()
```

`PyPDFLoader` reads each page of the PDF and creates a `Document` object with:
- `page_content`: the raw text of that page
- `metadata`: `{"source": "path/to/file.pdf", "page": 0}`

The metadata is critical — it's how we show "Source: Page 3" in answers.

---

### Step 2 — Chunk the Text
**Code:** `src/ingestion.py` → `chunk_documents()`

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""],
)
```

**Why chunking?** You can't embed the entire document as one vector — you'd lose precision. A 100-page PDF embedded as one vector can't tell you *which part* answers your question.

**Why 1000 characters?** It's a good balance:
- Too small (< 200 chars): chunks lose context, answers become fragmented
- Too large (> 2000 chars): retrieval becomes imprecise, you retrieve irrelevant content

**Why 200 character overlap?** Information at the boundary of two chunks would be lost otherwise. A sentence that starts at chunk 1 and ends at chunk 2 would be split — overlap preserves this.

**Why `RecursiveCharacterTextSplitter`?** It tries to split on paragraph breaks first (`\n\n`), then line breaks, then sentences, then words. This preserves natural language boundaries. A `CharacterTextSplitter` would blindly cut at 1000 chars regardless.

---

### Step 3 — Embed Each Chunk
**Code:** `src/ingestion.py` → `get_embeddings()`, `create_vectorstore()`

```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
```

An **embedding** converts text into a list of numbers (a vector). The key property: **semantically similar text produces similar vectors**.

For example:
- "How do I reset my password?" → `[0.12, -0.45, 0.87, ...]` (768 numbers)
- "Steps to change your login credentials" → `[0.11, -0.43, 0.85, ...]` (very close)
- "The stock market rose 2% today" → `[0.89, 0.23, -0.12, ...]` (very different)

Google's `text-embedding-004` outputs **768-dimensional** vectors. We use this same model for both indexing and querying — it's critical they match, or similarity search doesn't work.

---

### Step 4 — Store in ChromaDB
**Code:** `src/ingestion.py` → `create_vectorstore()`

```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
```

ChromaDB stores:
1. The **vectors** (768 floats per chunk) — used for similarity search
2. The **original text** of each chunk — returned with search results
3. The **metadata** (source file, page number) — shown to the user

`persist_directory="./chroma_db"` saves everything to disk. Next time you restart the app, you load the existing ChromaDB instead of re-embedding — that's what `load_existing_vectorstore()` does.

---

## Phase 2: Query Pipeline

### Step 1 — Embed the Question
When you type "What are the main findings?" that text goes through the same `text-embedding-004` model and becomes a 768-dim vector.

### Step 2 — Similarity Search
**Code:** `src/chain.py` → `build_rag_chain()`

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)
```

ChromaDB computes **cosine similarity** between your question's vector and every stored chunk vector. It returns the top-4 most similar chunks. Cosine similarity measures the angle between vectors — vectors pointing in the same direction have similarity ≈ 1.0.

### Step 3 — Build the Prompt
**Code:** `src/chain.py` → `format_docs()`, `SYSTEM_PROMPT`

The 4 retrieved chunks are formatted into a context string:
```
[Source 1 - Page 3]
...chunk text...

---

[Source 2 - Page 7]
...chunk text...
```

This context is injected into the system prompt alongside your question.

### Step 4 — Generate the Answer
**Code:** `src/chain.py` → `build_rag_chain()`

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
```

Gemini 2.0 Flash receives:
- **System**: The 4 context chunks + strict instructions to only answer from context
- **Human**: Your question

`temperature=0` makes the output **deterministic** — the same question gives the same answer. For factual Q&A, you don't want creative variation.

### Step 5 — Parse and Return
`StrOutputParser()` extracts the string content from Gemini's response object. The Streamlit UI displays it in the chat.

---

## The LCEL Chain

**Code:** `src/chain.py` → `build_rag_chain()`

```python
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

LCEL (LangChain Expression Language) uses `|` (pipe) to compose steps. Reading left to right:

1. Input question splits into two parallel paths:
   - `retriever | format_docs`: fetches top-4 chunks and formats them as a string → fills `{context}`
   - `RunnablePassthrough()`: passes the question through unchanged → fills `{question}`
2. Both outputs are injected into `prompt` (the `ChatPromptTemplate`)
3. The filled prompt goes to `llm` (Gemini)
4. `StrOutputParser()` extracts the text

When `LANGCHAIN_TRACING_V2=true` is set in `.env`, every step is automatically sent to LangSmith — you can see exactly what context was retrieved and what the LLM received.

---

## Why This Architecture Works

| Property | How it's achieved |
|----------|------------------|
| Answers are grounded | System prompt explicitly forbids answering outside context |
| Sources are cited | `format_docs()` adds `[Source N - Page X]` labels |
| No re-embedding on restart | ChromaDB persists to `./chroma_db/` on disk |
| Cheap to run | Gemini free tier + no embedding cost per query (only at index time) |
| Debuggable | LangSmith traces show exactly what was retrieved and prompted |

---

## Data Flow Diagram (Text)

```
app.py                     src/ingestion.py              ChromaDB (disk)
  │                              │                              │
  │── uploaded PDF ──────────────► load_pdf()                  │
  │                              │── chunk_documents()          │
  │                              │── create_vectorstore() ──────►
  │                              │                              │
  │── user question ────────────────────────────────────────────►
  │                                                             │── similarity_search(k=4)
  │                                                             │
  │                         src/chain.py                        │
  │                              │◄── 4 chunks ─────────────────┘
  │                              │── format_docs()
  │                              │── prompt.format(context, question)
  │                              │── Gemini 2.0 Flash
  │◄── answer string ────────────┘
```
