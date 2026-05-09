"""
Streamlit UI for the RAG Chatbot.

Run with: streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.ingestion import ingest_pdf
from src.chain import build_rag_chain, format_docs
from src.cost_tracker import CostTracker
from src.guardrails import check_input, check_output
from src.utils import check_api_key

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG Chatbot — Chat with Your PDF")
st.caption("LangChain + ChromaDB + Gemini 2.0 Flash | Hybrid search (BM25 + dense) | Traced by LangSmith")

try:
    check_api_key()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

# ── Session State ────────────────────────────────────────────────────────────
for key in ("messages", "chain", "retriever", "answer_chain"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "messages" else []

# ── Sidebar: PDF Upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and st.session_state.chain is None:
        with st.spinner("Processing PDF — building hybrid index..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            vectorstore, chunks = ingest_pdf(tmp_path)
            chain, retriever, answer_chain = build_rag_chain(vectorstore, chunks=chunks)
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.answer_chain = answer_chain
            os.unlink(tmp_path)

        st.success(f"✅ Indexed {len(chunks)} chunks. Start chatting!")

    if st.session_state.chain:
        if st.button("🗑 Clear & Upload New PDF"):
            for key in ("messages", "chain", "retriever", "answer_chain"):
                st.session_state[key] = None if key != "messages" else []
            st.rerun()

    st.markdown("---")
    st.markdown("**Model:** Gemini 2.0 Flash")
    st.markdown("**Embeddings:** BAAI/bge-small-en-v1.5 (local)")
    st.markdown("**Retriever:** Hybrid — BM25 + dense (50/50 RRF)")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**Guardrails:** Input (injection + relevance) + Output (grounding)")
    st.markdown("**Tracing:** LangSmith")

# ── Chat history ─────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("usage"):
            st.caption(message["usage"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(f"**Page {src['page']}**")
                    st.text(src["snippet"])

# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.chain is None:
        st.error("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ── Input guardrail ───────────────────────────────────────────────
        input_check = check_input(prompt)
        if not input_check:
            with st.chat_message("assistant"):
                st.warning(f"Query blocked: {input_check.reason}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ {input_check.reason}",
                "usage": "",
                "sources": [],
            })
        else:
            # ── Retrieve → stream → output guardrail ─────────────────────
            docs = st.session_state.retriever.invoke(prompt)
            context = format_docs(docs)
            sources = [
                {"page": int(d.metadata.get("page", 0)), "snippet": d.page_content[:300]}
                for d in docs
            ]

            tracker = CostTracker()
            with st.chat_message("assistant"):
                response = st.write_stream(
                    st.session_state.answer_chain.stream(
                        {"context": context, "question": prompt},
                        config={"callbacks": [tracker]},
                    )
                )

                # Output guardrail — runs on the complete answer
                output_check = check_output(response, context)
                if not output_check:
                    st.warning(f"Grounding warning: {output_check.reason}")

                usage = tracker.summary
                usage_text = (
                    f"Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out"
                    f" | Est. cost: ${usage.cost_usd:.6f}"
                ) if usage.total_tokens > 0 else ""
                if usage_text:
                    st.caption(usage_text)
                if sources:
                    with st.expander("Sources", expanded=False):
                        for src in sources:
                            st.markdown(f"**Page {src['page']}**")
                            st.text(src["snippet"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "usage": usage_text,
                "sources": sources,
            })
