"""
Streamlit UI for the RAG Chatbot.

Run with: streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.ingestion import ingest_pdf
from src.chain import build_rag_chain, ask
from src.utils import check_api_key

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG Chatbot — Chat with Your PDF")
st.caption("LangChain + ChromaDB + Gemini 2.0 Flash (free) | Traced by LangSmith")

# Validate API key before rendering anything else
try:
    check_api_key()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

# ── Session State ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# ── Sidebar: PDF Upload ──
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and st.session_state.chain is None:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            vectorstore = ingest_pdf(tmp_path)
            chain, _ = build_rag_chain(vectorstore)
            st.session_state.chain = chain
            os.unlink(tmp_path)

        st.success("✅ PDF processed! Start chatting below.")

    if st.session_state.chain:
        if st.button("🗑 Clear & Upload New PDF"):
            st.session_state.messages = []
            st.session_state.chain = None
            st.rerun()

    st.markdown("---")
    st.markdown("**Model:** Gemini 2.0 Flash")
    st.markdown("**Embeddings:** text-embedding-004")
    st.markdown("**Vector DB:** ChromaDB (local)")
    st.markdown("**Tracing:** LangSmith")

# ── Chat Interface ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.chain is None:
        st.error("Please upload a PDF first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(st.session_state.chain, prompt)
                st.markdown(result["answer"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
        })
