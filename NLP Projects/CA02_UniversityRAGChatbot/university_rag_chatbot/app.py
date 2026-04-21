"""
=============================================================
 app.py  –  Streamlit UI
 University Knowledge Base Chatbot  (RAG Demo)
 MSc AI | NLP Module (B9AI006) | Dublin Business School
=============================================================

Run with:
    streamlit run app.py
"""

import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

# ── Load .env API keys ─────────────────────────────────────────────────────────
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="University KB Chatbot | RAG Demo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Session State  (persists across Streamlit reruns)
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "messages": [],          # chat history
    "pipeline": None,        # RAGPipeline instance
    "kb_ready": False,       # True once vector store is built
    "doc_stats": {},         # num_pages, num_chunks
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════════════════════════
# ──  SIDEBAR  ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/7/73/Dublin_Business_School_logo.svg/200px-Dublin_Business_School_logo.svg.png", width=140)
    st.title("⚙️ Configuration")

    # ── 1. API Key ──────────────────────────────────────────────────────────
    st.subheader("🔑 Anthropic API Key")
    api_key = st.text_input(
        "Enter your Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Your key is only stored in this session and never logged.",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    # ── 2. Document Upload ──────────────────────────────────────────────────
    st.subheader("📄 Upload University Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs (handbooks, policies, syllabi …)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # ── 5. RAG Hyperparameters ──────────────────────────────────────────────
    with st.expander("🔧 Advanced RAG Settings"):
        chunk_size    = st.slider("Chunk size (characters)", 200, 1200, 500, 50,
                                  help="Larger = more context per chunk; smaller = more precise retrieval.")
        chunk_overlap = st.slider("Chunk overlap",           0,   200,  50, 10,
                                  help="Overlap prevents information loss at chunk boundaries.")
        top_k         = st.slider("Top-K retrieved chunks",  1,    10,   3,  1,
                                  help="How many chunks to retrieve per query.")

    # ── 6. Build Knowledge Base Button ──────────────────────────────────────
    build_disabled = len(uploaded_files) == 0
    if st.button("🏗️  Build Knowledge Base", type="primary", disabled=build_disabled):
        if not api_key:
            st.error("❌ Please enter your OpenAI API key above.")
        else:
            with st.spinner("Processing documents … this may take a minute."):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Save uploaded files to a temp directory
                        for uf in uploaded_files:
                            dest = Path(tmpdir) / uf.name
                            dest.write_bytes(uf.read())

                        # Build pipeline
                        pipeline = RAGPipeline(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            top_k=top_k,
                            temperature=0.0,
                        )
                        num_pages, num_chunks = pipeline.ingest_and_build(tmpdir)

                    st.session_state.pipeline  = pipeline
                    st.session_state.kb_ready  = True
                    st.session_state.doc_stats = {
                        "files":  len(uploaded_files),
                        "pages":  num_pages,
                        "chunks": num_chunks,
                    }
                    st.success(
                        f"✅ Knowledge base built!\n\n"
                        f"📄 {len(uploaded_files)} file(s) | "
                        f"{num_pages} pages | {num_chunks} chunks"
                    )
                except Exception as e:
                    st.error(f"❌ Error building knowledge base: {e}")

    # ── KB status badge ─────────────────────────────────────────────────────
    if st.session_state.kb_ready:
        stats = st.session_state.doc_stats
        st.success(
            f"✅ KB Active | "
            f"{stats.get('files',0)} files | "
            f"{stats.get('pages',0)} pages | "
            f"{stats.get('chunks',0)} chunks"
        )

    st.divider()

    # ── Clear chat ───────────────────────────────────────────────────────────
    if st.button("🗑️  Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ──  MAIN AREA  ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.title("🎓 University Knowledge Base Chatbot")
st.caption(
    "RAG-powered Q&A assistant | MSc Artificial Intelligence | "
    "NLP Module (B9AI006) | Dublin Business School"
)

# ── Info banner ──────────────────────────────────────────────────────────────
if not st.session_state.kb_ready:
    st.info(
        "👈  Upload PDF documents in the sidebar and click **Build Knowledge Base** to start. "
        "Try uploading your module handbook, assignment brief, or university policies."
    )

# ── Architecture diagram (collapsible) ───────────────────────────────────────
with st.expander("📐 System Architecture", expanded=False):
    st.markdown("""
    ```
    User Question
         │
         ▼
    [Embed Question]  ←──  sentence-transformers/all-MiniLM-L6-v2  (local, free)
         │
         ▼
    [FAISS Vector Search]  ──►  Top-K relevant chunks
         │
         ▼
    [Build Prompt]  =  System prompt + Retrieved context + User question
         │
         ▼
    [LLM Generation]  ←──  Anthropic Claude 3 Haiku
         │
         ▼
    Grounded Answer + Source Citations
    ```
    """)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Chat History Display
# ══════════════════════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages (RAG mode)
        if msg["role"] == "assistant":
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 Retrieved Sources  ({len(sources)} chunks)"):
                    for i, doc in enumerate(sources, 1):
                        src  = Path(doc.metadata.get("source", "Unknown")).name
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Chunk {i}**  |  📄 `{src}`  |  Page `{page}`")
                        snippet = (
                            doc.page_content[:400] + " …"
                            if len(doc.page_content) > 400
                            else doc.page_content
                        )
                        st.caption(snippet)
                        if i < len(sources):
                            st.divider()

            # Metrics row
            meta = msg.get("meta", {})
            if meta:
                c1, c2, c3 = st.columns(3)
                c1.metric("Mode",          meta.get("mode",  "—")[:25])
                c2.metric("Response Time", f"{meta.get('time', 0):.2f}s")
                c3.metric("Sources Used",  meta.get("n_src", 0))

# ══════════════════════════════════════════════════════════════════════════════
# Chat Input
# ══════════════════════════════════════════════════════════════════════════════
if user_input := st.chat_input("Ask a question about your university documents …"):

    # ── Guard checks ─────────────────────────────────────────────────────────
    if not st.session_state.kb_ready:
        st.error("Please build the knowledge base first (upload PDFs → click Build Knowledge Base).")
        st.stop()
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    # ── Display user message ─────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Generate response ────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):

            pipeline: RAGPipeline = st.session_state.pipeline  # already built with haiku

            try:
                result = pipeline.answer_with_rag(user_input)
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
                st.stop()

        # ── Display answer ────────────────────────────────────────────────────
        st.markdown(result["answer"])

        # ── Display retrieved sources ─────────────────────────────────────────
        if result["sources"]:
            with st.expander(f"📚 Retrieved Sources  ({len(result['sources'])} chunks)"):
                for i, doc in enumerate(result["sources"], 1):
                    src  = Path(doc.metadata.get("source", "Unknown")).name
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(f"**Chunk {i}**  |  📄 `{src}`  |  Page `{page}`")
                    snippet = (
                        doc.page_content[:400] + " …"
                        if len(doc.page_content) > 400
                        else doc.page_content
                    )
                    st.caption(snippet)
                    if i < len(result["sources"]):
                        st.divider()

        # ── Metrics ───────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode",          result["mode"][:25])
        c2.metric("Response Time", f"{result['response_time']:.2f}s")
        c3.metric("Sources Used",  result["num_sources"])

    # ── Persist to chat history ───────────────────────────────────────────────
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "meta": {
            "mode":  result["mode"],
            "time":  result["response_time"],
            "n_src": result["num_sources"],
        },
    })
