"""
=============================================================
 rag_pipeline.py  –  Core RAG Pipeline
 University Knowledge Base Chatbot
 MSc AI | NLP Module (B9AI006) | Dublin Business School
=============================================================

Architecture Overview:
─────────────────────
  PDF + DOCX Documents
       │
       ▼
  PyPDFLoader / Docx2txtLoader  ──►  RecursiveCharacterTextSplitter
                                              │
                                              ▼
                                   HuggingFaceEmbeddings (local)
                                              │
                                              ▼
                                         FAISS VectorStore
                                              │
                                  ┌───────────┴───────────┐
                                  │  Bi-encoder Retrieval  │
                                  │  (top fetch_k chunks)  │
                                  └───────────┬───────────┘
                                              │
                                  ┌───────────▼───────────┐
                                  │  Cross-Encoder Rerank  │
                                  │  (ms-marco-MiniLM-L6)  │
                                  └───────────┬───────────┘
                                              │
                               User Question  ▼
                                       Top-K Chunks
                                              │
                                              ▼
                                    ChatPromptTemplate
                                              │
                                              ▼
                                   Anthropic Claude Haiku
                                              │
                                              ▼
                               Grounded Answer + Sources
"""

import os
import time
from typing import List, Dict, Optional
from pathlib import Path

# ── LangChain imports ──────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic

# ── Cross-encoder reranker ─────────────────────────────────────────────────────
from sentence_transformers import CrossEncoder


# ══════════════════════════════════════════════════════════════════════════════
# RAGPipeline Class
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    End-to-end RAG pipeline with cross-encoder reranking.

    Responsibilities
    ----------------
    1. Load and chunk PDF and DOCX documents
    2. Embed chunks and store in FAISS
    3. Retrieve candidate chunks with bi-encoder (FAISS)
    4. Rerank candidates with cross-encoder (ms-marco-MiniLM-L6)
    5. Generate a grounded answer using an LLM

    Parameters
    ----------
    anthropic_model : str   – Anthropic Claude model name
    chunk_size      : int   – Character size of each text chunk (800 preserves tables/lists)
    chunk_overlap   : int   – Overlap between consecutive chunks
    top_k           : int   – Final number of chunks passed to the LLM after reranking
    temperature     : float – LLM temperature (0.0 = deterministic)
    use_reranker    : bool  – Enable cross-encoder reranking (recommended: True)
    """

    def __init__(
        self,
        anthropic_model: str = "claude-3-haiku-20240307",
        chunk_size: int = 800,
        chunk_overlap: int = 80,
        top_k: int = 5,
        temperature: float = 0.0,
        use_reranker: bool = True,
    ):
        self.anthropic_model = anthropic_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.fetch_k = top_k * 3   # retrieve 3× more, rerank down to top_k
        self.temperature = temperature
        self.use_reranker = use_reranker
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None

        # ── Embeddings: HuggingFace (free, runs locally – no API key needed) ──
        # all-MiniLM-L6-v2 is a lightweight 22MB model, fast on CPU
        print("🔄 Loading HuggingFace embedding model (first run downloads ~22MB) …")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ── Cross-encoder reranker ─────────────────────────────────────────────
        # Stage 2: scores (query, chunk) pairs directly for better precision.
        # ms-marco-MiniLM-L-6-v2 is a fast, compact model (~23MB).
        if use_reranker:
            print("🔄 Loading cross-encoder reranker (ms-marco-MiniLM-L-6-v2) …")
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("✅ Cross-encoder reranker ready")
        else:
            self.cross_encoder = None

        # ── LLM: Anthropic Claude 3 Haiku (lowest cost / token usage) ─────────
        self.llm = ChatAnthropic(
            model=anthropic_model,
            temperature=temperature,
            max_tokens=1024,
        )
        print(f"✅ LLM       : Anthropic {anthropic_model}")
        print(f"✅ Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local)")
        print(f"✅ Reranker  : {'cross-encoder/ms-marco-MiniLM-L-6-v2' if use_reranker else 'disabled'}")

        # ── Text Splitter ───────────────────────────────────────────────────────
        # chunk_size=800 keeps table rows and bullet lists intact.
        # Overlap=80 ensures context is not lost at boundaries.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Document Loading
    # ══════════════════════════════════════════════════════════════════════════

    def load_pdfs(self, pdf_dir: str) -> List[Document]:
        """
        Load all PDFs from a directory using LangChain's DirectoryLoader.
        Each page becomes a separate Document with metadata:
          - source : file path
          - page   : page number (0-indexed)
        """
        loader = DirectoryLoader(
            pdf_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages from PDFs in '{pdf_dir}'")
        return documents

    def load_single_pdf(self, pdf_path: str) -> List[Document]:
        """Load a single PDF file."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages from '{Path(pdf_path).name}'")
        return documents

    # ══════════════════════════════════════════════════════════════════════════
    # Document Processing
    # ══════════════════════════════════════════════════════════════════════════

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split raw documents into smaller, overlapping chunks.

        Why chunking?
        ─────────────
        LLMs have a context window limit. Chunking ensures we only pass the
        most relevant portion of a document, reducing noise and cost.

        Why overlap?
        ────────────
        Overlap prevents important information from being cut at chunk
        boundaries, improving retrieval recall.
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Split {len(documents)} pages → {len(chunks)} chunks "
              f"(size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    # ══════════════════════════════════════════════════════════════════════════
    # Vector Store
    # ══════════════════════════════════════════════════════════════════════════

    def build_vectorstore(
        self, chunks: List[Document], save_path: str = None
    ) -> FAISS:
        """
        Convert text chunks into vector embeddings and store in FAISS.

        FAISS (Facebook AI Similarity Search):
        ───────────────────────────────────────
        An efficient library for approximate nearest-neighbour search.
        At query time, the user's question is also embedded, and FAISS
        returns the top-K most similar document vectors.

        Parameters
        ----------
        chunks    : List[Document] – Pre-processed text chunks
        save_path : str            – Optional path to persist the index to disk
        """
        print(f"🔄 Building FAISS vector store from {len(chunks)} chunks ...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        if save_path:
            self.vectorstore.save_local(save_path)
            print(f"💾 Vector store saved to '{save_path}'")

        # Create retriever from the vector store
        # Retriever kept for compatibility; answer_with_rag uses
        # vectorstore.similarity_search directly to support reranking.
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.fetch_k},  # fetch_k for reranker candidate pool
        )
        print("✅ Vector store ready!")
        return self.vectorstore

    def load_vectorstore(self, load_path: str) -> None:
        """
        Load a previously saved FAISS index from disk.
        Avoids re-embedding documents every run (saves time and API cost).
        """
        self.vectorstore = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True,   # required for local FAISS files
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.fetch_k},  # fetch_k for reranker candidate pool
        )
        print(f"✅ Vector store loaded from '{load_path}'")

    # ══════════════════════════════════════════════════════════════════════════
    # RAG Answer  (Retrieval + Generation)
    # ══════════════════════════════════════════════════════════════════════════

    def answer_with_rag(self, question: str) -> Dict:
        """
        Answer a question using full RAG pipeline:
          1. Embed the question
          2. Retrieve fetch_k candidate chunks from FAISS (bi-encoder stage)
          3. Rerank candidates with the cross-encoder, keep top_k  (reranker stage)
          4. Build a prompt containing the reranked context
          5. Send to LLM for grounded answer generation

        Why two-stage retrieval?
        ────────────────────────
        The bi-encoder (FAISS) embeds query and chunks independently, so it
        matches on general semantic similarity – "submission deadline" looks
        similar across ALL subject PDFs.
        The cross-encoder reads the (query, chunk) pair together, giving it
        full attention over both texts. It can distinguish "NLP submission
        deadline" context from "ML submission deadline" context, dramatically
        improving subject-specific precision.

        Returns
        -------
        dict with keys: answer, sources, mode, response_time, num_sources
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialised. Call build_vectorstore() first.")

        start_time = time.time()

        # ── Stage 1: Bi-encoder retrieval (broad candidate set) ───────────────
        # Fetch fetch_k = top_k * 3 candidates from FAISS so the reranker has
        # enough candidates to select the truly relevant top_k from.
        candidate_docs: List[Document] = self.vectorstore.similarity_search(
            question, k=self.fetch_k
        )

        # ── Stage 2: Cross-encoder reranking (precision stage) ────────────────
        # The cross-encoder scores every (question, chunk) pair jointly.
        # This is slower than the bi-encoder but far more accurate, and it
        # only runs on fetch_k (~15) pairs, keeping latency acceptable.
        if self.use_reranker and self.cross_encoder is not None:
            pairs = [[question, doc.page_content] for doc in candidate_docs]
            ce_scores = self.cross_encoder.predict(pairs)
            ranked = sorted(
                zip(ce_scores, candidate_docs),
                key=lambda x: x[0],
                reverse=True,
            )
            retrieved_docs = [doc for _, doc in ranked[: self.top_k]]
        else:
            retrieved_docs = candidate_docs[: self.top_k]

        # ── Step 2: Format context string ──────────────────────────────────────
        context = "\n\n---\n\n".join([
            f"[Source: {Path(doc.metadata.get('source', 'Unknown')).name}  |  "
            f"Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ])

        # ── Step 3: Build grounded prompt ──────────────────────────────────────
        # The system message explicitly restricts the LLM to the provided context,
        # which is the primary mechanism for reducing hallucination in RAG.
        rag_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful university knowledge assistant.
Answer the student's question using ONLY the information in the context below.
If the answer is not contained in the context, respond with:
"I could not find this information in the provided university documents."

Do NOT invent or assume any information beyond what is given.
Be concise, accurate, and cite the source where possible.

─── CONTEXT ───────────────────────────────────────────
{context}
────────────────────────────────────────────────────────""",
            ),
            ("human", "{question}"),
        ])

        # ── Step 4: Generate answer ─────────────────────────────────────────────
        chain = rag_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        elapsed = time.time() - start_time

        return {
            "answer": answer,
            "sources": retrieved_docs,
            "mode": "RAG (Retrieval + Generation)",
            "response_time": elapsed,
            "num_sources": len(retrieved_docs),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Convenience: Full Ingest Pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def ingest_and_build(
        self, pdf_dir: str, vectorstore_path: str = "vectorstore"
    ):
        """
        One-shot pipeline: load PDFs → chunk → embed → save FAISS index.
        Run once; afterwards use load_vectorstore() for fast restarts.

        Returns
        -------
        (num_pages, num_chunks) : tuple[int, int]
        """
        print("\n📚 Starting document ingestion pipeline ...")
        print("=" * 55)
        docs = self.load_pdfs(pdf_dir)
        chunks = self.process_documents(docs)
        self.build_vectorstore(chunks, save_path=vectorstore_path)
        print("=" * 55)
        print(f"✅ Knowledge base ready! ({len(docs)} pages → {len(chunks)} chunks)")
        return len(docs), len(chunks)
