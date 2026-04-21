"""
=============================================================
 ingest.py  –  One-Time Document Ingestion Script
 University Knowledge Base Chatbot
 MSc AI | NLP Module (B9AI006) | Dublin Business School
=============================================================

Run this script ONCE to:
  1. Load all PDF files from the data/ directory
  2. Split them into overlapping chunks
  3. Generate embeddings via OpenAI
  4. Save the FAISS index to disk

After running, the Streamlit app and evaluator can reuse
the saved index without re-embedding every time.

Usage
─────
    python ingest.py
    python ingest.py --pdf_dir data --output vectorstore
    python ingest.py --pdf_dir data --chunk_size 600 --chunk_overlap 60
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into a FAISS vector store."
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="data",
        help="Directory containing PDF files (default: data/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vectorstore",
        help="Output path for the FAISS index (default: vectorstore/)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Character size of each text chunk (default: 500)",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Overlap between consecutive chunks (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate environment ───────────────────────────────────────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("❌  ANTHROPIC_API_KEY not found.")
        print("    Create a .env file with:  ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # ── Validate PDF directory ─────────────────────────────────────────────────
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"❌  PDF directory not found: '{pdf_dir}'")
        print(f"    Create the folder and place your PDFs inside it.")
        sys.exit(1)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        print(f"❌  No PDF files found in '{pdf_dir}'")
        print("    Place at least one .pdf file in the directory.")
        sys.exit(1)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  📥  University KB  –  Document Ingestion")
    print("═" * 55)
    print(f"  PDF directory  : {pdf_dir.resolve()}")
    print(f"  Output path    : {args.output}/")
    print(f"  Chunk size     : {args.chunk_size}")
    print(f"  Chunk overlap  : {args.chunk_overlap}")
    print(f"  LLM model      : claude-3-haiku-20240307")
    print(f"  Embeddings     : sentence-transformers/all-MiniLM-L6-v2 (local)")
    print(f"\n  Found {len(pdf_files)} PDF file(s):")
    for f in pdf_files:
        size_kb = f.stat().st_size / 1024
        print(f"    📄 {f.name}  ({size_kb:.1f} KB)")
    print("═" * 55)

    # ── Import and run pipeline ────────────────────────────────────────────────
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    num_pages, num_chunks = pipeline.ingest_and_build(
        str(pdf_dir), args.output
    )

    print("\n" + "═" * 55)
    print("  ✅  Ingestion Complete")
    print("═" * 55)
    print(f"  Pages processed : {num_pages}")
    print(f"  Chunks created  : {num_chunks}")
    print(f"  Vector store    : {args.output}/")
    print("═" * 55)
    print("\n  Next steps:")
    print("    1. Run the app:        streamlit run app.py")
    print("    2. Run evaluation:     python evaluation.py")
    print()


if __name__ == "__main__":
    main()
